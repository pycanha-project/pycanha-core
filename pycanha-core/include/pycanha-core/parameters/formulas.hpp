#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha {

class DerivativeParameterRegistry {
  public:
    DerivativeParameterRegistry() = default;
    explicit DerivativeParameterRegistry(std::shared_ptr<Parameters> parameters)
        : _parameters(std::move(parameters)) {}

    void associate(std::shared_ptr<Parameters> parameters) noexcept {
        _parameters = std::move(parameters);
        _parameter_names.clear();
    }

    void add_parameter(const std::string& parameter_name);
    bool remove_parameter(const std::string& parameter_name) noexcept;
    [[nodiscard]] bool contains(
        const std::string& parameter_name) const noexcept;
    [[nodiscard]] const std::vector<std::string>& parameter_names()
        const noexcept {
        return _parameter_names;
    }

  private:
    std::shared_ptr<Parameters> _parameters;
    std::vector<std::string> _parameter_names;
};

// The formulas grouped by what they read, as built by
// Formulas::dependencies(). Each list keeps insertion order and names a
// formula once.
struct FormulaDependencies {
    std::unordered_map<std::string, std::vector<std::shared_ptr<Formula>>>
        parameters;
    std::unordered_map<Entity, std::vector<std::shared_ptr<Formula>>,
                       Entity::Hash>
        entities;
};

// One formula a generator writes: the target and the expression text.
struct GeneratedFormula {
    Entity target;
    std::string expression;
};

// A generated formula that was not added, and why.
struct FailedFormula {
    Entity target;
    std::string message;
};

// What Formulas::add_generated_formulas did with each formula it was given.
struct GenerationReport {
    // Added where the target had no formula.
    std::size_t added{0U};
    // Added in place of a generated formula (of any origin) on the same
    // target, which was removed.
    std::size_t replaced{0U};
    // Not added because the target already had a user formula, which stays.
    std::size_t kept_user{0U};
    // Not added because the expression was refused. A formula already on the
    // target stays.
    std::vector<FailedFormula> failed;
};

class Formulas {
  public:
    Formulas();
    Formulas(std::shared_ptr<ThermalNetwork> network,
             std::shared_ptr<Parameters> parameters);

    Formulas(const Formulas&) = delete;
    Formulas& operator=(const Formulas&) = delete;
    Formulas(Formulas&&) noexcept = default;
    Formulas& operator=(Formulas&&) noexcept = default;
    ~Formulas() = default;

    void associate(std::shared_ptr<ThermalNetwork> network,
                   std::shared_ptr<Parameters> parameters);

    [[nodiscard]] std::shared_ptr<ThermalNetwork> network() const noexcept {
        return _network;
    }
    [[nodiscard]] std::shared_ptr<Parameters> parameters() const noexcept {
        return _parameters;
    }

    [[nodiscard]] DerivativeParameterRegistry&
    parameters_with_derivatives() noexcept {
        return _parameters_with_derivatives;
    }
    [[nodiscard]] const DerivativeParameterRegistry&
    parameters_with_derivatives() const noexcept {
        return _parameters_with_derivatives;
    }

    bool debug_formulas{false};

    // Parses the expression and builds its formula, without adding it.
    [[nodiscard]] std::shared_ptr<Formula> create_formula(
        Entity entity, const std::string& formula_string);
    [[nodiscard]] Formula& add_formula(const Entity& entity, double value);
    [[nodiscard]] Formula& add_formula(std::string_view entity, double value);
    [[nodiscard]] Formula& add_formula(const Entity& entity,
                                       const std::string& formula_string);
    [[nodiscard]] Formula& add_formula(std::string_view entity,
                                       const std::string& formula_string);
    // Adds a formula that runs `function` at its position in the insertion
    // order and writes the returned value to the target. It counts as
    // reading every entity, so every formula added after it may depend on
    // it.
    [[nodiscard]] CallableFormula& add_callable_formula(
        const Entity& entity, CallableFormula::Function function);
    [[nodiscard]] CallableFormula& add_callable_formula(
        std::string_view entity, CallableFormula::Function function);
    void set_temperature_variable_names(
        const std::unordered_set<std::string>* names) noexcept;

    // Both add a user formula; a clone or formula of another origin added
    // here becomes a user formula.
    void add_formula(const Formula& formula);
    void add_formula(const std::shared_ptr<Formula>& formula);
    bool remove_formula(const Entity& entity) noexcept;
    bool remove_formula(std::string_view entity) noexcept;

    // Adds the formulas a generator writes, in order, with their origin
    // (anything but user, which throws std::invalid_argument). On a target
    // that already has a formula:
    // - a user formula stays and the generated one is dropped;
    // - a generated formula, of any origin, is removed and the new one is
    //   added at the end of the list.
    // An expression that does not parse is skipped and reported. One WARN
    // per call sums up the collisions and failures.
    GenerationReport add_generated_formulas(
        FormulaOrigin origin, std::span<const GeneratedFormula> formulas);
    // Removes every formula of `origin` in one pass and returns how many.
    // A regeneration calls it before writing its formulas again.
    std::size_t remove_formulas(FormulaOrigin origin);

    // Whether `entity` is the target of a formula. O(1).
    [[nodiscard]] bool contains(const Entity& entity) const noexcept;
    [[nodiscard]] bool contains(std::string_view entity) const;
    // The formula whose target is `entity`, or null. O(1).
    [[nodiscard]] std::shared_ptr<Formula> find(const Entity& entity) const;
    [[nodiscard]] std::shared_ptr<Formula> find(std::string_view entity) const;

    // Checks every formula: its target and inputs must exist and it must
    // evaluate. Throws std::runtime_error naming the broken formulas. Does
    // nothing while the validation is current (see is_validation_current).
    void validate_for_execution();
    // Takes the value pointers of every target and input. Validates first
    // when needed, and does nothing while the compiled state is current, so a
    // solve on an unchanged model compiles nothing. Call it ahead of a solve
    // to pay the compile cost earlier.
    void compile_formulas();
    void apply_formulas();
    void apply_compiled_formulas();
    // Derivatives of the formulas that read only parameters; the others have
    // none.
    void calculate_derivatives();
    void lock_parameters_for_execution();
    void unlock_parameters() noexcept;
    // Current while nothing the formulas point into changed since the last
    // validation or compile: the formulas themselves, the parameter
    // structure, and the node and coupling structures of the network. A
    // node removed, or a node or coupling added, makes them stale; values
    // can change freely.
    [[nodiscard]] bool is_validation_current() const noexcept;
    [[nodiscard]] bool is_compiled_current() const noexcept;

    // The formulas in insertion order.
    [[nodiscard]] const std::vector<std::shared_ptr<Formula>>& formulas()
        const {
        compact();
        return _formulas;
    }

    // Built on each call by one pass over the formulas.
    [[nodiscard]] FormulaDependencies dependencies() const;

  private:
    // The counters a validation or compile was done against.
    struct Stamp {
        std::uint64_t formulas{0U};
        std::uint64_t parameters{0U};
        StructureVersions network;

        [[nodiscard]] bool operator==(const Stamp&) const = default;
    };

    [[nodiscard]] Entity resolve_entity(std::string_view entity) const;
    [[nodiscard]] std::optional<Entity> try_resolve_entity(
        std::string_view entity) const;
    [[nodiscard]] Stamp current_stamp() const noexcept;
    void insert(const std::shared_ptr<Formula>& formula, FormulaOrigin origin);
    void settle_network_storage();
    void compact() const;

    std::shared_ptr<ThermalNetwork> _network;
    std::shared_ptr<Parameters> _parameters;
    DerivativeParameterRegistry _parameters_with_derivatives;
    // Formulas in insertion order. A removal leaves a null hole, so it costs
    // O(1); compact() drops the holes in one pass before the list is read,
    // validated, compiled or applied. Mutable so that a const read can do it.
    mutable std::vector<std::shared_ptr<Formula>> _formulas;
    // Target entity -> position in _formulas.
    mutable std::unordered_map<Entity, std::size_t, Entity::Hash> _positions;
    mutable std::size_t _holes{0U};
    // Bumped by every add and removal.
    std::uint64_t _formulas_version{0U};
    const std::unordered_set<std::string>* _temperature_variable_names{nullptr};
    std::optional<Stamp> _validated;
    std::optional<Stamp> _compiled;
};

}  // namespace pycanha
