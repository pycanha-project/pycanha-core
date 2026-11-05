#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

    ParameterFormula create_parameter_formula(const Entity& entity,
                                              const std::string& parameter);
    [[nodiscard]] std::shared_ptr<Formula> create_formula(
        Entity entity, const std::string& formula_string);
    [[nodiscard]] ValueFormula& add_value_formula(Entity entity, double value);
    [[nodiscard]] ValueFormula& add_value_formula(std::string_view entity,
                                                  double value);
    [[nodiscard]] ParameterFormula& add_parameter_formula(
        Entity entity, const std::string& expression);
    [[nodiscard]] ParameterFormula& add_parameter_formula(
        std::string_view entity, const std::string& expression);
    [[nodiscard]] ExpressionFormula& add_expression_formula(
        Entity entity, const std::string& expression);
    [[nodiscard]] ExpressionFormula& add_expression_formula(
        std::string_view entity, const std::string& expression);
    [[nodiscard]] Formula& add_formula(const Entity& entity, double value);
    [[nodiscard]] Formula& add_formula(std::string_view entity, double value);
    [[nodiscard]] Formula& add_formula(const Entity& entity,
                                       const std::string& formula_string);
    [[nodiscard]] Formula& add_formula(std::string_view entity,
                                       const std::string& formula_string);
    void set_temperature_variable_names(
        const std::unordered_set<std::string>* names) noexcept;

    void add_formula(const Formula& formula);
    void add_formula(const std::shared_ptr<Formula>& formula);
    bool remove_formula(const Entity& entity) noexcept;
    bool remove_formula(std::string_view entity) noexcept;

    void validate_for_execution();
    void compile_formulas();
    void apply_formulas();
    void apply_compiled_formulas();
    void calculate_derivatives();
    void lock_parameters_for_execution();
    void unlock_parameters() noexcept;
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
    [[nodiscard]] Entity resolve_entity(std::string_view entity) const;
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
    const std::unordered_set<std::string>* _temperature_variable_names{nullptr};
    std::optional<std::uint64_t> _validated_structure_version;
    std::optional<std::uint64_t> _compiled_structure_version;
};

}  // namespace pycanha
