#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formula_program.hpp"
#include "pycanha-core/parameters/formula_shape.hpp"
#include "pycanha-core/parameters/origin.hpp"
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

// The targets of the formulas grouped by what they read, as built by
// Formulas::dependencies(). Each list keeps insertion order and names a
// target once.
struct FormulaDependencies {
    std::unordered_map<std::string, std::vector<Entity>> parameters;
    std::unordered_map<Entity, std::vector<Entity>, Entity::Hash> entities;
    // By table name as written in the formulas.
    std::unordered_map<std::string, std::vector<Entity>> tables;
};

// One formula a generator writes: the target and the expression text.
struct GeneratedFormula {
    Entity target;
    std::string expression;
};

// A formula written once for many formulas, with named slots: `{a}` inside
// an entity (T{a}, GL({a},{b})) takes a node number, `{g}` on its own a
// number. The target must be one entity, e.g. "GL({a},{b})" with the
// expression "{g} * k_al((T{a} + T{b}) / 2)".
struct FormulaTemplate {
    std::string target;
    std::string expression;
};

// The values of a template slot, one per formula.
struct NodeSlot {
    std::string name;
    std::span<const NodeNum> values;
};
struct NumberSlot {
    std::string name;
    std::span<const double> values;
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
    // Not added because the expression was refused or reads an entity that
    // does not exist. A formula already on the target stays.
    std::vector<FailedFormula> failed;
};

// The formulas of a model, in insertion order. Formulas written the same
// way apart from their numbers and node numbers share one parsed and
// lowered shape and are stored together, each with only its own numbers;
// Formula objects are built when one is asked for.
class Formulas {
    struct Batch;

  public:
    // The formulas in insertion order, each built as it is reached.
    class Range {
      public:
        class Iterator {
          public:
            using value_type = std::shared_ptr<Formula>;
            using difference_type = std::ptrdiff_t;

            Iterator() = default;
            Iterator(const Formulas* formulas, std::size_t position)
                : _formulas(formulas), _position(position) {
                skip_removed();
            }

            [[nodiscard]] std::shared_ptr<Formula> operator*() const;
            Iterator& operator++() {
                ++_position;
                skip_removed();
                return *this;
            }
            Iterator operator++(int) {
                auto previous = *this;
                ++*this;
                return previous;
            }
            [[nodiscard]] bool operator==(const Iterator&) const = default;

          private:
            void skip_removed();

            const Formulas* _formulas{nullptr};
            std::size_t _position{0U};
        };

        explicit Range(const Formulas& formulas) : _formulas(&formulas) {}

        [[nodiscard]] Iterator begin() const { return {_formulas, 0U}; }
        [[nodiscard]] Iterator end() const {
            return {_formulas, _formulas->_order.size()};
        }
        [[nodiscard]] std::size_t size() const noexcept {
            return _formulas->size();
        }
        [[nodiscard]] bool empty() const noexcept { return size() == 0U; }

      private:
        const Formulas* _formulas;
    };

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

    // Logs every formula's target and value at INFO as it is applied.
    // Opt-in tracing: INFO so that it also works in builds where DEBUG
    // records are compiled away. Applying formulas one at a time is slower.
    bool debug_formulas{false};

    // Parses the expression and builds its formula, without adding it.
    [[nodiscard]] std::shared_ptr<Formula> create_formula(
        const Entity& entity, const std::string& formula_string);
    // The add_* calls add user formulas and throw std::invalid_argument when
    // the target already has a formula, or for an expression that is
    // refused.
    void add_formula(const Entity& entity, double value);
    void add_formula(std::string_view entity, double value);
    void add_formula(const Entity& entity, const std::string& formula_string);
    void add_formula(std::string_view entity,
                     const std::string& formula_string);
    // Adds a formula that runs `function` at its position in the insertion
    // order and writes the returned value to the target. It counts as
    // reading every entity, so every formula added after it may depend on
    // it.
    void add_callable_formula(const Entity& entity,
                              CallableFormula::Function function);
    void add_callable_formula(std::string_view entity,
                              CallableFormula::Function function);
    // A clone or formula of another origin added here becomes a user
    // formula.
    void add_formula(const Formula& formula);
    void add_formula(const std::shared_ptr<Formula>& formula);
    // Adds one formula per slot value, all of the template's shape. Every
    // formula is checked first; on a problem (a target that already has a
    // formula or appears twice, an entity read that does not exist) nothing
    // is added. Throws std::invalid_argument, also for a malformed call:
    // a slot of the template without values, values for a slot it does not
    // use, or slots of different lengths.
    void add_formulas(const FormulaTemplate& shape,
                      std::span<const NodeSlot> nodes,
                      std::span<const NumberSlot> numbers);
    bool remove_formula(const Entity& entity) noexcept;
    bool remove_formula(std::string_view entity) noexcept;

    // Adds the formulas a generator writes, in order, with their origin
    // (anything but user, which throws std::invalid_argument). On a target
    // that already has a formula:
    // - a user formula stays and the generated one is dropped;
    // - a generated formula, of any origin, is removed and the new one is
    //   added at the end of the list.
    // A formula that is refused is skipped and reported. One WARN per call
    // sums up the collisions and failures.
    GenerationReport add_generated_formulas(
        Origin origin, std::span<const GeneratedFormula> formulas);
    // The same, for one formula per slot value of a template. A malformed
    // call (see add_formulas) throws std::invalid_argument and adds
    // nothing.
    GenerationReport add_generated_formulas(
        Origin origin, const FormulaTemplate& shape,
        std::span<const NodeSlot> nodes, std::span<const NumberSlot> numbers);
    // Removes every formula of `origin` in one pass and returns how many.
    // A regeneration calls it before writing its formulas again.
    std::size_t remove_formulas(Origin origin);

    // Moves the formula of `target` to just before the formula of `before`
    // in the insertion order, or to the end. Returns false, and moves
    // nothing, when either has no formula. O(number of formulas).
    bool move_formula(const Entity& target, const Entity& before);
    bool move_formula(std::string_view target, std::string_view before);
    bool move_formula_to_end(const Entity& target);
    bool move_formula_to_end(std::string_view target);

    // Whether `entity` is the target of a formula. O(1).
    [[nodiscard]] bool contains(const Entity& entity) const noexcept;
    [[nodiscard]] bool contains(std::string_view entity) const;
    // The formula whose target is `entity`, built for this call, or null.
    // O(1).
    [[nodiscard]] std::shared_ptr<Formula> find(const Entity& entity) const;
    [[nodiscard]] std::shared_ptr<Formula> find(std::string_view entity) const;

    // Checks every formula: its target, inputs and called tables must exist.
    // Throws std::runtime_error naming the broken formulas. Does nothing
    // while the validation is current (see is_validation_current).
    void validate_for_execution();
    // Builds the program that evaluates every formula. Formulas of the same
    // shape that run one after another share one run of operations, with
    // their own value pointers and numbers. Formulas that read no formula
    // target, and whose target no earlier formula reads, run first, grouped
    // by shape; the rest keep insertion order. That gives the same results
    // as insertion order. Validates first when needed, and does nothing
    // while the compiled state is current, so a solve on an unchanged model
    // compiles nothing. Call it ahead of a solve to pay the compile cost
    // earlier.
    void compile_formulas();
    // Publishes the tables evaluated at a parameter, compiles if anything
    // changed since the last compile, and runs the program.
    void apply_formulas();
    // Builds the derivatives of every shape that reads only parameters, so
    // that a shape without derivatives is reported now.
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

    // The formulas in insertion order. Each is built when the iteration
    // reaches it; nothing is kept.
    [[nodiscard]] Range formulas() const { return Range(*this); }
    [[nodiscard]] std::size_t size() const noexcept {
        return _positions.size();
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

    // A target as the index stores it; the network is the collection's.
    struct TargetKey {
        NodeNum node_1{0};
        NodeNum node_2{0};
        EntityType type{EntityType::t};

        [[nodiscard]] bool operator==(const TargetKey&) const = default;
        struct Hash {
            [[nodiscard]] std::size_t operator()(
                const TargetKey& key) const noexcept;
        };
    };

    // Formulas of one shape and origin, or one callable formula. A removed
    // formula keeps its place until the batch is packed.
    struct Batch {
        std::shared_ptr<const detail::FormulaShape> shape;
        std::shared_ptr<CallableFormula> callable;
        // Per formula, shape->text.node_slots node numbers and
        // shape->text.numbers.size() numbers.
        std::vector<NodeNum> nodes;
        std::vector<double> numbers;
        std::vector<std::uint8_t> alive;
        std::size_t live{0U};
        Origin origin{Origin::user};

        [[nodiscard]] std::size_t node_count() const noexcept;
        [[nodiscard]] std::size_t number_count() const noexcept;
        [[nodiscard]] std::span<const NodeNum> nodes_of(
            std::size_t instance) const;
        [[nodiscard]] std::span<const double> numbers_of(
            std::size_t instance) const;
    };

    struct BatchKey {
        const detail::FormulaShape* shape{nullptr};
        Origin origin{Origin::user};

        [[nodiscard]] bool operator==(const BatchKey&) const = default;
        struct Hash {
            [[nodiscard]] std::size_t operator()(
                const BatchKey& key) const noexcept;
        };
    };

    // A template call checked and resolved to its shape.
    struct PreparedTemplate;
    class Collisions;

    [[nodiscard]] Entity resolve_entity(std::string_view entity) const;
    [[nodiscard]] std::optional<Entity> try_resolve_entity(
        std::string_view entity) const;
    [[nodiscard]] Stamp current_stamp() const noexcept;
    [[nodiscard]] static TargetKey key_of(const Entity& entity) noexcept;
    [[nodiscard]] Entity target_of(detail::FormulaRef formula) const;
    [[nodiscard]] Entity target_of(const Batch& batch,
                                   std::size_t instance) const;
    [[nodiscard]] std::shared_ptr<Formula> make_formula(
        detail::FormulaRef formula) const;
    [[nodiscard]] bool is_alive(detail::FormulaRef formula) const noexcept;

    // The shape of `text`: the cached one when it was built for the current
    // parameters, else a new one.
    [[nodiscard]] std::shared_ptr<const detail::FormulaShape> shape_for(
        detail::ShapeText text, std::string_view display);
    [[nodiscard]] std::shared_ptr<const detail::FormulaShape> adopt_shape(
        const std::shared_ptr<const detail::FormulaShape>& shape);
    // Why a formula of `shape` with these node numbers cannot be added (an
    // entity it reads does not exist), if it cannot.
    [[nodiscard]] std::optional<std::string> missing_input(
        const detail::FormulaShape& shape,
        std::span<const NodeNum> nodes) const;
    // Makes the load storage a formula writes or reads dense now, so a solve
    // never allocates it.
    void settle_load_storage(const detail::FormulaShape& shape,
                             std::span<const NodeNum> nodes);
    [[nodiscard]] std::uint32_t batch_for(
        const std::shared_ptr<const detail::FormulaShape>& shape,
        Origin origin);
    // Appends a formula; its target must have none.
    void append(std::uint32_t batch, const Entity& target,
                std::span<const NodeNum> nodes,
                std::span<const double> numbers);
    void append_callable(const std::shared_ptr<CallableFormula>& callable,
                         Origin origin);
    void erase(detail::FormulaRef formula) noexcept;
    // The text of `entity = expression` as a formula, checked.
    [[nodiscard]] std::pair<std::shared_ptr<const detail::FormulaShape>,
                            detail::ShapeValues>
    read_formula(const Entity& entity, const std::string& expression);
    void insert_user(const std::shared_ptr<const detail::FormulaShape>& shape,
                     const Entity& target, const detail::ShapeValues& values);
    // Adds a generated formula to `batch`, under the collision rules.
    void insert_generated(std::uint32_t batch, const Entity& target,
                          std::span<const NodeNum> nodes,
                          std::span<const double> numbers,
                          GenerationReport& report, Collisions& collisions);
    [[nodiscard]] PreparedTemplate prepare_template(
        const FormulaTemplate& shape, std::span<const NodeSlot> nodes,
        std::span<const NumberSlot> numbers);
    [[nodiscard]] std::optional<std::size_t> order_position(
        detail::FormulaRef formula) const noexcept;
    [[nodiscard]] std::optional<std::string> problem(
        detail::FormulaRef formula) const;
    void settle_network_storage();
    void compact();
    void pack(std::uint32_t batch, std::vector<std::uint32_t>& moved);
    [[nodiscard]] detail::FormulaPointers collect_pointers(
        std::span<const detail::FormulaRef> formulas) const;
    // Per number slot of the run's shape: the value every formula of the
    // run has there, or none.
    [[nodiscard]] std::vector<std::optional<double>> shared_numbers(
        std::span<const std::size_t> run) const;
    void build_program();
    void run_program();
    void run_program_logged();

    std::shared_ptr<ThermalNetwork> _network;
    std::shared_ptr<Parameters> _parameters;
    DerivativeParameterRegistry _parameters_with_derivatives;
    std::vector<Batch> _batches;
    // Every formula in insertion order. A removed formula stays until the
    // next compile, where the order drops it (see compact()).
    std::vector<detail::FormulaRef> _order;
    std::size_t _removed{0U};
    // Target -> formula.
    std::unordered_map<TargetKey, detail::FormulaRef, TargetKey::Hash>
        _positions;
    // The batch new formulas of a shape and origin go to.
    std::unordered_map<BatchKey, std::uint32_t, BatchKey::Hash> _open;
    // Shapes by key, while a batch uses them.
    std::unordered_map<std::string, std::weak_ptr<const detail::FormulaShape>>
        _shapes;
    // Bumped by every add, removal and move.
    std::uint64_t _formulas_version{0U};
    std::optional<Stamp> _validated;
    std::optional<Stamp> _compiled;
    detail::FormulaProgram _program;
};

}  // namespace pycanha
