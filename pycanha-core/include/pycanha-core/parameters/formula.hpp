#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula_kind.hpp"  // IWYU pragma: export
#include "pycanha-core/parameters/formula_shape.hpp"
#include "pycanha-core/parameters/origin.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/parameters/table.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha {

class Formulas;

// A formula sets one target entity. Formulas run in insertion order, so a
// formula may read what an earlier one set. They are evaluated by the
// Formulas collection, which compiles all of them into one program.
class Formula {
  public:
    using DependencyList = std::vector<std::string>;

    Formula(const Formula&) = default;
    Formula& operator=(const Formula&) = default;
    Formula(Formula&&) noexcept = default;
    Formula& operator=(Formula&&) noexcept = default;
    virtual ~Formula() = default;

    explicit Formula(Entity entity) : _entity(std::move(entity)) {}

    bool operator==(const Formula& other) const {
        return entity().is_same_as(other.entity());
    }

    struct Hash {
        [[nodiscard]] std::size_t operator()(const Formula& formula) const {
            return Entity::Hash{}(formula.entity());
        }
    };

    // The target. Read-only: the collection indexes formulas by target.
    [[nodiscard]] const Entity& entity() const noexcept { return _entity; }

    [[nodiscard]] virtual FormulaKind kind() const noexcept = 0;

    // Set by the Formulas collection when the formula is added.
    [[nodiscard]] virtual Origin origin() const noexcept = 0;

    // Names of the parameters the formula reads, in input order.
    [[nodiscard]] virtual const DependencyList& parameter_dependencies()
        const noexcept;

    // Entities (temperatures, capacities, loads, couplings) the formula reads.
    [[nodiscard]] virtual const std::vector<Entity>& entity_dependencies()
        const noexcept;

    // Names of the tables the formula calls.
    [[nodiscard]] virtual const std::vector<std::string>& table_dependencies()
        const noexcept;

    virtual void calculate_derivatives() {}

    [[nodiscard]] virtual std::unique_ptr<Formula> clone() const = 0;

    // The value the formula gives for the current model values.
    [[nodiscard]] virtual double get_value() const = 0;
    // Null for a formula without derivatives.
    [[nodiscard]] virtual std::vector<double>* get_derivative_values() = 0;

  private:
    friend class Formulas;

    virtual void set_origin(Origin origin) noexcept = 0;

    Entity _entity;
};

// Any expression of numbers, parameters, table calls and, when given a
// network, entities. Formulas written the same way apart from their numbers
// and node numbers share one shape, parsed and lowered to evaluator
// operations once; a formula keeps only its own numbers and node numbers.
// Its expression is printed from them on request. Derivatives exist for
// formulas that read only parameters; they are built on the first
// calculate_derivatives() call of any formula of the shape.
class ExpressionFormula final : public Formula {
    // Restricts the shape constructor to the Formulas collection.
    class Key {
        Key() = default;
        friend class ExpressionFormula;
        friend class Formulas;
    };

  public:
    // A constant value.
    ExpressionFormula(const Entity& entity, double value);
    // Throws std::invalid_argument for an expression that does not parse,
    // reads an unknown name or calls an unknown function.
    ExpressionFormula(const Entity& entity, Parameters& parameters,
                      const std::string& expression,
                      ThermalNetwork* network = nullptr);
    // A formula of an existing shape, as the Formulas collection hands it
    // out.
    ExpressionFormula(Key /*key*/, const Entity& entity,
                      std::shared_ptr<const detail::FormulaShape> shape,
                      detail::ShapeValues values, Parameters* parameters,
                      ThermalNetwork* network, Origin origin);

    [[nodiscard]] FormulaKind kind() const noexcept override;
    [[nodiscard]] const DependencyList& parameter_dependencies()
        const noexcept override;
    [[nodiscard]] const std::vector<Entity>& entity_dependencies()
        const noexcept override;
    [[nodiscard]] const std::vector<std::string>& table_dependencies()
        const noexcept override;

    void calculate_derivatives() override;

    [[nodiscard]] std::unique_ptr<Formula> clone() const override;
    [[nodiscard]] double get_value() const override;
    [[nodiscard]] std::vector<double>* get_derivative_values() override;
    // Printed from the shape and the formula's numbers on the first call.
    [[nodiscard]] const std::string& expression() const;
    [[nodiscard]] Origin origin() const noexcept override { return _origin; }

  private:
    friend class Formulas;

    void set_origin(Origin origin) noexcept override { _origin = origin; }
    void resolve_inputs();
    // Throws std::invalid_argument when an entity read does not exist.
    void check_inputs() const;
    [[nodiscard]] std::vector<double> input_values() const;
    [[nodiscard]] std::vector<detail::TableView> table_views(
        const std::vector<std::string>& tables) const;

    Parameters* _parameters{nullptr};
    ThermalNetwork* _network{nullptr};
    std::shared_ptr<const detail::FormulaShape> _shape;
    detail::ShapeValues _values;
    // The entities read, in the order of the shape's entity inputs.
    std::vector<Entity> _inputs;
    mutable std::string _expression;
    std::unique_ptr<std::vector<double>> _derivatives;
    Origin _origin{Origin::user};
};

// Runs a C++ callable and writes the value it returns to the target. It runs
// on every apply, at its position in the insertion order, like any other
// formula. What it reads cannot be known, so it counts as reading every
// entity: every formula after it may depend on its result. It lists no
// dependencies and has no derivatives.
class CallableFormula final : public Formula {
  public:
    using Function = std::function<double()>;

    // Throws std::invalid_argument for an empty function.
    CallableFormula(const Entity& entity, Function function);

    [[nodiscard]] FormulaKind kind() const noexcept override {
        return FormulaKind::Callable;
    }

    [[nodiscard]] std::unique_ptr<Formula> clone() const override;
    // The last value written, without calling the function; NaN before the
    // first apply.
    [[nodiscard]] double get_value() const override;
    [[nodiscard]] std::vector<double>* get_derivative_values() override;
    [[nodiscard]] const Function& function() const noexcept;
    [[nodiscard]] Origin origin() const noexcept override { return _origin; }

  private:
    friend class Formulas;

    void set_origin(Origin origin) noexcept override { _origin = origin; }
    // Calls the function, keeps the value and returns it.
    double call() {
        _value = _function();
        return _value;
    }

    Function _function;
    double _value{std::numeric_limits<double>::quiet_NaN()};
    Origin _origin{Origin::user};
};

}  // namespace pycanha
