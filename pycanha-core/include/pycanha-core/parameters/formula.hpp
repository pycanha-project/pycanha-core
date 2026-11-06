#pragma once

#include <symengine/basic.h>
#include <symengine/symbol.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula_program.hpp"
#include "pycanha-core/parameters/origin.hpp"
#include "pycanha-core/parameters/parameters.hpp"

namespace pycanha {

namespace detail {

// A formula expression after parsing: the expression tree and its free
// symbols, unique and sorted by name. That order is the order of a formula's
// inputs and of its derivative values.
struct ParsedExpression {
    SymEngine::RCP<const SymEngine::Basic> tree;
    std::vector<SymEngine::RCP<const SymEngine::Symbol>> symbols;
};

// Parses a formula expression. Python's `**` is accepted for powers and
// GL(i,j) / GR(i,j) for couplings. Throws std::invalid_argument when the text
// does not parse or uses array access.
[[nodiscard]] ParsedExpression parse_expression(const std::string& expression);

}  // namespace detail

// What a formula reads. It follows from the formula's inputs and decides how
// the formula is evaluated and whether it has derivatives.
enum class FormulaKind : std::uint8_t {
    // Reads nothing: the value is computed once, when the formula is created.
    Constant,
    // Reads parameters only (including the time, the values of tables
    // evaluated at a parameter, and table calls). These are the formulas
    // with derivatives.
    Parameters,
    // Reads at least one entity (temperature, capacity, load or coupling).
    Entities,
    // Runs a C++ callable, which may read anything: it counts as reading
    // every entity.
    Callable,
};

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

    [[nodiscard]] virtual FormulaKind kind() const noexcept;

    // Set by the Formulas collection when the formula is added. Each
    // subclass stores it where its layout has room.
    [[nodiscard]] virtual Origin origin() const noexcept = 0;

    // Names of the parameters the formula reads, in input order.
    [[nodiscard]] const DependencyList& parameter_dependencies()
        const noexcept {
        return _dependencies;
    }

    // Entities (temperatures, capacities, loads, couplings) the formula reads.
    [[nodiscard]] const std::vector<Entity>& entity_dependencies()
        const noexcept {
        return _entity_dependencies;
    }

    // Names of the tables the formula calls.
    [[nodiscard]] virtual const std::vector<std::string>& table_dependencies()
        const noexcept;

    virtual void calculate_derivatives() {}

    [[nodiscard]] virtual std::unique_ptr<Formula> clone() const = 0;

    // The value the formula gives for the current model values.
    [[nodiscard]] virtual double get_value() const = 0;
    // Null for a formula without derivatives.
    [[nodiscard]] virtual std::vector<double>* get_derivative_values() = 0;

  protected:
    // Why the formula cannot run on the current model, if it cannot.
    [[nodiscard]] virtual std::optional<std::string> problem() const;

    void set_parameter_dependencies(DependencyList dependencies) {
        _dependencies = std::move(dependencies);
    }

    void set_entity_dependencies(std::vector<Entity> dependencies) {
        _entity_dependencies = std::move(dependencies);
    }

  private:
    friend class Formulas;

    virtual void set_origin(Origin origin) noexcept = 0;

    Entity _entity;
    DependencyList _dependencies;
    std::vector<Entity> _entity_dependencies;
};

// Any expression of numbers, parameters, table calls and, when given a
// network, entities. It is lowered to evaluator operations once, when it is
// created; the expression text is kept for printing and for derivatives. A
// constant expression is evaluated once and keeps only its value.
// Derivatives exist for formulas that read only parameters; they are built
// on the first calculate_derivatives() call, from the expression text.
class ExpressionFormula final : public Formula {
  public:
    // A constant value.
    ExpressionFormula(const Entity& entity, double value);
    ExpressionFormula(const Entity& entity, Parameters& parameters,
                      std::string expression,
                      ThermalNetwork* network = nullptr);
    // Same, from an expression already parsed from `expression`.
    ExpressionFormula(const Entity& entity, Parameters& parameters,
                      std::string expression,
                      const detail::ParsedExpression& parsed,
                      ThermalNetwork* network = nullptr);

    [[nodiscard]] FormulaKind kind() const noexcept override;
    [[nodiscard]] const std::vector<std::string>& table_dependencies()
        const noexcept override;

    void calculate_derivatives() override;

    [[nodiscard]] std::unique_ptr<Formula> clone() const override;
    [[nodiscard]] double get_value() const override;
    [[nodiscard]] std::vector<double>* get_derivative_values() override;
    [[nodiscard]] const std::string& expression() const noexcept;
    [[nodiscard]] Origin origin() const noexcept override { return _origin; }

  private:
    friend class Formulas;

    void set_origin(Origin origin) noexcept override { _origin = origin; }
    [[nodiscard]] std::optional<std::string> problem() const override;

    enum class BindingKind : std::uint8_t { Parameter, Entity };

    // One input, in symbol order. `slot` indexes parameter_dependencies()
    // or entity_dependencies().
    struct SymbolBinding {
        BindingKind kind{BindingKind::Parameter};
        std::size_t slot{0U};
    };

    // Only sensitivity solves use derivatives, so this is allocated on the
    // first calculate_derivatives() call.
    struct DerivativeState {
        std::vector<detail::LoweredExpression> lowered;
        std::vector<double> values;
    };

    void initialize_expression(const detail::ParsedExpression& parsed);
    [[nodiscard]] std::vector<detail::TableView> table_views(
        const std::vector<std::string>& tables) const;
    [[nodiscard]] std::vector<double> input_values() const;
    [[nodiscard]] double evaluate_symbol_value(
        const SymbolBinding& binding) const;
    [[nodiscard]] std::string binding_name(const SymbolBinding& binding) const;
    // Value pointers of the inputs, in input order; null for an input that
    // no longer exists. Used by the compile, which allocates load storage.
    [[nodiscard]] std::vector<double*> input_pointers() const;
    [[nodiscard]] bool reads_entity(std::size_t input) const noexcept;
    // Null for a constant.
    [[nodiscard]] const detail::LoweredExpression* lowered() const noexcept {
        return _lowered == nullptr ? nullptr : &_lowered->expression;
    }
    // Empty for a constant.
    [[nodiscard]] std::span<const SymbolBinding> bindings() const noexcept;

    Parameters* _parameters{nullptr};
    ThermalNetwork* _network{nullptr};
    std::string _expression;
    // The constant's value.
    double _value{0.0};
    // What a formula with inputs needs to run. It depends only on the
    // expression, so copies of the formula share it, and a constant, which
    // keeps only its value, has none.
    struct Lowered {
        detail::LoweredExpression expression;
        // Parameter inputs by slot, which follows a parameter rename.
        std::vector<Index> parameter_indices;
        std::vector<SymbolBinding> bindings;
    };
    std::shared_ptr<const Lowered> _lowered;
    std::unique_ptr<DerivativeState> _derivatives;
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
