#pragma once

#include <symengine/basic.h>
#include <symengine/lambda_double.h>
#include <symengine/symbol.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
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
    // Reads parameters only (including the time and the time variables).
    // These are the formulas with derivatives.
    Parameters,
    // Reads at least one entity (temperature, capacity, load or coupling).
    Entities,
    // Runs a C++ callable, which may read anything: it counts as reading
    // every entity.
    Callable,
};

// A formula sets one target entity. Formulas run in insertion order, so a
// formula may read what an earlier one set.
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

    virtual void compile_formula() = 0;
    virtual void apply_formula() = 0;
    virtual void apply_compiled_formula() = 0;
    virtual void calculate_derivatives() {}

    [[nodiscard]] virtual std::unique_ptr<Formula> clone() const = 0;

    [[nodiscard]] virtual double get_value() const = 0;
    // Null for a formula without derivatives.
    [[nodiscard]] virtual std::vector<double>* get_derivative_values() = 0;

  protected:
    void set_parameter_dependencies(DependencyList dependencies) {
        _dependencies = std::move(dependencies);
    }

    void set_entity_dependencies(std::vector<Entity> dependencies) {
        _entity_dependencies = std::move(dependencies);
    }

    [[nodiscard]] double* entity_data_ptr() const noexcept {
        return _entity_data;
    }
    void set_entity_data_ptr(double* ptr) noexcept { _entity_data = ptr; }

  private:
    Entity _entity;
    DependencyList _dependencies;
    std::vector<Entity> _entity_dependencies;
    double* _entity_data{nullptr};
};

// Any expression of numbers, parameters and, when given a network, entities.
// A constant expression is evaluated once, when the formula is created.
// Derivatives exist for formulas that read only parameters; they are built on
// the first calculate_derivatives() call, from the expression text.
class ExpressionFormula final : public Formula {
  public:
    // A constant value.
    ExpressionFormula(const Entity& entity, double value);
    ExpressionFormula(const Entity& entity, Parameters& parameters,
                      std::string expression,
                      ThermalNetwork* network = nullptr);
    // Same, from an expression already parsed from `expression`.
    ExpressionFormula(const Entity& entity, Parameters& parameters,
                      std::string expression, detail::ParsedExpression parsed,
                      ThermalNetwork* network = nullptr);

    void compile_formula() override;
    void apply_formula() override;
    void apply_compiled_formula() override;
    void calculate_derivatives() override;

    [[nodiscard]] std::unique_ptr<Formula> clone() const override;
    [[nodiscard]] double get_value() const override;
    [[nodiscard]] std::vector<double>* get_derivative_values() override;
    [[nodiscard]] const std::string& expression() const noexcept;

  private:
    enum class BindingKind : std::uint8_t { Parameter, Entity };

    // One input, in symbol order. `slot` indexes parameter_dependencies()
    // and _parameter_indices, or entity_dependencies().
    struct SymbolBinding {
        BindingKind kind{BindingKind::Parameter};
        std::size_t slot{0U};
    };

    // Only sensitivity solves use derivatives, so this is allocated on the
    // first calculate_derivatives() call.
    struct DerivativeState {
        std::vector<SymEngine::RCP<const SymEngine::Basic>> expressions;
        std::vector<SymEngine::LambdaRealDoubleVisitor> compiled;
        std::vector<bool> compiled_ready;
        std::vector<double> values;
    };

    void initialize_expression(detail::ParsedExpression parsed);
    void check_compiled() const;
    [[nodiscard]] std::vector<double> compiled_inputs() const;

    [[nodiscard]] SymEngine::vec_basic lambda_inputs() const;
    [[nodiscard]] double evaluate_expression(
        const SymEngine::RCP<const SymEngine::Basic>& expr) const;
    [[nodiscard]] double evaluate_symbol_value(
        const SymbolBinding& binding) const;
    [[nodiscard]] double* resolve_symbol_ptr(
        const SymbolBinding& binding) const;
    [[nodiscard]] std::string binding_name(const SymbolBinding& binding) const;

    Parameters* _parameters{nullptr};
    ThermalNetwork* _network{nullptr};
    std::string _expression;
    // The constant's value, or the last value computed.
    double _value{0.0};
    std::vector<Index> _parameter_indices;
    std::vector<SymbolBinding> _bindings;

    // Null for a constant, whose value is all that is kept.
    SymEngine::RCP<const SymEngine::Basic> _parsed_expr;
    std::vector<SymEngine::RCP<const SymEngine::Symbol>> _symbols;

    // Built by compile_formula() for a formula with inputs.
    std::unique_ptr<SymEngine::LambdaRealDoubleVisitor> _compiled_eval;
    std::vector<double*> _input_ptrs;
    std::unique_ptr<DerivativeState> _derivatives;
    bool _compiled{false};
    std::uint64_t _compiled_structure_version{0U};
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

    void compile_formula() override;
    void apply_formula() override;
    void apply_compiled_formula() override;

    [[nodiscard]] std::unique_ptr<Formula> clone() const override;
    // The last value written, without calling the function; NaN before the
    // first apply.
    [[nodiscard]] double get_value() const override;
    [[nodiscard]] std::vector<double>* get_derivative_values() override;
    [[nodiscard]] const Function& function() const noexcept;

  private:
    Function _function;
    double _value{std::numeric_limits<double>::quiet_NaN()};
};

}  // namespace pycanha
