#pragma once

#include <symengine/basic.h>
#include <symengine/lambda_double.h>
#include <symengine/symbol.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
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

    [[nodiscard]] Entity& entity() noexcept { return _entity; }
    [[nodiscard]] const Entity& entity() const noexcept { return _entity; }

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

    void add_parameter_dependency(std::string dependency) {
        _dependencies.push_back(std::move(dependency));
    }

    void set_parameter_dependencies(DependencyList dependencies) {
        _dependencies = std::move(dependencies);
    }

    void clear_parameter_dependencies() noexcept { _dependencies.clear(); }

    virtual void compile_formula() = 0;
    virtual void apply_formula() = 0;
    virtual void apply_compiled_formula() = 0;
    virtual void calculate_derivatives() {}

    [[nodiscard]] virtual std::unique_ptr<Formula> clone() const = 0;

    [[nodiscard]] virtual double get_value() const = 0;
    [[nodiscard]] virtual std::vector<double>* get_derivative_values() = 0;

  protected:
    DependencyList& mutable_dependencies() noexcept { return _dependencies; }

    void set_entity_dependencies(std::vector<Entity> dependencies) {
        _entity_dependencies = std::move(dependencies);
    }

    [[nodiscard]] double* entity_data_ptr() noexcept { return _entity_data; }
    void set_entity_data_ptr(double* ptr) noexcept { _entity_data = ptr; }

  private:
    Entity _entity;
    DependencyList _dependencies;
    std::vector<Entity> _entity_dependencies;
    double* _entity_data{nullptr};
};

// Only sensitivity solves need the derivatives with respect to the inputs, so
// they are built on the first calculate_derivatives() call, from the
// expression text.
class ParameterFormula final : public Formula {
  public:
    ParameterFormula(const Entity& entity, Parameters& parameters,
                     std::string expression);
    // Same, from an expression already parsed from `expression`.
    ParameterFormula(const Entity& entity, Parameters& parameters,
                     std::string expression, detail::ParsedExpression parsed);

    void compile_formula() override;
    void apply_formula() override;
    void apply_compiled_formula() override;
    void calculate_derivatives() override;

    [[nodiscard]] double get_value() const override;
    [[nodiscard]] std::vector<double>* get_derivative_values() override;
    [[nodiscard]] std::unique_ptr<Formula> clone() const override;
    [[nodiscard]] const std::string& expression() const noexcept;

  private:
    void initialize_expression(detail::ParsedExpression parsed);
    [[nodiscard]] SymEngine::vec_basic lambda_inputs() const;
    [[nodiscard]] double evaluate_expression(
        const SymEngine::RCP<const SymEngine::Basic>& expr) const;

    Parameters* _parameters;
    std::string _expression;
    std::vector<Index> _parameter_indices;
    std::vector<SymEngine::RCP<const SymEngine::Symbol>> _symbols;
    SymEngine::RCP<const SymEngine::Basic> _parsed_expr;
    std::vector<SymEngine::RCP<const SymEngine::Basic>> _derivative_exprs;
    SymEngine::LambdaRealDoubleVisitor _compiled_eval;
    std::vector<SymEngine::LambdaRealDoubleVisitor> _compiled_derivs;
    std::vector<double*> _parameter_ptrs;
    std::uint64_t _compiled_structure_version{0U};
    double _cached_value{0.0};
    std::vector<double> _derivatives;
    std::vector<bool> _compiled_derivs_ready;
    bool _compiled{false};
};

class ValueFormula final : public Formula {
  public:
    explicit ValueFormula(Entity entity)
        : Formula(std::move(entity)), _value(this->entity().get_value()) {}

    void compile_formula() override {
        auto* entity_ptr = entity().get_value_ref();
        if (entity_ptr == nullptr) {
            throw std::runtime_error(
                "ValueFormula could not obtain entity pointer");
        }
        set_entity_data_ptr(entity_ptr);
    }

    void apply_formula() override {
        if (!entity().set_value(_value)) {
            throw std::runtime_error("ValueFormula could not assign entity");
        }
    }

    void apply_compiled_formula() override {
        if (entity_data_ptr() == nullptr) {
            throw std::runtime_error(
                "ValueFormula needs to be compiled before applying");
        }
        *entity_data_ptr() = _value;
    }

    [[nodiscard]] double get_value() const override { return _value; }

    [[nodiscard]] std::vector<double>* get_derivative_values() override {
        return &_derivatives;
    }

    [[nodiscard]] const std::vector<double>& derivative_values()
        const noexcept {
        return _derivatives;
    }

    void add_derivative_value(double value) { _derivatives.push_back(value); }

    void set_derivative_values(std::vector<double> values) {
        _derivatives = std::move(values);
    }

    void clear_derivative_values() noexcept { _derivatives.clear(); }

    [[nodiscard]] std::unique_ptr<Formula> clone() const override {
        return std::make_unique<ValueFormula>(*this);
    }

    void set_value(double value) noexcept { _value = value; }

  private:
    double _value{0.0};
    std::vector<double> _derivatives;
};

// Reads parameters and, when given a network, entities. Derivatives exist
// only for expressions without entity inputs, and are built on the first
// calculate_derivatives() call, from the expression text.
class ExpressionFormula final : public Formula {
  public:
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

    void initialize_expression(detail::ParsedExpression parsed);

    [[nodiscard]] SymEngine::vec_basic lambda_inputs() const;
    [[nodiscard]] double evaluate_expression(
        const SymEngine::RCP<const SymEngine::Basic>& expr) const;
    [[nodiscard]] double evaluate_symbol_value(
        const SymbolBinding& binding) const;
    [[nodiscard]] double* resolve_symbol_ptr(
        const SymbolBinding& binding) const;
    [[nodiscard]] std::string binding_name(const SymbolBinding& binding) const;

    Parameters* _parameters;
    ThermalNetwork* _network{nullptr};
    std::string _expression;
    double _cached_value{0.0};
    std::vector<double> _derivatives;
    std::vector<Index> _parameter_indices;
    std::vector<SymbolBinding> _bindings;

    SymEngine::RCP<const SymEngine::Basic> _parsed_expr;
    std::vector<SymEngine::RCP<const SymEngine::Symbol>> _symbols;
    std::vector<SymEngine::RCP<const SymEngine::Basic>> _derivative_exprs;

    SymEngine::LambdaRealDoubleVisitor _compiled_eval;
    std::vector<SymEngine::LambdaRealDoubleVisitor> _compiled_derivs;

    std::vector<double*> _param_ptrs;
    std::vector<bool> _compiled_derivs_ready;
    bool _has_entity_dependencies{false};
    bool _compiled{false};
    std::uint64_t _compiled_structure_version{0U};
};

}  // namespace pycanha
