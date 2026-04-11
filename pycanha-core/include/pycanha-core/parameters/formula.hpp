#pragma once

#include <symengine/basic.h>
#include <symengine/lambda_double.h>
#include <symengine/symbol.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/parameters.hpp"

namespace pycanha {

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

    [[nodiscard]] const DependencyList& parameter_dependencies()
        const noexcept {
        return _dependencies;
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

    [[nodiscard]] double* entity_data_ptr() noexcept { return _entity_data; }
    void set_entity_data_ptr(double* ptr) noexcept { _entity_data = ptr; }

  private:
    Entity _entity;
    DependencyList _dependencies;
    double* _entity_data{nullptr};
};

class ParameterFormula final : public Formula {
  public:
    ParameterFormula(Entity entity, Parameters& parameters,
                     std::string parameter_name)
        : Formula(std::move(entity)),
          _parameters(&parameters),
          _parameter_name(std::move(parameter_name)) {
        if (_parameters == nullptr) {
            throw std::invalid_argument(
                "ParameterFormula requires parameter storage");
        }

        const auto parameter_idx = _parameters->get_idx(_parameter_name);
        if (!parameter_idx.has_value()) {
            throw std::invalid_argument(
                "ParameterFormula references unknown parameter '" +
                _parameter_name + "'");
        }

        _parameter_idx = *parameter_idx;
        if (_parameters->get_double_ptr(_parameter_idx) == nullptr) {
            throw std::invalid_argument("ParameterFormula expects parameter '" +
                                        _parameter_name +
                                        "' to hold a double value");
        }

        mutable_dependencies().push_back(_parameter_name);
    }

    void compile_formula() override {
        auto* entity_ptr = entity().get_value_ref();
        if (entity_ptr == nullptr) {
            throw std::runtime_error(
                "ParameterFormula could not obtain entity pointer");
        }
        set_entity_data_ptr(entity_ptr);

        auto* parameter_ptr = _parameters->get_double_ptr(_parameter_idx);
        if (parameter_ptr == nullptr) {
            throw std::runtime_error(
                "ParameterFormula expects parameter to hold a double value");
        }

        _parameter_data = parameter_ptr;
        _compiled_structure_version = _parameters->get_structure_version();
    }

    void apply_formula() override {
        const auto* parameter_value =
            _parameters->get_double_ptr(_parameter_idx);
        if (parameter_value == nullptr) {
            throw std::runtime_error(
                "ParameterFormula expects parameter to hold a double value");
        }
        if (!entity().set_value(*parameter_value)) {
            throw std::runtime_error(
                "ParameterFormula could not assign entity value");
        }
    }

    void apply_compiled_formula() override {
        if (entity_data_ptr() == nullptr || _parameter_data == nullptr) {
            throw std::runtime_error(
                "ParameterFormula needs to be compiled before applying");
        }
        if (_parameters->get_structure_version() !=
            _compiled_structure_version) {
            throw std::runtime_error(
                "ParameterFormula compiled state is stale after structural "
                "parameter changes");
        }
        *entity_data_ptr() = *_parameter_data;
    }

    [[nodiscard]] double get_value() const override {
        const auto* parameter_value =
            _parameters->get_double_ptr(_parameter_idx);
        if (parameter_value == nullptr) {
            throw std::runtime_error(
                "ParameterFormula expects parameter to hold a double value");
        }
        return *parameter_value;
    }

    [[nodiscard]] std::vector<double>* get_derivative_values() override {
        return &_derivatives;
    }

    [[nodiscard]] std::unique_ptr<Formula> clone() const override {
        return std::make_unique<ParameterFormula>(*this);
    }

  private:
    Parameters* _parameters;
    std::string _parameter_name;
    Index _parameter_idx{-1};
    double* _parameter_data{nullptr};
    std::uint64_t _compiled_structure_version{0U};
    std::vector<double> _derivatives{1.0};
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

class ExpressionFormula final : public Formula {
  public:
    ExpressionFormula(Entity entity, Parameters& parameters,
                      std::string expression,
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

    struct SymbolBinding {
        std::string symbol_name;
        BindingKind kind{BindingKind::Parameter};
        Index parameter_idx{-1};
        Entity entity;
    };

    void initialize_expression();

    [[nodiscard]] SymEngine::vec_basic lambda_inputs() const;
    [[nodiscard]] double evaluate_expression(
        const SymEngine::RCP<const SymEngine::Basic>& expr) const;
    [[nodiscard]] double evaluate_symbol_value(
        const SymbolBinding& binding) const;
    [[nodiscard]] double* resolve_symbol_ptr(
        const SymbolBinding& binding) const;

    Parameters* _parameters;
    ThermalNetwork* _network{nullptr};
    std::string _expression;
    std::string _normalized_expression;
    double _cached_value{0.0};
    std::vector<double> _derivatives;
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
