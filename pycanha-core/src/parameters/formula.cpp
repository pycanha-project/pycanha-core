#include "pycanha-core/parameters/formula.hpp"

#include <symengine/derivative.h>
#include <symengine/eval_double.h>
#include <symengine/parser.h>
#include <symengine/real_double.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace pycanha {

namespace {

[[nodiscard]] std::string replace_python_power(std::string expression) {
    std::size_t position = 0U;
    while ((position = expression.find("**", position)) != std::string::npos) {
        expression.replace(position, 2U, "^");
        position += 1U;
    }
    return expression;
}

[[nodiscard]] std::string preprocess_expression(const std::string& expression) {
    if ((expression.find('[') != std::string::npos) ||
        (expression.find(']') != std::string::npos)) {
        throw std::invalid_argument(
            "ExpressionFormula does not support matrix or array access yet");
    }

    return replace_python_power(expression);
}

void collect_symbols(
    const SymEngine::RCP<const SymEngine::Basic>& expr,
    std::map<std::string, SymEngine::RCP<const SymEngine::Symbol>>& symbols) {
    if (auto* symbol = dynamic_cast<const SymEngine::Symbol*>(expr.get());
        symbol != nullptr) {
        symbols.emplace(symbol->get_name(),
                        SymEngine::symbol(symbol->get_name()));
        return;
    }

    for (const auto& argument : expr->get_args()) {
        collect_symbols(argument, symbols);
    }
}

[[nodiscard]] const double& expect_scalar_parameter(
    const Parameters::ThermalValue& value, const std::string& name) {
    const auto* scalar = std::get_if<double>(&value);
    if (scalar == nullptr) {
        throw std::runtime_error("ExpressionFormula expects parameter '" +
                                 name + "' to store a double");
    }
    return *scalar;
}

}  // namespace

ExpressionFormula::ExpressionFormula(ThermalEntity& entity,
                                     Parameters& parameters,
                                     std::string expression)
    : Formula(entity),
      _parameters(&parameters),
      _expression(std::move(expression)) {
    initialize_expression();
}

ExpressionFormula::ExpressionFormula(std::shared_ptr<ThermalEntity> entity,
                                     Parameters& parameters,
                                     std::string expression)
    : Formula(std::move(entity)),
      _parameters(&parameters),
      _expression(std::move(expression)) {
    initialize_expression();
}

void ExpressionFormula::initialize_expression() {
    if (_parameters == nullptr) {
        throw std::invalid_argument(
            "ExpressionFormula requires parameter storage");
    }

    _normalized_expression = preprocess_expression(_expression);

    try {
        _parsed_expr = SymEngine::parse(_normalized_expression);
    } catch (const std::exception& exception) {
        throw std::invalid_argument("ExpressionFormula could not parse '" +
                                    _expression + "': " + exception.what());
    }

    std::map<std::string, SymEngine::RCP<const SymEngine::Symbol>> symbols;
    collect_symbols(_parsed_expr, symbols);

    _symbols.clear();
    _bindings.clear();
    Formula::DependencyList dependencies;

    for (const auto& [symbol_name, symbol] : symbols) {
        ParameterBinding binding;
        binding.dependency_name = symbol_name;
        binding.parameter_name = symbol_name;

        if (!_parameters->contains(binding.parameter_name)) {
            throw std::invalid_argument(
                "ExpressionFormula references unknown parameter '" +
                binding.parameter_name + "'");
        }

        const auto value = _parameters->get_parameter(binding.parameter_name);
        static_cast<void>(
            expect_scalar_parameter(value, binding.parameter_name));

        _symbols.push_back(symbol);
        dependencies.push_back(binding.dependency_name);
        _bindings.push_back(std::move(binding));
    }

    set_parameter_dependencies(std::move(dependencies));

    _derivative_exprs.clear();
    _derivative_exprs.reserve(_symbols.size());
    for (const auto& symbol : _symbols) {
        _derivative_exprs.push_back(SymEngine::diff(_parsed_expr, symbol));
    }

    _derivatives.assign(_symbols.size(), 0.0);
    _param_ptrs.clear();
    _compiled_derivs.clear();
    _compiled_derivs_ready.clear();
    _compiled = false;
}

SymEngine::vec_basic ExpressionFormula::lambda_inputs() const {
    SymEngine::vec_basic inputs;
    inputs.reserve(_symbols.size());
    for (const auto& symbol : _symbols) {
        inputs.push_back(symbol);
    }
    return inputs;
}

double ExpressionFormula::evaluate_symbol_value(
    const ParameterBinding& binding) const {
    if ((_parameters == nullptr) ||
        !_parameters->contains(binding.parameter_name)) {
        throw std::runtime_error("ExpressionFormula lost parameter '" +
                                 binding.parameter_name + "'");
    }

    const auto value = _parameters->get_parameter(binding.parameter_name);
    return expect_scalar_parameter(value, binding.parameter_name);
}

double* ExpressionFormula::resolve_symbol_ptr(
    const ParameterBinding& binding) const {
    if ((_parameters == nullptr) ||
        !_parameters->contains(binding.parameter_name)) {
        throw std::runtime_error("ExpressionFormula lost parameter '" +
                                 binding.parameter_name + "'");
    }

    const auto value = _parameters->get_parameter(binding.parameter_name);
    static_cast<void>(expect_scalar_parameter(value, binding.parameter_name));

    auto* parameter_ptr = static_cast<double*>(
        _parameters->get_value_ptr(binding.parameter_name));
    if (parameter_ptr == nullptr) {
        throw std::runtime_error(
            "ExpressionFormula could not obtain parameter pointer for '" +
            binding.parameter_name + "'");
    }

    return parameter_ptr;
}

double ExpressionFormula::evaluate_expression(
    const SymEngine::RCP<const SymEngine::Basic>& expr) const {
    if (_symbols.empty()) {
        return SymEngine::eval_double(*expr);
    }

    SymEngine::map_basic_basic substitutions;
    for (std::size_t index = 0U; index < _symbols.size(); ++index) {
        substitutions.emplace(
            _symbols[index],
            SymEngine::number(evaluate_symbol_value(_bindings[index])));
    }

    return SymEngine::eval_double(*expr->subs(substitutions));
}

void ExpressionFormula::compile_formula() {
    auto* entity_ptr = entity().get_value_ref();
    if (entity_ptr == nullptr) {
        throw std::runtime_error(
            "ExpressionFormula could not obtain entity pointer");
    }
    set_entity_data_ptr(entity_ptr);

    _param_ptrs.clear();
    _param_ptrs.reserve(_bindings.size());
    for (const auto& binding : _bindings) {
        _param_ptrs.push_back(resolve_symbol_ptr(binding));
    }

    const auto inputs = lambda_inputs();
    try {
        _compiled_eval = SymEngine::LambdaRealDoubleVisitor();
        _compiled_eval.init(inputs, *_parsed_expr, true);
    } catch (const std::exception& exception) {
        throw std::runtime_error(
            "ExpressionFormula could not compile expression '" + _expression +
            "': " + exception.what());
    }

    _compiled_derivs.clear();
    _compiled_derivs_ready.clear();
    _compiled_derivs.reserve(_derivative_exprs.size());
    _compiled_derivs_ready.reserve(_derivative_exprs.size());

    for (const auto& derivative : _derivative_exprs) {
        try {
            auto visitor = SymEngine::LambdaRealDoubleVisitor();
            visitor.init(inputs, *derivative, true);
            _compiled_derivs.push_back(std::move(visitor));
            _compiled_derivs_ready.push_back(true);
        } catch (const std::exception&) {
            _compiled_derivs.emplace_back();
            _compiled_derivs_ready.push_back(false);
        }
    }

    _compiled = true;
}

void ExpressionFormula::apply_formula() {
    _cached_value = evaluate_expression(_parsed_expr);
    entity().set_value(_cached_value);
}

void ExpressionFormula::apply_compiled_formula() {
    if (!_compiled || (entity_data_ptr() == nullptr)) {
        throw std::runtime_error(
            "ExpressionFormula needs to be compiled before applying");
    }

    std::vector<double> inputs(_param_ptrs.size(), 0.0);
    for (std::size_t index = 0U; index < _param_ptrs.size(); ++index) {
        if (_param_ptrs[index] == nullptr) {
            throw std::runtime_error(
                "ExpressionFormula has an invalid compiled parameter pointer");
        }
        inputs[index] = *_param_ptrs[index];
    }

    _compiled_eval.call(&_cached_value,
                        inputs.empty() ? nullptr : inputs.data());
    *entity_data_ptr() = _cached_value;
}

void ExpressionFormula::calculate_derivatives() {
    _derivatives.assign(_derivative_exprs.size(), 0.0);

    if (!_compiled) {
        for (std::size_t index = 0U; index < _derivative_exprs.size();
             ++index) {
            _derivatives[index] = evaluate_expression(_derivative_exprs[index]);
        }
        return;
    }

    std::vector<double> inputs(_param_ptrs.size(), 0.0);
    for (std::size_t index = 0U; index < _param_ptrs.size(); ++index) {
        if (_param_ptrs[index] == nullptr) {
            throw std::runtime_error(
                "ExpressionFormula has an invalid compiled parameter pointer");
        }
        inputs[index] = *_param_ptrs[index];
    }

    for (std::size_t index = 0U; index < _derivative_exprs.size(); ++index) {
        if ((index < _compiled_derivs_ready.size()) &&
            _compiled_derivs_ready[index]) {
            _compiled_derivs[index].call(
                &_derivatives[index], inputs.empty() ? nullptr : inputs.data());
            continue;
        }

        _derivatives[index] = evaluate_expression(_derivative_exprs[index]);
    }
}

double ExpressionFormula::get_value() const {
    return evaluate_expression(_parsed_expr);
}

std::vector<double>* ExpressionFormula::get_derivative_values() {
    return &_derivatives;
}

std::unique_ptr<Formula> ExpressionFormula::clone() const {
    auto entity_clone = std::shared_ptr<ThermalEntity>(entity().clone());
    auto clone = std::make_unique<ExpressionFormula>(std::move(entity_clone),
                                                     *_parameters, _expression);
    clone->_cached_value = _cached_value;
    clone->_derivatives = _derivatives;
    if (_compiled) {
        clone->compile_formula();
    }
    return clone;
}

const std::string& ExpressionFormula::expression() const noexcept {
    return _expression;
}

}  // namespace pycanha