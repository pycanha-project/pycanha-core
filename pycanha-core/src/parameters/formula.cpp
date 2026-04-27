#include "pycanha-core/parameters/formula.hpp"

#include <spdlog/spdlog.h>
#include <symengine/basic.h>
#include <symengine/derivative.h>
#include <symengine/dict.h>
#include <symengine/eval_double.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>
#include <symengine/real_double.h>
#include <symengine/symbol.h>
#include <symengine/symengine_rcp.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {

namespace {

using ExpressionNode = SymEngine::RCP<const SymEngine::Basic>;
using SymbolMap =
    std::map<std::string, SymEngine::RCP<const SymEngine::Symbol>>;

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

void collect_symbols(const ExpressionNode& expr, SymbolMap& symbols) {
    std::vector<ExpressionNode> pending{expr};

    while (!pending.empty()) {
        const ExpressionNode current = pending.back();
        pending.pop_back();

        if (const auto* symbol =
                dynamic_cast<const SymEngine::Symbol*>(current.get());
            symbol != nullptr) {
            symbols.emplace(symbol->get_name(),
                            SymEngine::symbol(symbol->get_name()));
            continue;
        }

        const auto arguments = current->get_args();
        std::copy(arguments.rbegin(), arguments.rend(),
                  std::back_inserter(pending));
    }
}

}  // namespace

ParameterFormula::ParameterFormula(Entity entity, Parameters& parameters,
                                   std::string expression)
    : Formula(entity),
      _parameters(&parameters),
      _expression(std::move(expression)) {
    initialize_expression();
}

void ParameterFormula::initialize_expression() {
    if (_parameters == nullptr) {
        throw std::invalid_argument(
            "ParameterFormula requires parameter storage");
    }

    _normalized_expression =
        preprocess_expression(detail::preprocess_formula_symbols(_expression));

    try {
        _parsed_expr = SymEngine::parse(_normalized_expression);
    } catch (const std::exception& exception) {
        throw std::invalid_argument("ParameterFormula could not parse '" +
                                    _expression + "': " + exception.what());
    }

    SymbolMap symbols;
    collect_symbols(_parsed_expr, symbols);
    if (symbols.empty()) {
        throw std::invalid_argument(
            "ParameterFormula requires at least one parameter or time symbol");
    }

    _symbols.clear();
    _parameter_indices.clear();
    Formula::DependencyList dependencies;

    for (const auto& [symbol_name, symbol] : symbols) {
        const auto parameter_idx = _parameters->get_idx(symbol_name);
        if (!parameter_idx.has_value()) {
            throw std::invalid_argument(
                "ParameterFormula references unknown parameter '" +
                symbol_name + "'");
        }
        const auto parameter_name =
            _parameters->get_parameter_name(*parameter_idx);
        if (!parameter_name.has_value()) {
            throw std::invalid_argument(
                "ParameterFormula references unknown parameter '" +
                symbol_name + "'");
        }
        if (_parameters->get_double_ptr(*parameter_idx) == nullptr) {
            throw std::invalid_argument("ParameterFormula expects parameter '" +
                                        symbol_name + "' to store a double");
        }

        _symbols.emplace_back(symbol);
        _parameter_indices.push_back(*parameter_idx);
        dependencies.push_back(*parameter_name);
    }

    set_parameter_dependencies(std::move(dependencies));

    _derivative_exprs.clear();
    _derivative_exprs.reserve(_symbols.size());
    std::transform(_symbols.begin(), _symbols.end(),
                   std::back_inserter(_derivative_exprs),
                   [this](const auto& symbol) {
                       return SymEngine::diff(_parsed_expr, symbol);
                   });
    _derivatives.assign(_symbols.size(), 0.0);
    _parameter_ptrs.clear();
    _compiled_derivs.clear();
    _compiled_derivs_ready.clear();
    _compiled = false;
}

SymEngine::vec_basic ParameterFormula::lambda_inputs() const {
    SymEngine::vec_basic inputs;
    inputs.reserve(_symbols.size());
    std::copy(_symbols.begin(), _symbols.end(), std::back_inserter(inputs));
    return inputs;
}

double ParameterFormula::evaluate_expression(
    const SymEngine::RCP<const SymEngine::Basic>& expr) const {
    SymEngine::map_basic_basic substitutions;
    for (std::size_t index = 0U; index < _symbols.size(); ++index) {
        const auto* parameter_value =
            _parameters->get_double_ptr(_parameter_indices[index]);
        if (parameter_value == nullptr) {
            throw std::runtime_error("ParameterFormula lost parameter '" +
                                     parameter_dependencies()[index] + "'");
        }
        substitutions.emplace(_symbols[index],
                              SymEngine::number(*parameter_value));
    }

    return SymEngine::eval_double(*expr->subs(substitutions));
}

void ParameterFormula::compile_formula() {
    auto* entity_ptr = entity().get_value_ref();
    if (entity_ptr == nullptr) {
        throw std::runtime_error(
            "ParameterFormula could not obtain entity pointer");
    }
    set_entity_data_ptr(entity_ptr);

    _parameter_ptrs.clear();
    _parameter_ptrs.reserve(_parameter_indices.size());
    for (const auto parameter_idx : _parameter_indices) {
        auto* parameter_ptr = _parameters->get_double_ptr(parameter_idx);
        if (parameter_ptr == nullptr) {
            throw std::runtime_error(
                "ParameterFormula could not obtain parameter pointer");
        }
        _parameter_ptrs.push_back(parameter_ptr);
    }

    const auto inputs = lambda_inputs();
    try {
        _compiled_eval = SymEngine::LambdaRealDoubleVisitor();
        _compiled_eval.init(inputs, *_parsed_expr, true);
    } catch (const std::exception& exception) {
        throw std::runtime_error(
            "ParameterFormula could not compile expression '" + _expression +
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
            _compiled_derivs.emplace_back(std::move(visitor));
            _compiled_derivs_ready.emplace_back(true);
        } catch (const std::exception&) {
            _compiled_derivs.emplace_back();
            _compiled_derivs_ready.emplace_back(false);
        }
    }

    _compiled_structure_version = _parameters->get_structure_version();
    _compiled = true;
}

void ParameterFormula::apply_formula() {
    _cached_value = evaluate_expression(_parsed_expr);
    if (!entity().set_value(_cached_value)) {
        throw std::runtime_error(
            "ParameterFormula could not assign entity value");
    }
}

void ParameterFormula::apply_compiled_formula() {
    if (!_compiled || (entity_data_ptr() == nullptr)) {
        throw std::runtime_error(
            "ParameterFormula needs to be compiled before applying");
    }
    if (_parameters->get_structure_version() != _compiled_structure_version) {
        throw std::runtime_error(
            "ParameterFormula compiled state is stale after structural "
            "parameter changes");
    }

    std::vector<double> inputs(_parameter_ptrs.size(), 0.0);
    for (std::size_t index = 0U; index < _parameter_ptrs.size(); ++index) {
        if (_parameter_ptrs[index] == nullptr) {
            throw std::runtime_error(
                "ParameterFormula has an invalid compiled parameter pointer");
        }
        inputs[index] = *_parameter_ptrs[index];
    }

    _compiled_eval.call(&_cached_value,
                        inputs.empty() ? nullptr : inputs.data());
    *entity_data_ptr() = _cached_value;
}

void ParameterFormula::calculate_derivatives() {
    _derivatives.assign(_derivative_exprs.size(), 0.0);

    if (!_compiled) {
        for (std::size_t index = 0U; index < _derivative_exprs.size();
             ++index) {
            _derivatives[index] = evaluate_expression(_derivative_exprs[index]);
        }
        return;
    }

    if (_parameters->get_structure_version() != _compiled_structure_version) {
        throw std::runtime_error(
            "ParameterFormula compiled derivatives are stale after structural "
            "parameter changes");
    }

    std::vector<double> inputs(_parameter_ptrs.size(), 0.0);
    for (std::size_t index = 0U; index < _parameter_ptrs.size(); ++index) {
        if (_parameter_ptrs[index] == nullptr) {
            throw std::runtime_error(
                "ParameterFormula has an invalid compiled parameter pointer");
        }
        inputs[index] = *_parameter_ptrs[index];
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

double ParameterFormula::get_value() const {
    return evaluate_expression(_parsed_expr);
}

std::vector<double>* ParameterFormula::get_derivative_values() {
    return &_derivatives;
}

std::unique_ptr<Formula> ParameterFormula::clone() const {
    auto clone =
        std::make_unique<ParameterFormula>(entity(), *_parameters, _expression);
    clone->_cached_value = _cached_value;
    clone->_derivatives = _derivatives;
    if (_compiled) {
        clone->compile_formula();
    }
    return clone;
}

const std::string& ParameterFormula::expression() const noexcept {
    return _expression;
}

ExpressionFormula::ExpressionFormula(Entity entity, Parameters& parameters,
                                     std::string expression,
                                     ThermalNetwork* network)
    : Formula(entity),
      _parameters(&parameters),
      _network(network),
      _expression(std::move(expression)) {
    initialize_expression();
}

void ExpressionFormula::initialize_expression() {
    if (_parameters == nullptr) {
        throw std::invalid_argument(
            "ExpressionFormula requires parameter storage");
    }

    _normalized_expression =
        preprocess_expression(detail::preprocess_formula_symbols(_expression));

    try {
        _parsed_expr = SymEngine::parse(_normalized_expression);
    } catch (const std::exception& exception) {
        throw std::invalid_argument("ExpressionFormula could not parse '" +
                                    _expression + "': " + exception.what());
    }

    SymbolMap symbols;
    collect_symbols(_parsed_expr, symbols);

    _symbols.clear();
    _bindings.clear();
    Formula::DependencyList dependencies;
    _has_entity_dependencies = false;

    for (const auto& [symbol_name, symbol] : symbols) {
        SymbolBinding binding;
        binding.symbol_name = symbol_name;

        if (_parameters->contains(symbol_name)) {
            const auto parameter_idx = _parameters->get_idx(symbol_name);
            if (!parameter_idx.has_value()) {
                throw std::invalid_argument(
                    "ExpressionFormula could not bind parameter '" +
                    symbol_name + "'");
            }
            const auto parameter_name =
                _parameters->get_parameter_name(*parameter_idx);
            if (!parameter_name.has_value()) {
                throw std::invalid_argument(
                    "ExpressionFormula could not bind parameter '" +
                    symbol_name + "'");
            }

            binding.kind = BindingKind::Parameter;
            binding.symbol_name = *parameter_name;
            binding.parameter_idx = *parameter_idx;
            if (_parameters->get_double_ptr(binding.parameter_idx) == nullptr) {
                throw std::invalid_argument(
                    "ExpressionFormula expects parameter '" + symbol_name +
                    "' to store a double");
            }
        } else if (_network != nullptr) {
            const auto entity_binding =
                Entity::from_internal_symbol(*_network, symbol_name);
            if (!entity_binding.has_value() || !entity_binding->exists()) {
                throw std::invalid_argument(
                    "ExpressionFormula references unknown symbol '" +
                    symbol_name + "'");
            }

            binding.kind = BindingKind::Entity;
            binding.entity = *entity_binding;
            _has_entity_dependencies = true;

            if (binding.entity.is_same_as(entity())) {
                SPDLOG_LOGGER_WARN(get_logger(),
                                   "ExpressionFormula for '{}' references "
                                   "itself on the right-hand side",
                                   entity().string_representation());
            }
        } else {
            throw std::invalid_argument(
                "ExpressionFormula references unknown parameter '" +
                symbol_name + "'");
        }

        _symbols.emplace_back(symbol);
        dependencies.emplace_back(binding.symbol_name);
        _bindings.emplace_back(std::move(binding));
    }

    set_parameter_dependencies(std::move(dependencies));

    _derivative_exprs.clear();
    if (!_has_entity_dependencies) {
        _derivative_exprs.reserve(_symbols.size());
        std::transform(_symbols.begin(), _symbols.end(),
                       std::back_inserter(_derivative_exprs),
                       [this](const auto& symbol) {
                           return SymEngine::diff(_parsed_expr, symbol);
                       });
        _derivatives.assign(_symbols.size(), 0.0);
    } else {
        _derivatives.clear();
    }

    _param_ptrs.clear();
    _compiled_derivs.clear();
    _compiled_derivs_ready.clear();
    _compiled = false;
}

SymEngine::vec_basic ExpressionFormula::lambda_inputs() const {
    SymEngine::vec_basic inputs;
    inputs.reserve(_symbols.size());
    std::copy(_symbols.begin(), _symbols.end(), std::back_inserter(inputs));
    return inputs;
}

double ExpressionFormula::evaluate_symbol_value(
    const SymbolBinding& binding) const {
    if (binding.kind == BindingKind::Entity) {
        if (!binding.entity.exists()) {
            throw std::runtime_error("ExpressionFormula lost entity '" +
                                     binding.symbol_name + "'");
        }

        return binding.entity.get_value();
    }

    if (_parameters == nullptr) {
        throw std::runtime_error("ExpressionFormula lost parameter storage");
    }

    const auto* parameter_value =
        _parameters->get_double_ptr(binding.parameter_idx);
    if (parameter_value == nullptr) {
        throw std::runtime_error("ExpressionFormula lost parameter '" +
                                 binding.symbol_name + "'");
    }

    return *parameter_value;
}

double* ExpressionFormula::resolve_symbol_ptr(
    const SymbolBinding& binding) const {
    if (binding.kind == BindingKind::Entity) {
        auto* entity_ptr = binding.entity.get_value_ref();
        if (entity_ptr == nullptr) {
            throw std::runtime_error(
                "ExpressionFormula could not obtain entity pointer for '" +
                binding.symbol_name + "'");
        }

        return entity_ptr;
    }

    if (_parameters == nullptr) {
        throw std::runtime_error("ExpressionFormula lost parameter storage");
    }

    auto* parameter_ptr = _parameters->get_double_ptr(binding.parameter_idx);
    if (parameter_ptr == nullptr) {
        throw std::runtime_error(
            "ExpressionFormula could not obtain parameter pointer for '" +
            binding.symbol_name + "'");
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
    std::transform(
        _bindings.begin(), _bindings.end(), std::back_inserter(_param_ptrs),
        [this](const auto& binding) { return resolve_symbol_ptr(binding); });

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
            _compiled_derivs.emplace_back(std::move(visitor));
            _compiled_derivs_ready.emplace_back(true);
        } catch (const std::exception&) {
            _compiled_derivs.emplace_back();
            _compiled_derivs_ready.emplace_back(false);
        }
    }

    _compiled_structure_version = _parameters->get_structure_version();
    _compiled = true;
}

void ExpressionFormula::apply_formula() {
    _cached_value = evaluate_expression(_parsed_expr);
    if (!entity().set_value(_cached_value)) {
        throw std::runtime_error(
            "ExpressionFormula could not assign entity value");
    }
}

void ExpressionFormula::apply_compiled_formula() {
    if (!_compiled || (entity_data_ptr() == nullptr)) {
        throw std::runtime_error(
            "ExpressionFormula needs to be compiled before applying");
    }
    if (_parameters->get_structure_version() != _compiled_structure_version) {
        throw std::runtime_error(
            "ExpressionFormula compiled state is stale after structural "
            "parameter changes");
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
    if (_has_entity_dependencies) {
        throw std::runtime_error(
            "ExpressionFormula derivatives are not available for "
            "entity-dependent expressions");
    }

    _derivatives.assign(_derivative_exprs.size(), 0.0);

    if (!_compiled) {
        for (std::size_t index = 0U; index < _derivative_exprs.size();
             ++index) {
            _derivatives[index] = evaluate_expression(_derivative_exprs[index]);
        }
        return;
    }

    if (_parameters->get_structure_version() != _compiled_structure_version) {
        throw std::runtime_error(
            "ExpressionFormula compiled derivatives are stale after structural "
            "parameter changes");
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
    auto clone = std::make_unique<ExpressionFormula>(entity(), *_parameters,
                                                     _expression, _network);
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
