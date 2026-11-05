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
#include <ranges>
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
using SymbolNode = SymEngine::RCP<const SymEngine::Symbol>;

[[nodiscard]] std::string replace_python_power(std::string expression) {
    std::size_t position = 0U;
    while ((position = expression.find("**", position)) != std::string::npos) {
        expression.replace(position, 2U, "^");
        position += 1U;
    }
    return expression;
}

[[nodiscard]] std::string preprocess_expression(const std::string& expression) {
    if ((expression.contains('[')) || (expression.contains(']'))) {
        throw std::invalid_argument(
            "Formulas do not support matrix or array access yet");
    }

    return replace_python_power(expression);
}

// Free symbols of `expr`, unique and sorted by name.
[[nodiscard]] std::vector<SymbolNode> collect_symbols(
    const ExpressionNode& expr) {
    std::map<std::string, SymbolNode> symbols;
    std::vector<ExpressionNode> pending{expr};

    while (!pending.empty()) {
        const ExpressionNode current = pending.back();
        pending.pop_back();

        if (const auto* symbol =
                dynamic_cast<const SymEngine::Symbol*>(current.get());
            symbol != nullptr) {
            symbols.emplace(
                symbol->get_name(),
                SymEngine::rcp_static_cast<const SymEngine::Symbol>(current));
            continue;
        }

        const auto arguments = current->get_args();
        std::ranges::copy(std::views::reverse(arguments),
                          std::back_inserter(pending));
    }

    std::vector<SymbolNode> sorted;
    sorted.reserve(symbols.size());
    std::ranges::copy(std::views::values(symbols), std::back_inserter(sorted));
    return sorted;
}

// Derivatives of `expression` with respect to each symbol. The text is parsed
// again: the derivatives are rarely needed, so nothing is kept for them until
// they are.
[[nodiscard]] std::vector<ExpressionNode> differentiate(
    const std::string& expression, const std::vector<SymbolNode>& symbols) {
    const auto parsed = detail::parse_expression(expression);
    std::vector<ExpressionNode> derivatives;
    derivatives.reserve(symbols.size());
    std::ranges::transform(symbols, std::back_inserter(derivatives),
                           [&parsed](const auto& symbol) {
                               return SymEngine::diff(parsed.tree, symbol);
                           });
    return derivatives;
}

// Compiles each derivative. One that cannot be compiled is marked not ready
// and is evaluated by substitution instead.
void compile_derivatives(
    const SymEngine::vec_basic& inputs,
    const std::vector<ExpressionNode>& derivatives,
    std::vector<SymEngine::LambdaRealDoubleVisitor>& compiled,
    std::vector<bool>& ready) {
    compiled.clear();
    ready.clear();
    compiled.reserve(derivatives.size());
    ready.reserve(derivatives.size());
    for (const auto& derivative : derivatives) {
        try {
            auto visitor = SymEngine::LambdaRealDoubleVisitor();
            visitor.init(inputs, *derivative, true);
            compiled.emplace_back(std::move(visitor));
            ready.emplace_back(true);
        } catch (const std::exception&) {
            compiled.emplace_back();
            ready.emplace_back(false);
        }
    }
}

}  // namespace

namespace detail {

ParsedExpression parse_expression(const std::string& expression) {
    const auto normalized =
        preprocess_expression(preprocess_formula_symbols(expression));

    ParsedExpression parsed;
    try {
        parsed.tree = SymEngine::parse(normalized);
    } catch (const std::exception& exception) {
        throw std::invalid_argument("Formula could not parse '" + expression +
                                    "': " + exception.what());
    }
    parsed.symbols = collect_symbols(parsed.tree);
    return parsed;
}

}  // namespace detail

ParameterFormula::ParameterFormula(const Entity& entity, Parameters& parameters,
                                   std::string expression)
    : Formula(entity),
      _parameters(&parameters),
      _expression(std::move(expression)) {
    initialize_expression(detail::parse_expression(_expression));
}

ParameterFormula::ParameterFormula(const Entity& entity, Parameters& parameters,
                                   std::string expression,
                                   detail::ParsedExpression parsed)
    : Formula(entity),
      _parameters(&parameters),
      _expression(std::move(expression)) {
    initialize_expression(std::move(parsed));
}

void ParameterFormula::initialize_expression(detail::ParsedExpression parsed) {
    if (parsed.symbols.empty()) {
        throw std::invalid_argument(
            "ParameterFormula requires at least one parameter or time symbol");
    }

    _parameter_indices.clear();
    _parameter_indices.reserve(parsed.symbols.size());
    Formula::DependencyList dependencies;
    dependencies.reserve(parsed.symbols.size());

    for (const auto& symbol : parsed.symbols) {
        const auto& symbol_name = symbol->get_name();
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

        _parameter_indices.push_back(*parameter_idx);
        dependencies.push_back(*parameter_name);
    }

    set_parameter_dependencies(std::move(dependencies));
    _parsed_expr = parsed.tree;
    _symbols = std::move(parsed.symbols);

    _derivative_exprs.clear();
    _derivatives.clear();
    _parameter_ptrs.clear();
    _compiled_derivs.clear();
    _compiled_derivs_ready.clear();
    _compiled = false;
}

SymEngine::vec_basic ParameterFormula::lambda_inputs() const {
    SymEngine::vec_basic inputs;
    inputs.reserve(_symbols.size());
    std::ranges::copy(_symbols, std::back_inserter(inputs));
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

    try {
        _compiled_eval = SymEngine::LambdaRealDoubleVisitor();
        _compiled_eval.init(lambda_inputs(), *_parsed_expr, true);
    } catch (const std::exception& exception) {
        throw std::runtime_error(
            "ParameterFormula could not compile expression '" + _expression +
            "': " + exception.what());
    }

    // Compiled again on the next calculate_derivatives() call.
    _compiled_derivs.clear();
    _compiled_derivs_ready.clear();

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
    if (_derivative_exprs.size() != _symbols.size()) {
        _derivative_exprs = differentiate(_expression, _symbols);
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
            "ParameterFormula compiled derivatives are stale after structural "
            "parameter changes");
    }

    if (_compiled_derivs.size() != _derivative_exprs.size()) {
        compile_derivatives(lambda_inputs(), _derivative_exprs,
                            _compiled_derivs, _compiled_derivs_ready);
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
        if (_compiled_derivs_ready[index]) {
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
    auto clone = std::make_unique<ParameterFormula>(
        entity(), *_parameters, _expression,
        detail::ParsedExpression{.tree = _parsed_expr, .symbols = _symbols});
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

ExpressionFormula::ExpressionFormula(const Entity& entity,
                                     Parameters& parameters,
                                     std::string expression,
                                     ThermalNetwork* network)
    : Formula(entity),
      _parameters(&parameters),
      _network(network),
      _expression(std::move(expression)) {
    initialize_expression(detail::parse_expression(_expression));
}

ExpressionFormula::ExpressionFormula(const Entity& entity,
                                     Parameters& parameters,
                                     std::string expression,
                                     detail::ParsedExpression parsed,
                                     ThermalNetwork* network)
    : Formula(entity),
      _parameters(&parameters),
      _network(network),
      _expression(std::move(expression)) {
    initialize_expression(std::move(parsed));
}

void ExpressionFormula::initialize_expression(detail::ParsedExpression parsed) {
    _bindings.clear();
    _bindings.reserve(parsed.symbols.size());
    _parameter_indices.clear();
    Formula::DependencyList dependencies;
    std::vector<Entity> entity_inputs;

    for (const auto& symbol : parsed.symbols) {
        const auto& symbol_name = symbol->get_name();

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
            if (_parameters->get_double_ptr(*parameter_idx) == nullptr) {
                throw std::invalid_argument(
                    "ExpressionFormula expects parameter '" + symbol_name +
                    "' to store a double");
            }

            _bindings.push_back(SymbolBinding{.kind = BindingKind::Parameter,
                                              .slot = dependencies.size()});
            _parameter_indices.push_back(*parameter_idx);
            dependencies.push_back(*parameter_name);
        } else if (_network != nullptr) {
            const auto input =
                Entity::from_internal_symbol(*_network, symbol_name);
            if (!input.has_value() || !input->exists()) {
                throw std::invalid_argument(
                    "ExpressionFormula references unknown symbol '" +
                    symbol_name + "'");
            }

            if (input->is_same_as(entity())) {
                SPDLOG_LOGGER_WARN(get_logger(),
                                   "ExpressionFormula for '{}' references "
                                   "itself on the right-hand side",
                                   entity().string_representation());
            }

            _bindings.push_back(SymbolBinding{.kind = BindingKind::Entity,
                                              .slot = entity_inputs.size()});
            entity_inputs.push_back(*input);
        } else {
            throw std::invalid_argument(
                "ExpressionFormula references unknown parameter '" +
                symbol_name + "'");
        }
    }

    _has_entity_dependencies = !entity_inputs.empty();
    set_parameter_dependencies(std::move(dependencies));
    set_entity_dependencies(std::move(entity_inputs));
    _parsed_expr = parsed.tree;
    _symbols = std::move(parsed.symbols);

    _derivative_exprs.clear();
    _derivatives.clear();
    _param_ptrs.clear();
    _compiled_derivs.clear();
    _compiled_derivs_ready.clear();
    _compiled = false;
}

SymEngine::vec_basic ExpressionFormula::lambda_inputs() const {
    SymEngine::vec_basic inputs;
    inputs.reserve(_symbols.size());
    std::ranges::copy(_symbols, std::back_inserter(inputs));
    return inputs;
}

std::string ExpressionFormula::binding_name(
    const SymbolBinding& binding) const {
    if (binding.kind == BindingKind::Entity) {
        return entity_dependencies()[binding.slot].string_representation();
    }
    return parameter_dependencies()[binding.slot];
}

double ExpressionFormula::evaluate_symbol_value(
    const SymbolBinding& binding) const {
    if (binding.kind == BindingKind::Entity) {
        const auto& input = entity_dependencies()[binding.slot];
        if (!input.exists()) {
            throw std::runtime_error("ExpressionFormula lost entity '" +
                                     binding_name(binding) + "'");
        }

        return input.get_value();
    }

    const auto* parameter_value =
        _parameters->get_double_ptr(_parameter_indices[binding.slot]);
    if (parameter_value == nullptr) {
        throw std::runtime_error("ExpressionFormula lost parameter '" +
                                 binding_name(binding) + "'");
    }

    return *parameter_value;
}

double* ExpressionFormula::resolve_symbol_ptr(
    const SymbolBinding& binding) const {
    auto* value_ptr =
        binding.kind == BindingKind::Entity
            ? entity_dependencies()[binding.slot].get_value_ref()
            : _parameters->get_double_ptr(_parameter_indices[binding.slot]);
    if (value_ptr == nullptr) {
        throw std::runtime_error(
            "ExpressionFormula could not obtain a pointer for '" +
            binding_name(binding) + "'");
    }

    return value_ptr;
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
    std::ranges::transform(
        _bindings, std::back_inserter(_param_ptrs),
        [this](const auto& binding) { return resolve_symbol_ptr(binding); });

    try {
        _compiled_eval = SymEngine::LambdaRealDoubleVisitor();
        _compiled_eval.init(lambda_inputs(), *_parsed_expr, true);
    } catch (const std::exception& exception) {
        throw std::runtime_error(
            "ExpressionFormula could not compile expression '" + _expression +
            "': " + exception.what());
    }

    // Compiled again on the next calculate_derivatives() call.
    _compiled_derivs.clear();
    _compiled_derivs_ready.clear();

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

    if (_derivative_exprs.size() != _symbols.size()) {
        _derivative_exprs = differentiate(_expression, _symbols);
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

    if (_compiled_derivs.size() != _derivative_exprs.size()) {
        compile_derivatives(lambda_inputs(), _derivative_exprs,
                            _compiled_derivs, _compiled_derivs_ready);
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
        if (_compiled_derivs_ready[index]) {
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
    auto clone = std::make_unique<ExpressionFormula>(
        entity(), *_parameters, _expression,
        detail::ParsedExpression{.tree = _parsed_expr, .symbols = _symbols},
        _network);
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
