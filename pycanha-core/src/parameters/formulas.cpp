#include "pycanha-core/parameters/formulas.hpp"

#include <spdlog/spdlog.h>
#include <symengine/basic.h>
#include <symengine/eval_double.h>
#include <symengine/parser.h>
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
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {

void DerivativeParameterRegistry::add_parameter(
    const std::string& parameter_name) {
    if (_parameters == nullptr) {
        throw std::runtime_error(
            "Derivative parameter registry requires parameter storage");
    }
    if (!_parameters->contains(parameter_name)) {
        throw std::invalid_argument("Unknown parameter '" + parameter_name +
                                    "' cannot be added to the derivative set");
    }

    const auto parameter_idx = _parameters->get_idx(parameter_name);
    if (!parameter_idx.has_value()) {
        throw std::invalid_argument("Unknown parameter '" + parameter_name +
                                    "' cannot be added to the derivative set");
    }

    const auto resolved_name = _parameters->get_parameter_name(*parameter_idx);
    if (!resolved_name.has_value()) {
        throw std::invalid_argument("Unknown parameter '" + parameter_name +
                                    "' cannot be added to the derivative set");
    }

    if (contains(*resolved_name)) {
        return;
    }

    // TODO: validate that derivative-selected parameters are used only by
    // ParameterFormula entries once the formula taxonomy migration is complete.
    _parameter_names.push_back(*resolved_name);
}

bool DerivativeParameterRegistry::remove_parameter(
    const std::string& parameter_name) noexcept {
    if (_parameters == nullptr) {
        return false;
    }

    const auto parameter_idx = _parameters->get_idx(parameter_name);
    if (!parameter_idx.has_value()) {
        return false;
    }

    const auto resolved_name = _parameters->get_parameter_name(*parameter_idx);
    if (!resolved_name.has_value()) {
        return false;
    }

    const auto iterator = std::find(_parameter_names.begin(),
                                    _parameter_names.end(), *resolved_name);
    if (iterator == _parameter_names.end()) {
        return false;
    }

    _parameter_names.erase(iterator);
    return true;
}

bool DerivativeParameterRegistry::contains(
    const std::string& parameter_name) const noexcept {
    if (_parameters == nullptr) {
        return false;
    }

    const auto parameter_idx = _parameters->get_idx(parameter_name);
    if (!parameter_idx.has_value()) {
        return false;
    }

    const auto resolved_name = _parameters->get_parameter_name(*parameter_idx);
    if (!resolved_name.has_value()) {
        return false;
    }

    return std::find(_parameter_names.begin(), _parameter_names.end(),
                     *resolved_name) != _parameter_names.end();
}

[[nodiscard]] std::shared_ptr<ThermalNetwork> ensure_network(
    const std::shared_ptr<ThermalNetwork>& network) {
    if (network == nullptr) {
        throw std::runtime_error("Formulas requires a thermal network");
    }
    return network;
}

[[nodiscard]] std::shared_ptr<Parameters> ensure_parameters(
    const std::shared_ptr<Parameters>& parameters) {
    if (parameters == nullptr) {
        throw std::runtime_error("Formulas requires parameters");
    }
    return parameters;
}

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

Formulas::Formulas() : _network(nullptr), _parameters(nullptr) {}

Formulas::Formulas(std::shared_ptr<ThermalNetwork> network,
                   std::shared_ptr<Parameters> parameters)
    : _network(std::move(network)),
      _parameters(std::move(parameters)),
      _parameters_with_derivatives(_parameters) {}

void Formulas::associate(std::shared_ptr<ThermalNetwork> network,
                         std::shared_ptr<Parameters> parameters) {
    _network = std::move(network);
    _parameters = std::move(parameters);
    _parameters_with_derivatives.associate(_parameters);
    _validated_structure_version.reset();
    _compiled_structure_version.reset();
}

Entity Formulas::resolve_entity(std::string_view entity) const {
    const auto network = ensure_network(_network);
    const auto resolved = Entity::from_string(*network, entity);
    if (!resolved.has_value()) {
        throw std::invalid_argument("Unknown formula target '" +
                                    std::string(entity) + "'");
    }

    return *resolved;
}

std::shared_ptr<Formula> Formulas::create_formula(
    Entity entity, const std::string& formula_string) {
    [[maybe_unused]] auto validated_network = ensure_network(_network);
    auto parameter_storage = ensure_parameters(_parameters);

    const auto normalized = preprocess_expression(
        detail::preprocess_formula_symbols(formula_string));

    ExpressionNode parsed_expression;
    try {
        parsed_expression = SymEngine::parse(normalized);
    } catch (const std::exception& exception) {
        throw std::invalid_argument("Formula could not parse '" +
                                    formula_string + "': " + exception.what());
    }

    SymbolMap symbols;
    collect_symbols(parsed_expression, symbols);
    if (symbols.empty()) {
        auto formula = std::make_shared<ValueFormula>(entity);
        formula->set_value(SymEngine::eval_double(*parsed_expression));
        return formula;
    }

    bool has_entity_symbols = false;
    bool all_parameter_symbols = true;

    for (const auto& [symbol_name, symbol] : symbols) {
        (void)symbol;

        if (parameter_storage->contains(symbol_name)) {
            continue;
        }

        all_parameter_symbols = false;

        if ((_temperature_variable_names != nullptr) &&
            _temperature_variable_names->contains(symbol_name)) {
            throw std::invalid_argument("TemperatureVariable '" + symbol_name +
                                        "' in formulas is not implemented yet");
        }

        const auto referenced_entity =
            Entity::from_internal_symbol(*_network, symbol_name);
        if (!referenced_entity.has_value() || !referenced_entity->exists()) {
            throw std::invalid_argument("Unknown symbol '" + symbol_name +
                                        "' in formula");
        }

        has_entity_symbols = true;
        if (referenced_entity->is_same_as(entity)) {
            SPDLOG_LOGGER_WARN(get_logger(),
                               "Formula for '{}' references itself on the "
                               "right-hand side",
                               entity.string_representation());
        }
    }

    if (all_parameter_symbols) {
        return std::make_shared<ParameterFormula>(entity, *parameter_storage,
                                                  formula_string);
    }

    if (has_entity_symbols) {
        return std::make_shared<ExpressionFormula>(
            entity, *parameter_storage, formula_string, _network.get());
    }

    return std::make_shared<ExpressionFormula>(entity, *parameter_storage,
                                               formula_string);
}

void Formulas::set_temperature_variable_names(
    const std::unordered_set<std::string>* names) noexcept {
    _temperature_variable_names = names;
}

ParameterFormula Formulas::create_parameter_formula(
    Entity entity, const std::string& parameter) {
    [[maybe_unused]] auto validated_network = ensure_network(_network);
    auto parameter_storage = ensure_parameters(_parameters);
    return {entity, *parameter_storage, parameter};
}

ValueFormula& Formulas::add_value_formula(Entity entity, double value) {
    auto formula = std::make_shared<ValueFormula>(entity);
    formula->set_value(value);
    add_formula(formula);
    return *formula;
}

ValueFormula& Formulas::add_value_formula(std::string_view entity,
                                          double value) {
    return add_value_formula(resolve_entity(entity), value);
}

ParameterFormula& Formulas::add_parameter_formula(
    Entity entity, const std::string& expression) {
    auto formula = std::make_shared<ParameterFormula>(
        entity, *ensure_parameters(_parameters), expression);
    add_formula(formula);
    return *formula;
}

ParameterFormula& Formulas::add_parameter_formula(
    std::string_view entity, const std::string& expression) {
    return add_parameter_formula(resolve_entity(entity), expression);
}

ExpressionFormula& Formulas::add_expression_formula(
    Entity entity, const std::string& expression) {
    auto formula = std::make_shared<ExpressionFormula>(
        entity, *ensure_parameters(_parameters), expression, _network.get());
    add_formula(formula);
    return *formula;
}

ExpressionFormula& Formulas::add_expression_formula(
    std::string_view entity, const std::string& expression) {
    return add_expression_formula(resolve_entity(entity), expression);
}

Formula& Formulas::add_formula(Entity entity, double value) {
    return add_value_formula(entity, value);
}

Formula& Formulas::add_formula(std::string_view entity, double value) {
    return add_value_formula(entity, value);
}

Formula& Formulas::add_formula(Entity entity,
                               const std::string& formula_string) {
    auto formula = create_formula(entity, formula_string);
    add_formula(formula);
    return *formula;
}

Formula& Formulas::add_formula(std::string_view entity,
                               const std::string& formula_string) {
    return add_formula(resolve_entity(entity), formula_string);
}

void Formulas::add_formula(const Formula& formula) {
    auto clone = formula.clone();
    auto shared = std::shared_ptr<Formula>(std::move(clone));
    add_formula(shared);
}

void Formulas::add_formula(const std::shared_ptr<Formula>& formula) {
    if (formula == nullptr) {
        throw std::invalid_argument("Cannot add a null formula");
    }

    const auto duplicate = std::find_if(
        _formulas.begin(), _formulas.end(), [&formula](const auto& existing) {
            return existing->entity().is_same_as(formula->entity());
        });
    if (duplicate != _formulas.end()) {
        throw std::invalid_argument("A formula for entity '" +
                                    formula->entity().string_representation() +
                                    "' already exists");
    }

    // Materialize writable target storage immediately so sparse node
    // attributes become formula-managed as soon as the formula is registered.
    if (formula->entity().exists() && formula->entity().writable()) {
        static_cast<void>(formula->entity().get_value_ref());
    }

    _formulas.push_back(formula);
    _validated_structure_version.reset();
    _compiled_structure_version.reset();
    for (const auto& dependency : formula->parameter_dependencies()) {
        _parameter_dependencies[dependency].push_back(formula);
    }
}

bool Formulas::remove_formula(const Entity& entity) noexcept {
    const auto iterator = std::find_if(
        _formulas.begin(), _formulas.end(), [&entity](const auto& formula) {
            return formula->entity().is_same_as(entity);
        });
    if (iterator == _formulas.end()) {
        SPDLOG_LOGGER_INFO(get_logger(),
                           "Formula '{}' was not present for removal",
                           entity.string_representation());
        return false;
    }

    _formulas.erase(iterator);
    _parameter_dependencies.clear();
    for (const auto& formula : _formulas) {
        for (const auto& dependency : formula->parameter_dependencies()) {
            _parameter_dependencies[dependency].push_back(formula);
        }
    }
    _validated_structure_version.reset();
    _compiled_structure_version.reset();
    return true;
}

bool Formulas::remove_formula(std::string_view entity) noexcept {
    try {
        return remove_formula(resolve_entity(entity));
    } catch (const std::exception&) {
        SPDLOG_LOGGER_INFO(get_logger(),
                           "Formula target '{}' was not present for removal",
                           std::string(entity));
        return false;
    }
}

void Formulas::validate_for_execution() {
    auto parameter_storage = ensure_parameters(_parameters);

    for (const auto& formula : _formulas) {
        static_cast<void>(formula->get_value());
    }

    _validated_structure_version = parameter_storage->get_structure_version();
    _compiled_structure_version.reset();
}

void Formulas::compile_formulas() {
    if (!is_validation_current()) {
        validate_for_execution();
    }

    for (const auto& formula : _formulas) {
        formula->compile_formula();
    }

    _compiled_structure_version = _parameters->get_structure_version();
}

void Formulas::apply_formulas() {
    if (!debug_formulas) {
        for (const auto& formula : _formulas) {
            formula->apply_formula();
        }
        return;
    }

    const auto logger = get_logger();
    for (const auto& formula : _formulas) {
        formula->apply_formula();
        SPDLOG_LOGGER_INFO(logger, "[Formula Debug] {} = {}",
                           formula->entity().string_representation(),
                           formula->get_value());
    }
}

void Formulas::apply_compiled_formulas() {
    if (!is_compiled_current()) {
        throw std::runtime_error(
            "Compiled formulas are stale or unavailable for execution");
    }

    if (!debug_formulas) {
        for (const auto& formula : _formulas) {
            formula->apply_compiled_formula();
        }
        return;
    }

    const auto logger = get_logger();
    for (const auto& formula : _formulas) {
        formula->apply_compiled_formula();
        SPDLOG_LOGGER_INFO(logger, "[Formula Debug] {} = {}",
                           formula->entity().string_representation(),
                           formula->get_value());
    }
}

void Formulas::calculate_derivatives() {
    for (const auto& formula : _formulas) {
        formula->calculate_derivatives();
    }
}

void Formulas::lock_parameters_for_execution() {
    auto parameter_storage = ensure_parameters(_parameters);
    if (!is_validation_current()) {
        throw std::runtime_error(
            "Formulas validation is stale; validate before locking "
            "parameters for execution");
    }

    parameter_storage->lock_structure();
}

void Formulas::unlock_parameters() noexcept {
    if (_parameters == nullptr) {
        return;
    }

    _parameters->unlock_structure();
}

bool Formulas::is_validation_current() const noexcept {
    return (_parameters != nullptr) &&
           _validated_structure_version.has_value() &&
           (_parameters->get_structure_version() ==
            *_validated_structure_version);
}

bool Formulas::is_compiled_current() const noexcept {
    return (_parameters != nullptr) &&
           _compiled_structure_version.has_value() &&
           (_parameters->get_structure_version() ==
            *_compiled_structure_version);
}

}  // namespace pycanha
