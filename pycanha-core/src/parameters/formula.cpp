#include "pycanha-core/parameters/formula.hpp"

#include <spdlog/spdlog.h>
#include <symengine/basic.h>
#include <symengine/derivative.h>
#include <symengine/parser.h>
#include <symengine/symbol.h>
#include <symengine/symengine_rcp.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <format>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula_program.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/parameters/table.hpp"
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

// Free symbols of `expr`, unique and sorted by name. The names of called
// functions are not symbols.
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

[[nodiscard]] const std::vector<std::string>& no_tables() noexcept {
    static const std::vector<std::string> empty;
    return empty;
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

FormulaKind Formula::kind() const noexcept {
    if (!_entity_dependencies.empty()) {
        return FormulaKind::Entities;
    }
    if (!_dependencies.empty()) {
        return FormulaKind::Parameters;
    }
    return FormulaKind::Constant;
}

const std::vector<std::string>& Formula::table_dependencies() const noexcept {
    return no_tables();
}

std::optional<std::string> Formula::problem() const {
    if (!_entity.exists()) {
        return "its target does not exist";
    }
    if (!_entity.writable()) {
        return "its target cannot be written";
    }
    return std::nullopt;
}

ExpressionFormula::ExpressionFormula(const Entity& entity, double value)
    : Formula(entity), _expression(std::format("{}", value)), _value(value) {}

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
                                     const detail::ParsedExpression& parsed,
                                     ThermalNetwork* network)
    : Formula(entity),
      _parameters(&parameters),
      _network(network),
      _expression(std::move(expression)) {
    initialize_expression(parsed);
}

void ExpressionFormula::initialize_expression(
    const detail::ParsedExpression& parsed) {
    std::vector<SymbolBinding> input_bindings;
    input_bindings.reserve(parsed.symbols.size());
    std::vector<Index> parameter_indices;
    Formula::DependencyList dependencies;
    std::vector<Entity> entity_inputs;

    for (const auto& symbol : parsed.symbols) {
        const auto& symbol_name = symbol->get_name();

        if (_parameters->contains(symbol_name)) {
            const auto parameter_idx = _parameters->get_idx(symbol_name);
            const auto parameter_name =
                parameter_idx.has_value()
                    ? _parameters->get_parameter_name(*parameter_idx)
                    : std::nullopt;
            if (!parameter_name.has_value() ||
                (_parameters->get_double_ptr(*parameter_idx) == nullptr)) {
                throw std::invalid_argument(
                    "ExpressionFormula expects parameter '" + symbol_name +
                    "' to store a double");
            }

            input_bindings.push_back(SymbolBinding{
                .kind = BindingKind::Parameter, .slot = dependencies.size()});
            parameter_indices.push_back(*parameter_idx);
            dependencies.push_back(*parameter_name);
        } else if (_parameters->has_table(symbol_name)) {
            throw std::invalid_argument(std::format(
                "'{0}' is a table; call it as {0}(argument)", symbol_name));
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

            input_bindings.push_back(SymbolBinding{
                .kind = BindingKind::Entity, .slot = entity_inputs.size()});
            entity_inputs.push_back(*input);
        } else {
            throw std::invalid_argument(
                "ExpressionFormula references unknown parameter '" +
                symbol_name + "'");
        }
    }

    set_parameter_dependencies(std::move(dependencies));
    set_entity_dependencies(std::move(entity_inputs));

    detail::LoweredExpression program;
    try {
        program = detail::lower_expression(*parsed.tree, parsed.symbols,
                                           *_parameters);
    } catch (const std::invalid_argument& exception) {
        throw std::invalid_argument("Formula '" + _expression +
                                    "': " + exception.what());
    }

    _derivatives.reset();
    if (parsed.symbols.empty() && program.tables.empty()) {
        // A constant: its value is all that needs keeping.
        _value = detail::evaluate_lowered(program, {}, {});
        _lowered.reset();
        return;
    }
    _lowered = std::make_shared<const Lowered>(
        Lowered{.expression = std::move(program),
                .parameter_indices = std::move(parameter_indices),
                .bindings = std::move(input_bindings)});
}

FormulaKind ExpressionFormula::kind() const noexcept {
    if (!entity_dependencies().empty()) {
        return FormulaKind::Entities;
    }
    if (!parameter_dependencies().empty() || !table_dependencies().empty()) {
        return FormulaKind::Parameters;
    }
    return FormulaKind::Constant;
}

const std::vector<std::string>& ExpressionFormula::table_dependencies()
    const noexcept {
    return _lowered == nullptr ? no_tables() : _lowered->expression.tables;
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

    const auto* value =
        _parameters->get_double_ptr(_lowered->parameter_indices[binding.slot]);
    if (value == nullptr) {
        throw std::runtime_error("ExpressionFormula lost parameter '" +
                                 binding_name(binding) + "'");
    }
    return *value;
}

std::span<const ExpressionFormula::SymbolBinding> ExpressionFormula::bindings()
    const noexcept {
    if (_lowered == nullptr) {
        return {};
    }
    return _lowered->bindings;
}

std::vector<double> ExpressionFormula::input_values() const {
    std::vector<double> values;
    values.reserve(bindings().size());
    std::ranges::transform(bindings(), std::back_inserter(values),
                           [this](const SymbolBinding& binding) {
                               return evaluate_symbol_value(binding);
                           });
    return values;
}

std::vector<double*> ExpressionFormula::input_pointers() const {
    std::vector<double*> pointers;
    pointers.reserve(bindings().size());
    for (const auto& binding : bindings()) {
        if (binding.kind == BindingKind::Entity) {
            const auto& input = entity_dependencies()[binding.slot];
            pointers.push_back(input.exists() ? input.get_value_ref()
                                              : nullptr);
            continue;
        }
        pointers.push_back(_parameters->get_double_ptr(
            _lowered->parameter_indices[binding.slot]));
    }
    return pointers;
}

bool ExpressionFormula::reads_entity(std::size_t input) const noexcept {
    return bindings()[input].kind == BindingKind::Entity;
}

std::vector<detail::TableView> ExpressionFormula::table_views(
    const std::vector<std::string>& tables) const {
    std::vector<detail::TableView> views;
    views.reserve(tables.size());
    const double* time = _parameters->time_ptr();
    for (const auto& name : tables) {
        const auto* table = _parameters->find_table(name);
        if (table == nullptr) {
            throw std::runtime_error("ExpressionFormula calls table '" + name +
                                     "', which does not exist");
        }
        views.push_back(table->view(time));
    }
    return views;
}

std::optional<std::string> ExpressionFormula::problem() const {
    if (auto target = Formula::problem(); target.has_value()) {
        return target;
    }
    const auto inputs = bindings();
    const auto missing =
        std::ranges::find_if(inputs, [this](const SymbolBinding& binding) {
            return binding.kind == BindingKind::Entity
                       ? !entity_dependencies()[binding.slot].exists()
                       : (_parameters->get_double_ptr(
                              _lowered->parameter_indices[binding.slot]) ==
                          nullptr);
        });
    if (missing != inputs.end()) {
        return "it reads '" + binding_name(*missing) +
               "', which does not exist";
    }
    const auto& tables = table_dependencies();
    const auto absent =
        std::ranges::find_if(tables, [this](const std::string& name) {
            return !_parameters->has_table(name);
        });
    if (absent != tables.end()) {
        return "it calls table '" + *absent + "', which does not exist";
    }
    return std::nullopt;
}

void ExpressionFormula::calculate_derivatives() {
    if (kind() == FormulaKind::Entities) {
        throw std::runtime_error(
            "ExpressionFormula derivatives are not available for "
            "entity-dependent expressions");
    }

    if (_derivatives == nullptr) {
        _derivatives = std::make_unique<DerivativeState>();
    }
    auto& state = *_derivatives;
    if (bindings().empty()) {
        state.values.clear();
        return;
    }

    if (state.lowered.size() != bindings().size()) {
        // Parsed again: the derivatives are rarely needed, so nothing is
        // kept for them until they are.
        const auto parsed = detail::parse_expression(_expression);
        state.lowered.clear();
        state.lowered.reserve(parsed.symbols.size());
        for (const auto& symbol : parsed.symbols) {
            const auto derivative = SymEngine::diff(parsed.tree, symbol);
            try {
                state.lowered.push_back(detail::lower_expression(
                    *derivative, parsed.symbols, *_parameters));
            } catch (const std::invalid_argument& exception) {
                state.lowered.clear();
                throw std::runtime_error("Formula '" + _expression +
                                         "' has no derivative with "
                                         "respect to '" +
                                         symbol->get_name() +
                                         "': " + exception.what());
            }
        }
    }

    const auto values = input_values();
    state.values.resize(state.lowered.size());
    for (std::size_t index = 0U; index < state.lowered.size(); ++index) {
        const auto& derivative = state.lowered[index];
        state.values[index] = detail::evaluate_lowered(
            derivative, values, table_views(derivative.tables));
    }
}

double ExpressionFormula::get_value() const {
    if (_lowered == nullptr) {
        return _value;
    }
    return detail::evaluate_lowered(_lowered->expression, input_values(),
                                    table_views(_lowered->expression.tables));
}

std::vector<double>* ExpressionFormula::get_derivative_values() {
    if (_derivatives == nullptr) {
        _derivatives = std::make_unique<DerivativeState>();
    }
    return &_derivatives->values;
}

std::unique_ptr<Formula> ExpressionFormula::clone() const {
    auto clone = std::make_unique<ExpressionFormula>(entity(), _value);
    clone->set_parameter_dependencies(parameter_dependencies());
    clone->set_entity_dependencies(entity_dependencies());
    clone->_parameters = _parameters;
    clone->_network = _network;
    clone->_expression = _expression;
    clone->_lowered = _lowered;
    if (_derivatives != nullptr) {
        clone->_derivatives = std::make_unique<DerivativeState>();
        clone->_derivatives->values = _derivatives->values;
    }
    return clone;
}

const std::string& ExpressionFormula::expression() const noexcept {
    return _expression;
}

CallableFormula::CallableFormula(const Entity& entity, Function function)
    : Formula(entity), _function(std::move(function)) {
    if (!_function) {
        throw std::invalid_argument("CallableFormula for '" +
                                    this->entity().string_representation() +
                                    "' needs a function");
    }
}

std::unique_ptr<Formula> CallableFormula::clone() const {
    auto clone = std::make_unique<CallableFormula>(entity(), _function);
    clone->_value = _value;
    return clone;
}

double CallableFormula::get_value() const { return _value; }

std::vector<double>* CallableFormula::get_derivative_values() {
    return nullptr;
}

const CallableFormula::Function& CallableFormula::function() const noexcept {
    return _function;
}

}  // namespace pycanha
