#include "pycanha-core/parameters/formula.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula_program.hpp"
#include "pycanha-core/parameters/formula_shape.hpp"
#include "pycanha-core/parameters/origin.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/parameters/table.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {

namespace {

[[nodiscard]] const std::vector<std::string>& no_names() noexcept {
    static const std::vector<std::string> empty;
    return empty;
}

[[nodiscard]] const std::vector<Entity>& no_entities() noexcept {
    static const std::vector<Entity> empty;
    return empty;
}

}  // namespace

const Formula::DependencyList& Formula::parameter_dependencies()
    const noexcept {
    return no_names();
}

const std::vector<Entity>& Formula::entity_dependencies() const noexcept {
    return no_entities();
}

const std::vector<std::string>& Formula::table_dependencies() const noexcept {
    return no_names();
}

ExpressionFormula::ExpressionFormula(const Entity& entity, double value)
    : Formula(entity) {
    auto formula = detail::constant_formula(entity, value);
    _shape = detail::build_shape(std::move(formula.text), nullptr, nullptr,
                                 "constant");
    _values = std::move(formula.values);
}

ExpressionFormula::ExpressionFormula(const Entity& entity,
                                     Parameters& parameters,
                                     const std::string& expression,
                                     ThermalNetwork* network)
    : Formula(entity), _parameters(&parameters), _network(network) {
    auto formula = detail::templatize(entity, expression, &parameters, network);
    _shape = detail::build_shape(std::move(formula.text), &parameters, network,
                                 expression);
    _values = std::move(formula.values);
    resolve_inputs();
    check_inputs();
    if (_shape->reads_target) {
        SPDLOG_LOGGER_WARN(get_logger(),
                           "ExpressionFormula for '{}' references itself on "
                           "the right-hand side",
                           this->entity().string_representation());
    }
}

ExpressionFormula::ExpressionFormula(
    Key /*key*/, const Entity& entity,
    std::shared_ptr<const detail::FormulaShape> shape,
    detail::ShapeValues values, Parameters* parameters, ThermalNetwork* network,
    Origin origin)
    : Formula(entity),
      _parameters(parameters),
      _network(network),
      _shape(std::move(shape)),
      _values(std::move(values)),
      _origin(origin) {
    resolve_inputs();
}

void ExpressionFormula::resolve_inputs() {
    _inputs.clear();
    if (_shape->entities.empty()) {
        return;
    }
    if (_network == nullptr) {
        throw std::invalid_argument(
            "ExpressionFormula reads entities and needs a network");
    }
    _inputs.reserve(_shape->entities.size());
    std::ranges::transform(_shape->entities, std::back_inserter(_inputs),
                           [this](const detail::EntityPattern& pattern) {
                               return pattern.resolve(*_network, _values.nodes);
                           });
}

void ExpressionFormula::check_inputs() const {
    const auto missing = std::ranges::find_if(
        _inputs, [](const Entity& input) { return !input.exists(); });
    if (missing != _inputs.end()) {
        throw std::invalid_argument(
            "ExpressionFormula references unknown symbol '" +
            missing->string_representation() + "'");
    }
}

FormulaKind ExpressionFormula::kind() const noexcept { return _shape->kind; }

const Formula::DependencyList& ExpressionFormula::parameter_dependencies()
    const noexcept {
    return _shape->parameter_names;
}

const std::vector<Entity>& ExpressionFormula::entity_dependencies()
    const noexcept {
    return _inputs;
}

const std::vector<std::string>& ExpressionFormula::table_dependencies()
    const noexcept {
    return _shape->lowered.tables;
}

std::vector<double> ExpressionFormula::input_values() const {
    std::vector<double> values;
    values.reserve(_shape->inputs.size());
    for (const auto& input : _shape->inputs) {
        if (input.kind == detail::ShapeInput::Kind::Entity) {
            const auto& entity = _inputs[input.index];
            if (!entity.exists()) {
                throw std::runtime_error("ExpressionFormula lost entity '" +
                                         entity.string_representation() + "'");
            }
            values.push_back(entity.get_value());
            continue;
        }
        const auto* value = detail::ParameterAccess::value(
            *_parameters, _shape->parameter_indices[input.index]);
        if (value == nullptr) {
            throw std::runtime_error("ExpressionFormula lost parameter '" +
                                     _shape->parameter_names[input.index] +
                                     "'");
        }
        values.push_back(*value);
    }
    return values;
}

std::vector<detail::TableView> ExpressionFormula::table_views(
    const std::vector<std::string>& tables) const {
    std::vector<detail::TableView> views;
    if (tables.empty()) {
        return views;
    }
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

void ExpressionFormula::calculate_derivatives() {
    if (kind() == FormulaKind::Entities) {
        throw std::runtime_error(
            "ExpressionFormula derivatives are not available for "
            "entity-dependent expressions");
    }

    if (_derivatives == nullptr) {
        _derivatives = std::make_unique<std::vector<double>>();
    }
    auto& values = *_derivatives;
    if (_shape->inputs.empty()) {
        values.clear();
        return;
    }

    const auto& derivatives = detail::shape_derivatives(*_shape);
    const auto inputs = input_values();
    values.resize(derivatives.size());
    std::ranges::transform(
        derivatives, values.begin(),
        [this, &inputs](const detail::LoweredExpression& derivative) {
            return detail::evaluate_lowered(derivative, inputs, _values.numbers,
                                            table_views(derivative.tables));
        });
}

double ExpressionFormula::get_value() const {
    return detail::evaluate_lowered(_shape->lowered, input_values(),
                                    _values.numbers,
                                    table_views(_shape->lowered.tables));
}

std::vector<double>* ExpressionFormula::get_derivative_values() {
    if (_derivatives == nullptr) {
        _derivatives = std::make_unique<std::vector<double>>();
    }
    return _derivatives.get();
}

std::unique_ptr<Formula> ExpressionFormula::clone() const {
    auto clone = std::make_unique<ExpressionFormula>(
        Key{}, entity(), _shape, _values, _parameters, _network, _origin);
    if (_derivatives != nullptr) {
        clone->_derivatives =
            std::make_unique<std::vector<double>>(*_derivatives);
    }
    return clone;
}

const std::string& ExpressionFormula::expression() const {
    if (_expression.empty()) {
        _expression =
            detail::print_expression(*_shape, _values.nodes, _values.numbers);
    }
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
