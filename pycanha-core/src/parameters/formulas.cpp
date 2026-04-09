#include "pycanha-core/parameters/formulas.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha {

namespace {

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

}  // namespace

Formulas::Formulas() : _network(nullptr), _parameters(nullptr) {}

Formulas::Formulas(std::shared_ptr<ThermalNetwork> network,
                   std::shared_ptr<Parameters> parameters)
    : _network(std::move(network)), _parameters(std::move(parameters)) {}

void Formulas::associate(std::shared_ptr<ThermalNetwork> network,
                         std::shared_ptr<Parameters> parameters) {
    _network = std::move(network);
    _parameters = std::move(parameters);
    _validated_structure_version.reset();
    _compiled_structure_version.reset();
}

ParameterFormula Formulas::create_parameter_formula(
    Entity entity, const std::string& parameter) {
    [[maybe_unused]] auto validated_network = ensure_network(_network);
    auto parameter_storage = ensure_parameters(_parameters);
    return {entity, *parameter_storage, parameter};
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

    _formulas.push_back(formula);
    _validated_structure_version.reset();
    _compiled_structure_version.reset();
    for (const auto& dependency : formula->parameter_dependencies()) {
        _parameter_dependencies[dependency].push_back(formula);
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
    for (const auto& formula : _formulas) {
        formula->apply_formula();
    }
}

void Formulas::apply_compiled_formulas() {
    if (!is_compiled_current()) {
        throw std::runtime_error(
            "Compiled formulas are stale or unavailable for execution");
    }

    for (const auto& formula : _formulas) {
        formula->apply_compiled_formula();
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
