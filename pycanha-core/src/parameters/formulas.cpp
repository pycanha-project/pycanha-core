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
}

ParameterFormula Formulas::create_parameter_formula(
    ThermalEntity& entity, const std::string& parameter) {
    [[maybe_unused]] auto network = ensure_network(_network);
    auto parameters = ensure_parameters(_parameters);
    return {entity, *parameters, parameter};
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
    for (const auto& dependency : formula->parameter_dependencies()) {
        _parameter_dependencies[dependency].push_back(formula);
    }
}

void Formulas::apply_formulas() {
    for (const auto& formula : _formulas) {
        formula->apply_formula();
    }
}

}  // namespace pycanha
