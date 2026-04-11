#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

#include <spdlog/spdlog.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/parameters/variable.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {
ThermalMathematicalModel::ThermalMathematicalModel(std::string model_name)
    : _network(std::make_shared<ThermalNetwork>()),
      _parameters_shptr(std::make_shared<Parameters>()),
      _formulas_shptr(std::make_shared<Formulas>()),
      _thermal_data_shptr(std::make_shared<ThermalData>()),
      name(std::move(model_name)),
      parameters(*_parameters_shptr),
      formulas(*_formulas_shptr),
      thermal_data(*_thermal_data_shptr) {
    associate_resources();
    initialize_internal_time_parameter();
    const auto logger = get_logger();

    SPDLOG_LOGGER_TRACE(logger,
                        "ThermalMathematicalModel: default constructor");
}

ThermalMathematicalModel::ThermalMathematicalModel(
    std::string model_name, std::shared_ptr<Nodes> nodes,
    std::shared_ptr<ConductiveCouplings> conductive,
    std::shared_ptr<RadiativeCouplings> radiative)
    : _network(std::make_shared<ThermalNetwork>(
          std::move(nodes), std::move(conductive), std::move(radiative))),
      _parameters_shptr(std::make_shared<Parameters>()),
      _formulas_shptr(std::make_shared<Formulas>()),
      _thermal_data_shptr(std::make_shared<ThermalData>()),
      name(std::move(model_name)),
      parameters(*_parameters_shptr),
      formulas(*_formulas_shptr),
      thermal_data(*_thermal_data_shptr) {
    associate_resources();
    initialize_internal_time_parameter();
    const auto logger = get_logger();

    SPDLOG_LOGGER_TRACE(
        logger, "ThermalMathematicalModel: constructor with shared nodes");
}

ThermalMathematicalModel::ThermalMathematicalModel(
    std::string model_name, std::shared_ptr<Nodes> nodes,
    std::shared_ptr<ConductiveCouplings> conductive,
    std::shared_ptr<RadiativeCouplings> radiative,
    std::shared_ptr<Parameters> parameters, std::shared_ptr<Formulas> formulas,
    std::shared_ptr<ThermalData> thermal_data)
    : _network(std::make_shared<ThermalNetwork>(
          std::move(nodes), std::move(conductive), std::move(radiative))),
      _parameters_shptr(parameters != nullptr ? std::move(parameters)
                                              : std::make_shared<Parameters>()),
      _formulas_shptr(formulas != nullptr ? std::move(formulas)
                                          : std::make_shared<Formulas>()),
      _thermal_data_shptr(thermal_data != nullptr
                              ? std::move(thermal_data)
                              : std::make_shared<ThermalData>()),
      name(std::move(model_name)),
      parameters(*_parameters_shptr),
      formulas(*_formulas_shptr),
      thermal_data(*_thermal_data_shptr) {
    associate_resources();
    initialize_internal_time_parameter();
    const auto logger = get_logger();

    SPDLOG_LOGGER_TRACE(
        logger, "ThermalMathematicalModel: constructor with custom resources");
}

ThermalMathematicalModel::~ThermalMathematicalModel() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_TRACE(logger, "ThermalMathematicalModel: destructor");
}

ThermalNetwork& ThermalMathematicalModel::network() noexcept {
    return *_network;
}

const ThermalNetwork& ThermalMathematicalModel::network() const noexcept {
    return *_network;
}

std::shared_ptr<ThermalNetwork>
ThermalMathematicalModel::network_ptr() noexcept {
    return _network;
}

std::shared_ptr<const ThermalNetwork> ThermalMathematicalModel::network_ptr()
    const noexcept {
    return _network;
}

Nodes& ThermalMathematicalModel::nodes() noexcept { return _network->nodes(); }

const Nodes& ThermalMathematicalModel::nodes() const noexcept {
    return _network->nodes();
}

std::shared_ptr<Nodes> ThermalMathematicalModel::nodes_ptr() noexcept {
    return _network->nodes_ptr();
}

std::shared_ptr<const Nodes> ThermalMathematicalModel::nodes_ptr()
    const noexcept {
    return _network->nodes_ptr();
}

ConductiveCouplings& ThermalMathematicalModel::conductive_couplings() noexcept {
    return _network->conductive_couplings();
}

const ConductiveCouplings& ThermalMathematicalModel::conductive_couplings()
    const noexcept {
    return _network->conductive_couplings();
}

RadiativeCouplings& ThermalMathematicalModel::radiative_couplings() noexcept {
    return _network->radiative_couplings();
}

const RadiativeCouplings& ThermalMathematicalModel::radiative_couplings()
    const noexcept {
    return _network->radiative_couplings();
}

void ThermalMathematicalModel::add_node(Node node) { _network->add_node(node); }

void ThermalMathematicalModel::add_node(Index node_num) {
    Node node(static_cast<int>(node_num));
    _network->add_node(node);
}

void ThermalMathematicalModel::add_conductive_coupling(Index node_num_1,
                                                       Index node_num_2,
                                                       double value) {
    conductive_couplings().add_coupling(node_num_1, node_num_2, value);
}

void ThermalMathematicalModel::add_radiative_coupling(Index node_num_1,
                                                      Index node_num_2,
                                                      double value) {
    radiative_couplings().add_coupling(node_num_1, node_num_2, value);
}

void ThermalMathematicalModel::add_conductive_coupling(Coupling coupling) {
    conductive_couplings().add_coupling(coupling);
}

void ThermalMathematicalModel::add_radiative_coupling(Coupling coupling) {
    radiative_couplings().add_coupling(coupling);
}

void ThermalMathematicalModel::add_time_variable(const std::string& name,
                                                 Eigen::VectorXd x_data,
                                                 Eigen::VectorXd y_data,
                                                 InterpolationMethod interp,
                                                 ExtrapolationMethod extrap) {
    if (_temperature_variable_names.contains(name)) {
        throw std::invalid_argument("Variable name '" + name +
                                    "' is already used by a "
                                    "TemperatureVariable");
    }

    const auto [iterator, inserted] = _time_variables.try_emplace(
        name, name,
        LookupTable1D(std::move(x_data), std::move(y_data), interp, extrap),
        parameters, thermal_data, std::addressof(time));
    if (!inserted) {
        throw std::invalid_argument("TimeVariable '" + name +
                                    "' already exists");
    }

    (void)iterator;
}

void ThermalMathematicalModel::remove_time_variable(const std::string& name) {
    _time_variables.erase(name);
}

bool ThermalMathematicalModel::has_time_variable(
    const std::string& name) const noexcept {
    return _time_variables.contains(name);
}

const TimeVariable& ThermalMathematicalModel::get_time_variable(
    const std::string& name) const {
    const auto iterator = _time_variables.find(name);
    if (iterator == _time_variables.end()) {
        throw std::out_of_range("TimeVariable '" + name + "' does not exist");
    }

    return iterator->second;
}

void ThermalMathematicalModel::add_temperature_variable(
    const std::string& name, Eigen::VectorXd x_data, Eigen::VectorXd y_data,
    InterpolationMethod interp, ExtrapolationMethod extrap) {
    if (parameters.contains(name) || _time_variables.contains(name)) {
        throw std::invalid_argument("Variable name '" + name +
                                    "' is already in use");
    }

    const auto [iterator, inserted] = _temperature_variables.try_emplace(
        name, name,
        LookupTable1D(std::move(x_data), std::move(y_data), interp, extrap));
    if (!inserted) {
        throw std::invalid_argument("TemperatureVariable '" + name +
                                    "' already exists");
    }

    _temperature_variable_names.insert(iterator->first);
}

void ThermalMathematicalModel::remove_temperature_variable(
    const std::string& name) {
    _temperature_variables.erase(name);
    _temperature_variable_names.erase(name);
}

bool ThermalMathematicalModel::has_temperature_variable(
    const std::string& name) const noexcept {
    return _temperature_variables.contains(name);
}

const TemperatureVariable& ThermalMathematicalModel::get_temperature_variable(
    const std::string& name) const {
    const auto iterator = _temperature_variables.find(name);
    if (iterator == _temperature_variables.end()) {
        throw std::out_of_range("TemperatureVariable '" + name +
                                "' does not exist");
    }

    return iterator->second;
}

void ThermalMathematicalModel::callback_solver_loop() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_DEBUG(logger,
                        "ThermalMathematicalModel: callback_solver_loop");

    if (!callbacks_active) {
        return;
    }

    if (internal_callbacks_active) {
        internal_callback_solver_loop();
    }
    if (c_callbacks_active) {
        c_extern_callback_solver_loop(this);
    }
    if (python_callbacks_active) {
        python_extern_callback_solver_loop();
    }
}

void ThermalMathematicalModel::callback_transient_time_change() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_DEBUG(
        logger, "ThermalMathematicalModel: callback_transient_time_change");

    if (!callbacks_active) {
        return;
    }

    if (internal_callbacks_active) {
        internal_callback_transient_time_change();
    }
    if (c_callbacks_active) {
        c_extern_callback_transient_time_change(this);
    }
    if (python_callbacks_active) {
        python_extern_callback_transient_time_change();
    }
}

void ThermalMathematicalModel::callback_transient_after_timestep() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_DEBUG(
        logger, "ThermalMathematicalModel: callback_transient_after_timestep");

    if (!callbacks_active) {
        return;
    }

    if (internal_callbacks_active) {
        internal_callback_transient_after_timestep();
    }
    if (c_callbacks_active) {
        c_extern_callback_transient_after_timestep(this);
    }
    if (python_callbacks_active) {
        python_extern_callback_transient_after_timestep();
    }
}

void ThermalMathematicalModel::internal_callback_common() {
    if (_time_parameter_ptr != nullptr) {
        *_time_parameter_ptr = time;
    }

    for (auto& [name, variable] : _time_variables) {
        (void)name;
        variable.update();
    }

    if (parameters.is_structure_locked() && formulas.is_compiled_current()) {
        formulas.apply_compiled_formulas();
    } else {
        formulas.apply_formulas();
    }

    if (python_formulas_active) {
        python_apply_formulas();
    }
}

void ThermalMathematicalModel::internal_callback_solver_loop() {
    internal_callback_common();
}

void ThermalMathematicalModel::internal_callback_transient_time_change() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_DEBUG(logger,
                        "ThermalMathematicalModel: internal transient change");

    internal_callback_common();
}

void ThermalMathematicalModel::internal_callback_transient_after_timestep() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_DEBUG(logger,
                        "ThermalMathematicalModel: internal after timestep");

    internal_callback_common();
}

void ThermalMathematicalModel::associate_resources() {
    _formulas_shptr->associate(_network, _parameters_shptr);
    _formulas_shptr->set_temperature_variable_names(
        std::addressof(_temperature_variable_names));
    _thermal_data_shptr->associate(_network);
}

void ThermalMathematicalModel::initialize_internal_time_parameter() {
    if (!_parameters_shptr->contains("time")) {
        _parameters_shptr->add_internal_parameter("time", 0.0);
    }

    if (!_parameters_shptr->is_internal_parameter("time")) {
        SPDLOG_LOGGER_WARN(get_logger(),
                           "Parameter 'time' already exists and is not an "
                           "internal parameter; runtime time synchronization "
                           "is disabled");
        _time_parameter_ptr = nullptr;
        return;
    }

    const auto time_idx = _parameters_shptr->get_idx("time");
    if (!time_idx.has_value()) {
        _time_parameter_ptr = nullptr;
        return;
    }

    _time_parameter_ptr = _parameters_shptr->get_double_ptr(*time_idx);
}

}  // namespace pycanha
