#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

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

    if constexpr (DEBUG) {
        std::cout << "ThermalMathematicalModel: default constructor" << '\n';
    }
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

    if constexpr (DEBUG) {
        std::cout << "ThermalMathematicalModel: constructor with shared"
                  << " nodes" << '\n';
    }
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

    if constexpr (DEBUG) {
        std::cout << "ThermalMathematicalModel: constructor with custom"
                  << " resources" << '\n';
    }
}

ThermalMathematicalModel::~ThermalMathematicalModel() {
    if constexpr (DEBUG) {
        std::cout << "ThermalMathematicalModel: destructor " << this << '\n';
    }
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

void ThermalMathematicalModel::callback_solver_loop() {
    if constexpr (DEBUG) {
        std::cout << "ThermalMathematicalModel: callback_solver_loop" << '\n';
    }

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
    if constexpr (DEBUG) {
        std::cout << "ThermalMathematicalModel: callback_transient_time_change"
                  << '\n';
    }

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
    if constexpr (DEBUG) {
        std::cout
            << "ThermalMathematicalModel: callback_transient_after_timestep"
            << '\n';
    }

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
    formulas.apply_formulas();

    if (python_formulas_active) {
        python_apply_formulas();
    }
}

void ThermalMathematicalModel::internal_callback_solver_loop() {
    internal_callback_common();
}

void ThermalMathematicalModel::internal_callback_transient_time_change() {
    if constexpr (DEBUG) {
        std::cout << "ThermalMathematicalModel: internal transient change"
                  << '\n';
    }

    internal_callback_common();
}

void ThermalMathematicalModel::internal_callback_transient_after_timestep() {
    if constexpr (DEBUG) {
        std::cout << "ThermalMathematicalModel: internal after timestep"
                  << '\n';
    }

    internal_callback_common();
}

void ThermalMathematicalModel::associate_resources() {
    _formulas_shptr->associate(_network, _parameters_shptr);
    _thermal_data_shptr->associate(_network);
}

}  // namespace pycanha
