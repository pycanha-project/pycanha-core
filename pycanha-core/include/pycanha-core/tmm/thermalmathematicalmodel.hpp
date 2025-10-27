#pragma once

#include <functional>
#include <memory>
#include <string>

#include "pycanha-core/globals.hpp"

namespace pycanha {

class Coupling;
class ConductiveCouplings;
class Formulas;
class Node;
class Nodes;
class Parameters;
class RadiativeCouplings;
class ThermalData;
class ThermalNetwork;

class ThermalMathematicalModel {
  public:
    ThermalMathematicalModel(const ThermalMathematicalModel&) = delete;
    ThermalMathematicalModel& operator=(const ThermalMathematicalModel&) =
        delete;
    ThermalMathematicalModel(ThermalMathematicalModel&&) noexcept = delete;
    ThermalMathematicalModel& operator=(ThermalMathematicalModel&&) noexcept =
        delete;

    explicit ThermalMathematicalModel(std::string model_name);
    ThermalMathematicalModel(std::string model_name,
                             std::shared_ptr<Nodes> nodes,
                             std::shared_ptr<ConductiveCouplings> conductive,
                             std::shared_ptr<RadiativeCouplings> radiative);
    ThermalMathematicalModel(std::string model_name,
                             std::shared_ptr<Nodes> nodes,
                             std::shared_ptr<ConductiveCouplings> conductive,
                             std::shared_ptr<RadiativeCouplings> radiative,
                             std::shared_ptr<Parameters> parameters,
                             std::shared_ptr<Formulas> formulas,
                             std::shared_ptr<ThermalData> thermal_data);
    ~ThermalMathematicalModel();

    [[nodiscard]] ThermalNetwork& network() noexcept;
    [[nodiscard]] const ThermalNetwork& network() const noexcept;
    [[nodiscard]] std::shared_ptr<ThermalNetwork> network_ptr() noexcept;
    [[nodiscard]] std::shared_ptr<const ThermalNetwork> network_ptr()
        const noexcept;

    [[nodiscard]] Nodes& nodes() noexcept;
    [[nodiscard]] const Nodes& nodes() const noexcept;
    [[nodiscard]] std::shared_ptr<Nodes> nodes_ptr() noexcept;
    [[nodiscard]] std::shared_ptr<const Nodes> nodes_ptr() const noexcept;

    [[nodiscard]] ConductiveCouplings& conductive_couplings() noexcept;
    [[nodiscard]] const ConductiveCouplings& conductive_couplings()
        const noexcept;

    [[nodiscard]] RadiativeCouplings& radiative_couplings() noexcept;
    [[nodiscard]] const RadiativeCouplings& radiative_couplings()
        const noexcept;

    void add_node(Node node);
    void add_node(Index node_num);
    void add_conductive_coupling(Index node_num_1, Index node_num_2,
                                 double value);
    void add_radiative_coupling(Index node_num_1, Index node_num_2,
                                double value);
    void add_conductive_coupling(Coupling coupling);
    void add_radiative_coupling(Coupling coupling);

    bool callbacks_active = true;
    bool internal_callbacks_active = true;
    bool c_callbacks_active = false;
    bool python_callbacks_active = false;
    bool python_formulas_active = true;

    void callback_solver_loop();
    void callback_transient_time_change();
    void callback_transient_after_timestep();

    void internal_callback_solver_loop();
    void internal_callback_transient_time_change();
    void internal_callback_transient_after_timestep();
    std::function<void()> python_apply_formulas = []() {};

    void (*c_extern_callback_solver_loop)(ThermalMathematicalModel*) =
        [](ThermalMathematicalModel*) {};
    void (*c_extern_callback_transient_time_change)(ThermalMathematicalModel*) =
        [](ThermalMathematicalModel*) {};
    void (*c_extern_callback_transient_after_timestep)(
        ThermalMathematicalModel*) = [](ThermalMathematicalModel*) {};

    std::function<void()> python_extern_callback_solver_loop = []() {};
    std::function<void()> python_extern_callback_transient_time_change = []() {
    };
    std::function<void()> python_extern_callback_transient_after_timestep =
        []() {};

  private:
    std::shared_ptr<ThermalNetwork> _network;
    std::shared_ptr<Parameters> _parameters_shptr;
    std::shared_ptr<Formulas> _formulas_shptr;
    std::shared_ptr<ThermalData> _thermal_data_shptr;

    void associate_resources();
    inline void internal_callback_common();

  public:
    std::string name;
    double time{0.0};

    Parameters& parameters;
    Formulas& formulas;
    ThermalData& thermal_data;
};

}  // namespace pycanha
