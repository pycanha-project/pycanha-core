#pragma once

#include <memory>

#include "pycanha-core/parameters.hpp"

namespace pycanha {

class Node;
class Nodes;
class ConductiveCouplings;
class RadiativeCouplings;
class CouplingMatrices;
class SteadyStateNonSymmetricSolver;

class ThermalNetwork {
    friend class Nodes;
    friend class CouplingMatrices;
    friend class SteadyStateNonSymmetricSolver;

  public:
    ThermalNetwork();
    ThermalNetwork(std::shared_ptr<Nodes> nodes,
                   std::shared_ptr<ConductiveCouplings> conductive,
                   std::shared_ptr<RadiativeCouplings> radiative);

    ThermalNetwork(const ThermalNetwork&) = delete;
    ThermalNetwork& operator=(const ThermalNetwork&) = delete;
    ThermalNetwork(ThermalNetwork&&) noexcept = default;
    ThermalNetwork& operator=(ThermalNetwork&&) noexcept = default;
    ~ThermalNetwork() = default;

    void add_node(Node& node);
    void remove_node(Index node_num);

    [[nodiscard]] Nodes& nodes() noexcept;
    [[nodiscard]] const Nodes& nodes() const noexcept;

    [[nodiscard]] ConductiveCouplings& conductive_couplings() noexcept;
    [[nodiscard]] const ConductiveCouplings& conductive_couplings()
        const noexcept;

    [[nodiscard]] RadiativeCouplings& radiative_couplings() noexcept;
    [[nodiscard]] const RadiativeCouplings& radiative_couplings()
        const noexcept;

    [[nodiscard]] std::shared_ptr<Nodes> nodes_ptr() noexcept;
    [[nodiscard]] std::shared_ptr<const Nodes> nodes_ptr() const noexcept;

  private:
    std::shared_ptr<Nodes> _nodes;
    std::shared_ptr<ConductiveCouplings> _conductive_couplings;
    std::shared_ptr<RadiativeCouplings> _radiative_couplings;
};

}  // namespace pycanha
