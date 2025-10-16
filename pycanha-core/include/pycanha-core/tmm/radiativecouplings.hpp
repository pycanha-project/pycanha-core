#pragma once

#include <memory>

#include "pycanha-core/parameters.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/couplings.hpp"
#include "pycanha-core/tmm/nodes.hpp"

namespace pycanha {

class RadiativeCouplings {
    friend class ThermalNetwork;

  public:
    explicit RadiativeCouplings(std::shared_ptr<Nodes> nodes) noexcept;
    RadiativeCouplings(const RadiativeCouplings&) = default;
    RadiativeCouplings& operator=(const RadiativeCouplings&) = default;
    RadiativeCouplings(RadiativeCouplings&&) noexcept = default;
    RadiativeCouplings& operator=(RadiativeCouplings&&) noexcept = default;
    ~RadiativeCouplings() = default;

    void add_coupling(Index node_num_1, Index node_num_2, double value);
    void add_coupling(const Coupling& coupling);

    void set_coupling_value(Index node_num_1, Index node_num_2, double value);
    [[nodiscard]] double get_coupling_value(Index node_num_1, Index node_num_2);

  private:
    [[nodiscard]] Couplings& couplings() noexcept { return _couplings; }
    [[nodiscard]] const Couplings& couplings() const noexcept {
        return _couplings;
    }

    Couplings _couplings;
};

}  // namespace pycanha