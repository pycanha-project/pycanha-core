#pragma once

#include <memory>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/couplings.hpp"
#include "pycanha-core/tmm/nodes.hpp"

namespace pycanha {

class ConductiveCouplings {
    friend class ThermalNetwork;

  public:
    explicit ConductiveCouplings(std::shared_ptr<Nodes> nodes) noexcept;
    ConductiveCouplings(const ConductiveCouplings&) = default;
    ConductiveCouplings& operator=(const ConductiveCouplings&) = default;
    ConductiveCouplings(ConductiveCouplings&&) noexcept = default;
    ConductiveCouplings& operator=(ConductiveCouplings&&) noexcept = default;
    ~ConductiveCouplings() = default;

    void add_coupling(Index node_num_1, Index node_num_2, double value);
    void add_coupling(const Coupling& coupling);

    void set_coupling_value(Index node_num_1, Index node_num_2, double value);
    [[nodiscard]] double get_coupling_value(Index node_num_1, Index node_num_2);
    [[nodiscard]] double* get_coupling_value_ref(Index node_num_1,
                                                 Index node_num_2);

  private:
    [[nodiscard]] Couplings& couplings() noexcept { return _couplings; }
    [[nodiscard]] const Couplings& couplings() const noexcept {
        return _couplings;
    }

    Couplings _couplings;
};

}  // namespace pycanha
