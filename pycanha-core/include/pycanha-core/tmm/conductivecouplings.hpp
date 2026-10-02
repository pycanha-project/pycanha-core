#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/bulk.hpp"
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

    /// Bulk insertion by node number, see Couplings::add_couplings. The
    /// default merge overwrites, as add_coupling does.
    BulkReport add_couplings(std::span<const NodeNum> node_nums_1,
                             std::span<const NodeNum> node_nums_2,
                             std::span<const double> values,
                             CouplingMerge merge = CouplingMerge::OVERWRITE);
    BulkReport add_couplings(std::span<const std::int64_t> node_nums_1,
                             std::span<const std::int64_t> node_nums_2,
                             std::span<const double> values,
                             CouplingMerge merge = CouplingMerge::OVERWRITE);
    /// Bulk insertion by internal index, see Couplings::append_couplings.
    BulkReport append_couplings(std::span<const CouplingChunk> chunks);
    /// Bulk getters and setters, see Couplings::get_values / set_values.
    BulkReport get_values(std::span<const NodeNum> node_nums_1,
                          std::span<const NodeNum> node_nums_2,
                          std::span<double> values);
    BulkReport set_values(std::span<const NodeNum> node_nums_1,
                          std::span<const NodeNum> node_nums_2,
                          std::span<const double> values);
    [[nodiscard]] Couplings::CouplingArrays to_arrays();

    [[nodiscard]] CouplingMatrices& matrices() noexcept {
        return _couplings.get_coupling_matrices();
    }
    [[nodiscard]] const CouplingMatrices& matrices() const noexcept {
        return _couplings.get_coupling_matrices();
    }

  private:
    [[nodiscard]] Couplings& couplings() noexcept { return _couplings; }
    [[nodiscard]] const Couplings& couplings() const noexcept {
        return _couplings;
    }

    Couplings _couplings;
};

}  // namespace pycanha
