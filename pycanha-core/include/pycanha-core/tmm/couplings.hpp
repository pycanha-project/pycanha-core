#pragma once

#include <Eigen/Sparse>
#include <memory>
#include <optional>
#include <tuple>

#include "pycanha-core/parameters.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/nodes.hpp"

namespace pycanha {

/**
 * Aggregates the thermal coupling matrices for a network of `Nodes`.
 *
 * The class preserves the legacy public interface while delegating the actual
 * matrix management to a composed `CouplingMatrices` instance.
 */
class Couplings {
    friend class Nodes;
    friend class CouplingMatrices;
    friend class ConductiveCouplings;
    friend class RadiativeCouplings;
    friend class ThermalNetwork;
    friend class SteadyStateNonSymmetricSolver;

  public:
    explicit Couplings(std::shared_ptr<Nodes> nodes) noexcept;
    ~Couplings() = default;

    Couplings(const Couplings&) = default;
    Couplings& operator=(const Couplings&) = default;
    Couplings(Couplings&&) noexcept = default;
    Couplings& operator=(Couplings&&) noexcept = default;

    [[nodiscard]] const CouplingMatrices& get_coupling_matrices()
        const noexcept;
    [[nodiscard]] CouplingMatrices& get_coupling_matrices() noexcept;

    [[nodiscard]] double get_coupling_value(int usr_num_1, int usr_num_2);
    void set_coupling_value(int usr_num_1, int usr_num_2, double value);

    void add_ovw_coupling(int node_num_1, int node_num_2, double value);
    void add_ovw_coupling(const Coupling& coupling);
    void add_ovw_coupling_verbose(int node_num_1, int node_num_2, double value);
    void add_ovw_coupling_verbose(const Coupling& coupling);
    void add_sum_coupling(int node_num_1, int node_num_2, double value);
    void add_sum_coupling(const Coupling& coupling);
    void add_sum_coupling_verbose(int node_num_1, int node_num_2, double value);
    void add_sum_coupling_verbose(const Coupling& coupling);
    void add_new_coupling(int node_num_1, int node_num_2, double value);
    void add_new_coupling(const Coupling& coupling);
    void add_coupling(int usr_num_1, int usr_num_2, double value);
    void add_coupling(const Coupling& coupling);

    [[nodiscard]] double* get_coupling_value_ref(int usr_num_1, int usr_num_2);
    [[nodiscard]] IntAddress get_coupling_value_address(int usr_num_1,
                                                        int usr_num_2);

    [[nodiscard]] bool coupling_exists(int node_num_1, int node_num_2);
    [[nodiscard]] Coupling get_coupling_from_coupling_idx(Index cidx);

  private:
    void synchronize_structure();

    [[nodiscard]] std::optional<std::pair<Index, Index>>
    get_indices_from_node_numbers(int node_num_1, int node_num_2);

    [[nodiscard]] std::tuple<Index, Index,
                             Eigen::SparseMatrix<double, Eigen::RowMajor>*>
    get_indices_and_sparse_from_node_numbers(int node_num_1, int node_num_2);

    [[nodiscard]] bool is_coupling_trivial_zero_from_node_numbers(
        int node_num_1, int node_num_2);

    std::shared_ptr<Nodes> _nodes;
    CouplingMatrices _matrices;
};

}  // namespace pycanha
