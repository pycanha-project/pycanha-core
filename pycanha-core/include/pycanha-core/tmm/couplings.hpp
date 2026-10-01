#pragma once

#include <Eigen/Sparse>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/bulk.hpp"
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
    ~Couplings();

    // Each instance registers itself with its Nodes, so that a node inserted
    // in the middle of its block, or removed, remaps the stored couplings.
    Couplings(const Couplings& other);
    Couplings& operator=(const Couplings& other);
    Couplings(Couplings&& other) noexcept;
    Couplings& operator=(Couplings&& other) noexcept;

    [[nodiscard]] const CouplingMatrices& get_coupling_matrices()
        const noexcept;
    [[nodiscard]] CouplingMatrices& get_coupling_matrices() noexcept;

    [[nodiscard]] double get_coupling_value(Index node_num_1, Index node_num_2);
    void set_coupling_value(Index node_num_1, Index node_num_2, double value);

    void add_ovw_coupling(Index node_num_1, Index node_num_2, double value);
    void add_ovw_coupling(const Coupling& coupling);
    void add_ovw_coupling_verbose(Index node_num_1, Index node_num_2,
                                  double value);
    void add_ovw_coupling_verbose(const Coupling& coupling);
    void add_sum_coupling(Index node_num_1, Index node_num_2, double value);
    void add_sum_coupling(const Coupling& coupling);
    void add_sum_coupling_verbose(Index node_num_1, Index node_num_2,
                                  double value);
    void add_sum_coupling_verbose(const Coupling& coupling);
    void add_new_coupling(Index node_num_1, Index node_num_2, double value);
    void add_new_coupling(const Coupling& coupling);
    void add_coupling(Index node_num_1, Index node_num_2, double value);
    void add_coupling(const Coupling& coupling);

    [[nodiscard]] double* get_coupling_value_ref(Index node_num_1,
                                                 Index node_num_2);
    [[nodiscard]] IntAddress get_coupling_value_address(Index node_num_1,
                                                        Index node_num_2);

    [[nodiscard]] bool coupling_exists(Index node_num_1, Index node_num_2);
    [[nodiscard]] Coupling get_coupling_from_coupling_idx(Index cidx);

    /// General bulk insertion by node number.
    /**
     * The arrays are read in place, never copied, and every value is written
     * once, into the matrix storage. Entry i couples node_nums_1[i] and
     * node_nums_2[i], in either order. Dropped, and reported: an unknown
     * node, the same node on both ends, a negative or non-finite value, and
     * the whole call when the three arrays differ in length.
     *
     * The fastest path is taken per block (dd, db, bb) when the call's
     * entries of that block are sorted by (lower internal index, higher
     * internal index) and come after every coupling already stored there --
     * for an all-diffusive model simply "sorted by (smaller node number,
     * larger node number)", as a builder or a numpy grid produces them. They
     * are then appended directly, O(k). Otherwise they are sorted (O(k) by
     * row plus a sort inside each row) and merged into the block in place,
     * O(stored + k). A coupling that already exists, or that the call gives
     * more than once, is resolved by @p merge.
     *
     * Pointers into the coupling values (get_coupling_value_ref) are
     * invalidated.
     */
    BulkReport add_couplings(std::span<const NodeNum> node_nums_1,
                             std::span<const NodeNum> node_nums_2,
                             std::span<const double> values,
                             CouplingMerge merge = CouplingMerge::OVERWRITE);
    /// The same, reading 64-bit node numbers in place. A number outside the
    /// NodeNum range is rejected, never wrapped.
    BulkReport add_couplings(std::span<const std::int64_t> node_nums_1,
                             std::span<const std::int64_t> node_nums_2,
                             std::span<const double> values,
                             CouplingMerge merge = CouplingMerge::OVERWRITE);

    /// Specific bulk insertion by internal node index, see
    /// CouplingMatrices::append_couplings. The matrices are first sized to
    /// the current nodes.
    BulkReport append_couplings(std::span<const CouplingChunk> chunks);

    /// Values of many couplings; a missing coupling or node reads as NaN
    /// and is reported.
    BulkReport get_values(std::span<const NodeNum> node_nums_1,
                          std::span<const NodeNum> node_nums_2,
                          std::span<double> values);
    /// Changes the values of existing couplings, as set_coupling_value does;
    /// a missing coupling or node, or a negative value, is skipped and
    /// reported.
    BulkReport set_values(std::span<const NodeNum> node_nums_1,
                          std::span<const NodeNum> node_nums_2,
                          std::span<const double> values);

    /// Every stored coupling as three arrays, in internal order (the dd,
    /// then the db, then the bb block, each row by row).
    struct CouplingArrays {
        std::vector<NodeNum> node_1;
        std::vector<NodeNum> node_2;
        std::vector<double> values;
    };
    [[nodiscard]] CouplingArrays to_arrays();

    /// Sizes the matrices to the current nodes. Nodes appended at the end of
    /// their block grow the matrices lazily, here; every accessor calls it.
    void synchronize_structure();

  private:
    // Called by Nodes when the nodes of `type`'s block move: old_to_new maps
    // each old position in the block to its new one, -1 when removed.
    void remap_nodes(char type, std::span<const Index> old_to_new);
    // Called by Nodes when nodes change type and move across blocks:
    // old_to_new maps every old internal index to its new one, and the old
    // order had old_diff_count diffusive nodes.
    void reorder_nodes(std::span<const Index> old_to_new, Index old_diff_count);

    template <typename NodeNumber>
    BulkReport add_couplings_impl(std::span<const NodeNumber> node_nums_1,
                                  std::span<const NodeNumber> node_nums_2,
                                  std::span<const double> values,
                                  CouplingMerge merge);

    [[nodiscard]] std::optional<std::pair<Index, Index>>
    get_indices_from_node_numbers(Index node_num_1, Index node_num_2);

    [[nodiscard]] std::tuple<Index, Index,
                             Eigen::SparseMatrix<double, Eigen::RowMajor>*>
    get_indices_and_sparse_from_node_numbers(Index node_num_1,
                                             Index node_num_2);

    [[nodiscard]] bool is_coupling_trivial_zero_from_node_numbers(
        Index node_num_1, Index node_num_2);

    std::shared_ptr<Nodes> _nodes;
    CouplingMatrices _matrices;

    // Nodes::structure_version() the matrix sizes correspond to.
    std::uint64_t _synced_version = std::numeric_limits<std::uint64_t>::max();

    // Links of the intrusive observer list kept by _nodes.
    Couplings* _previous_observer = nullptr;
    Couplings* _next_observer = nullptr;
};

}  // namespace pycanha
