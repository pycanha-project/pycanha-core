#pragma once

// The CPU side of RadiativeRetracer: reducing a face-level exchange result to
// node couplings in the model's internal node order, writing them into the
// model's radiative coupling slots, and the load of the couplings that have
// no slot. Kept apart from the retracer so it can be tested without a GPU.

#include <span>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/radiative/results.hpp"

namespace pycanha {
class CouplingMatrices;
}  // namespace pycanha

namespace pycanha::radiative::detail {

// One trace reduced to nodes.
struct NodeCouplings {
    // Upper triangle by internal node index, num_nodes x num_nodes [m^2].
    SparseMatrix couplings;
    // Totals of the space, inactive and lost columns of the faces with a
    // node [m^2]; the space total is in `couplings` too, when there is a
    // space node.
    double space_total = 0.0;
    double inactive_total = 0.0;
    double lost_total = 0.0;
    // Couplings between two faces of the same node, which transport no heat.
    double intra_node_total = 0.0;
};

// Reduces the extensive face factors of an exchange result (upper triangle,
// num_faces + num_virtual_columns columns) to nodes. `face_node` holds the
// internal node index of every face, or -1 for a face without a node, whose
// couplings are dropped. The space column goes to `space_index` (-1 drops
// it); the inactive and lost columns are only totalled.
[[nodiscard]] NodeCouplings reduce_to_nodes(const SparseMatrix& factors,
                                            std::span<const Index> face_node,
                                            Index space_index, Index num_nodes);

// The union of the patterns of `traces` (all num_nodes x num_nodes, upper
// triangle), as an upper-triangular pattern with zero values.
[[nodiscard]] SparseMatrix pattern_union(
    std::span<const SparseMatrix* const> traces, Index num_nodes);

// What split_into_slots() did.
struct SplitResult {
    // The entries of the trace with no slot, upper triangle by internal node
    // index, num_nodes x num_nodes; 0 x 0 when every entry had a slot.
    SparseMatrix overflow;
    double overflow_total = 0.0;
    double trace_total = 0.0;
};

// Writes every entry of `trace` (upper triangle by internal node index) into
// its slot of the radiative coupling blocks, and 0 into every slot the trace
// has no entry for. The blocks must be compressed; their structure is not
// changed. One merge per row, O(entries + slots).
[[nodiscard]] SplitResult split_into_slots(const SparseMatrix& trace,
                                           CouplingMatrices& matrices);

// The load [W] of the overflow couplings at the temperatures `t` [K]:
// load_i = sigma * sum_j R_ij (T_j^4 - T_i^4). `load` is resized to t.size()
// and overwritten.
void overflow_load(const SparseMatrix& overflow, std::span<const double> t,
                   std::vector<double>& load);

}  // namespace pycanha::radiative::detail
