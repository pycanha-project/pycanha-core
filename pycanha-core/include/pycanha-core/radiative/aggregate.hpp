#pragma once

#include <Eigen/Dense>
#include <span>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/radiative/results.hpp"

namespace pycanha::radiative {

// Face -> node reduction (results are kept at face level; node views are
// computed on demand). `node_numbers` is the GMM per-face-slot node array
// (NO_NODE = unassigned, dropped from every aggregate); `face_areas` the
// per-slot areas. Rows/cols of the aggregated outputs are indexed by the
// position of the node in aggregate_nodes(node_numbers).

// Sorted unique node numbers with NO_NODE removed — the row/col labels of
// the aggregated outputs.
[[nodiscard]] std::vector<NodeNum> aggregate_nodes(
    std::span<const NodeNum> node_numbers);

struct AggregateResult {
    SparseMatrix matrix;
    // Total dropped onto the node diagonal by the symmetric reduction: face
    // pairs whose two faces belong to the same node. Reported rather than
    // silently discarded, because a large value means the node is not as
    // isothermal as treating it as one node assumes.
    double intra_node_total = 0.0;
};

// out(m, n) = sum over faces i of node m, j of node n of face_matrix(i, j),
// with i <= j and the result canonicalised into the UPPER TRIANGLE.
//
// The input is the extensive (m^2) face matrix a VF result stores, so the
// reduction is a plain sum — there is no area weighting to apply, and
// applying one anyway would scale every entry by a further A_i and yield a
// dimensionally meaningless m^4 matrix that still solves and still looks
// plausible. To condense an INTENSIVE matrix (Gebhart factors, say), scale
// its rows by the face areas first.
//
// Two things do not survive the mapping from faces to nodes, and both are
// handled here:
//  - node numbers are assigned independently of face-slot numbering, so a
//    face pair i < j can land on a node pair m > n; every write is
//    canonicalised to (min(m, n), max(m, n)) or the output would be an
//    arbitrary mix of both triangles rather than a triangle,
//  - pairs whose faces share a node fall on the diagonal. A node is
//    isothermal by definition, so radiation it exchanges with itself
//    transports no heat and the coupling network has no slot for it; the
//    diagonal is dropped and its total reported.
//
// Matrix results carry the virtual space/inactive/lost bucket columns after
// the real face columns. This overload drops them (missing column labels
// count as NO_NODE); use the row/column overload below to map a bucket to a
// real node (e.g. the space node).
[[nodiscard]] AggregateResult aggregate_matrix(
    const SparseMatrix& face_matrix, std::span<const NodeNum> node_numbers);

// General form: rows and columns labeled independently. `row_node_numbers`
// has one entry per matrix row, `col_node_numbers` one per matrix COLUMN —
// including the virtual bucket columns, so those can be assigned nodes or
// NO_NODE. Output rows/cols are indexed by position in
// aggregate_nodes(row_node_numbers) / aggregate_nodes(col_node_numbers).
//
// Rows and columns carry independent label sets here, so there is no
// triangle to canonicalise into and no diagonal that means self-coupling:
// this overload sums every mapped entry exactly where it lands and reports
// nothing discarded.
[[nodiscard]] AggregateResult aggregate_matrix(
    const SparseMatrix& face_matrix, std::span<const NodeNum> row_node_numbers,
    std::span<const NodeNum> col_node_numbers);

// W per node from W/m^2 per face: out(m) = sum_{i in m} flux[i] * area[i].
[[nodiscard]] Eigen::VectorXd aggregate_flux(
    const Eigen::VectorXd& face_flux_w_m2,
    std::span<const NodeNum> node_numbers, std::span<const double> face_areas);

}  // namespace pycanha::radiative
