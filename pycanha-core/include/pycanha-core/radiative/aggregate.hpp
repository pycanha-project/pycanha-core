#pragma once

#include <Eigen/Dense>
#include <span>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/radiative/sparse.hpp"

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

// out(m, n) = sum over faces i of node m, j of node n of
//             face_areas[i] * face_matrix(i, j).
// This is the area-weighted (extensive, m^2) reduction: with rows pre-scaled
// by the band emissivity it is exactly the node-level GR sum; divide by node
// areas afterwards for intensive (view-factor-like) node matrices.
//
// Matrix results carry the virtual space/inactive/lost bucket columns after
// the real face columns. This overload drops them (missing column labels
// count as NO_NODE); use the row/column overload below to map a bucket to a
// real node (e.g. the space node).
[[nodiscard]] SparseF64 aggregate_matrix(const SparseF64& face_matrix,
                                         std::span<const NodeNum> node_numbers,
                                         std::span<const double> face_areas);

// General form: rows and columns labeled independently. `row_node_numbers`
// has one entry per matrix row, `col_node_numbers` one per matrix COLUMN —
// including the virtual bucket columns, so those can be assigned nodes or
// NO_NODE. Output rows/cols are indexed by position in
// aggregate_nodes(row_node_numbers) / aggregate_nodes(col_node_numbers).
[[nodiscard]] SparseF64 aggregate_matrix(
    const SparseF64& face_matrix, std::span<const NodeNum> row_node_numbers,
    std::span<const NodeNum> col_node_numbers,
    std::span<const double> face_areas);

// W per node from W/m^2 per face: out(m) = sum_{i in m} flux[i] * area[i].
[[nodiscard]] Eigen::VectorXd aggregate_flux(
    const Eigen::VectorXd& face_flux_w_m2,
    std::span<const NodeNum> node_numbers, std::span<const double> face_areas);

}  // namespace pycanha::radiative
