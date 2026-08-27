#pragma once

#include <Eigen/Dense>
#include <span>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/radiative/results.hpp"

namespace pycanha::radiative {

// CPU Gebhart services (no Vulkan, built unconditionally): re-derive
// radiative exchange factors from a geometric VF matrix and per-face
// band emissivities without re-tracing — the diffuse-gray fast path. The
// MCRT exchange kernel computes the same quantity directly, so the two
// paths are cross-checkable within Monte-Carlo tolerance.

// Both solves read the VF matrix in the form VfResult stores it: the UPPER
// TRIANGLE of the symmetric, extensive G_ij = A_i F_ij. Face areas are
// therefore required, since recovering the two view factors from one stored
// entry is exactly F_ij = G_ij/A_i and F_ji = G_ij/A_j. A stored entry below
// the diagonal is rejected rather than folded in: it would double-count the
// coupling it duplicates.
//
// `space_fraction_policy` decides what each row's deficit (1 - row_sum)
// means: 1.0 (default) treats it as a real view to space; 0.0 renormalizes
// the row to sum to one (closed-enclosure assumption, i.e. the deficit is
// Monte-Carlo noise); values in between scale the deficit accordingly. Note
// that renormalizing rows partly undoes the reciprocity the stored G
// enforces — the two corrections pull against each other, and only a joint
// reciprocity-and-closure projection resolves that properly.

// Face-level Gebhart factors B = (I - F R)^-1 F E with R = diag(1 - eps),
// E = diag(eps). The dense solve limits this to small models (the guard
// throws above ~20k faces, pointing at gebhart_node_factors).
[[nodiscard]] SparseMatrix gebhart_factors(const SparseMatrix& vf,
                                           const Eigen::VectorXd& emissivity,
                                           std::span<const double> face_areas,
                                           double space_fraction_policy = 1.0);

// Node-level Gebhart GR matrix (m^2) for ANY model size:
// GR(m, n) = sum_{i in m} sum_{j in n} A_i eps_i B_ij, computed with one
// sparse factorization of (I - F R) and n_nodes sparse solves instead of a
// dense inverse. Rows/cols are indexed by position in
// aggregate_nodes(node_numbers); NO_NODE faces are dropped.
[[nodiscard]] SparseMatrix gebhart_node_factors(
    const SparseMatrix& vf, const Eigen::VectorXd& emissivity,
    std::span<const NodeNum> node_numbers, std::span<const double> face_areas,
    double space_fraction_policy = 1.0);

}  // namespace pycanha::radiative
