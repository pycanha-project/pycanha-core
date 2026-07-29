#pragma once

#include <Eigen/Dense>
#include <span>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/radiative/sparse.hpp"

namespace pycanha::radiative {

// CPU Gebhart services (no Vulkan, built unconditionally): re-derive
// radiative exchange factors from a geometric VF matrix and per-face-slot
// band emissivities without re-tracing — the diffuse-gray fast path. The
// MCRT exchange kernel computes the same quantity directly, so the two
// paths are cross-checkable within Monte-Carlo tolerance.

// Face-level Gebhart factors B = (I - F R)^-1 F E with R = diag(1 - eps),
// E = diag(eps). The dense solve limits this to small models (the guard
// throws above ~20k face slots, pointing at gebhart_node_factors).
//
// `space_fraction_policy` decides what each row's deficit (1 - row_sum)
// means: 1.0 (default) treats it as a real view to space; 0.0 renormalizes
// the row to sum to one (closed-enclosure assumption, i.e. the deficit is
// Monte-Carlo noise); values in between scale the deficit accordingly.
[[nodiscard]] SparseF64 gebhart_factors(const SparseF64& vf,
                                        const Eigen::VectorXd& emissivity,
                                        double space_fraction_policy = 1.0);

// Node-level Gebhart GR matrix (m^2) for ANY model size:
// GR(m, n) = sum_{i in m} sum_{j in n} A_i eps_i B_ij, computed with one
// sparse factorization of (I - F R) and n_nodes sparse solves instead of a
// dense inverse. Rows/cols are indexed by position in
// aggregate_nodes(node_numbers); NO_NODE slots are dropped.
[[nodiscard]] SparseF64 gebhart_node_factors(
    const SparseF64& vf, const Eigen::VectorXd& emissivity,
    std::span<const NodeNum> node_numbers, std::span<const double> face_areas,
    double space_fraction_policy = 1.0);

}  // namespace pycanha::radiative
