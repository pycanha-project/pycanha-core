#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <cstdint>
#include <optional>

#include "pycanha-core/radiative/materials.hpp"

namespace pycanha::radiative {

// Every sparse radiative result is a row-major Eigen sparse matrix: the same
// type the TMM coupling matrices use, and the one nanobind converts straight
// to scipy.sparse.csr_matrix without a hand-written binding.
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
// Column/row-pointer type of the CSR arrays above (Eigen's default `int`).
using SparseIndex = SparseMatrix::StorageIndex;

// Matrix results carry three VIRTUAL bucket columns appended after the
// num_face_slots real columns, so every parcel of emitted energy has an
// explicit, node-mappable destination:
//  - space:    rays/energy that escaped the scene,
//  - inactive: energy absorbed at inactive (non-radiative) faces
//              (exchange only — the geometric vf kernel scores inactive
//              faces at their own column),
//  - lost:     the purely mathematical residue (Russian-roulette balance,
//              max_bounces cutoff). Zero-mean adjustments make this entry
//              SIGNED and it may come out slightly negative.
// Full rows therefore account for everything: a vf row sums to exactly 1
// and an exchange row conserves energy exactly.
inline constexpr std::int64_t num_virtual_columns = 3;
inline constexpr std::int64_t space_column_offset = 0;
inline constexpr std::int64_t inactive_column_offset = 1;
inline constexpr std::int64_t lost_column_offset = 2;

// Every result carries its own statistics so callers can judge Monte-Carlo
// convergence without re-tracing.
struct TraceStats {
    std::uint64_t total_rays = 0;
    // Cumulative across accumulate() calls.
    std::uint64_t rays_per_face = 0;
    // Per-entry standard-error estimates.
    double mean_stderr = 0.0;
    double max_stderr = 0.0;
    // VF only: max |Ai*Fij - Aj*Fji| (normalized), measured on the RAW
    // estimates before they are combined. Measured after combining it would
    // be identically zero by construction and would stop being the
    // winding/parity check it exists to be.
    double reciprocity_residual = 0.0;
    // Exchange only: energy killed below the threshold.
    double lost_energy_fraction = 0.0;
    std::chrono::nanoseconds gpu_time{0};
};

// Matrix rows are face slots (global, both sides); columns are the same
// slots plus the virtual bucket columns above (cols == rows +
// num_virtual_columns). The consumer maps the buckets to real nodes (e.g.
// the space node) or drops them.
//
// The stored value is the SYMMETRIC, EXTENSIVE quantity
//
//     G_ij = A_i F_ij = A_j F_ji     (units m^2)
//
// and only the UPPER TRIANGLE (j >= i) of the real face columns is kept,
// matching the TMM coupling convention. Storing G rather than F is what
// makes that lossless: both view factors come back as F_ij = G_ij/A_i and
// F_ji = G_ij/A_j from the face areas the scene already owns, whereas an
// upper-triangular F would silently discard one direction. The diagonal is
// retained even though a planar face slot cannot see itself, so nothing is
// dropped without saying so. The three bucket columns are always > i, have
// no transpose partner, and pass through untriangulated.
struct VfResult {
    SparseMatrix vf;
    // Per-row sum over ALL columns of the RAW estimate, before combining
    // and before thresholding (== 1 exactly for rows that emitted). This is
    // closure, and it is deliberately not the row sum of `vf`: after
    // triangulation the stored triangle's row sums mean nothing of the
    // kind.
    Eigen::VectorXd row_sums;
    // The raw, untriangulated G with BOTH triangles, present only when
    // TriangulationConfig::keep_full_matrix asked for it. Entry (i, j)
    // holds A_i F_ij and entry (j, i) holds A_j F_ji, so the two
    // independent estimates of the same coupling can be read off and
    // compared directly.
    std::optional<SparseMatrix> full_vf;
    TraceStats stats;
};

struct ExchangeResult {
    Band band = Band::IR;
    SparseMatrix factors;
    TraceStats stats;
};

// Per-face-slot absorbed power; the solar kernel is O(Nf) and needs no
// matrix. Watts (extensive): node mapping is a plain per-node sum, and
// flux is watts / face area when needed.
struct SolarResult {
    Eigen::VectorXd direct;  // W absorbed, direct illumination
    Eigen::VectorXd total;   // W absorbed incl. reflections
    TraceStats stats;
};

}  // namespace pycanha::radiative
