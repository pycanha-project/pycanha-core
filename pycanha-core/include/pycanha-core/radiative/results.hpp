#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>

#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/sparse.hpp"

namespace pycanha::radiative {

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
    // VF only: max |Ai*Fij - Aj*Fji| (normalized).
    double reciprocity_residual = 0.0;
    // Exchange only: energy killed below the threshold.
    double lost_energy_fraction = 0.0;
    std::chrono::nanoseconds gpu_time{0};
};

// Matrix rows are face slots (global, both sides); columns are the same
// slots plus the virtual bucket columns above (cols == rows +
// num_virtual_columns). A vf row including its space column sums to
// exactly 1; the consumer maps the buckets to real nodes (e.g. the space
// node) or drops them.
struct VfResult {
    SparseF64 vf;
    // Per-row sum over ALL columns (== 1 exactly for rows that emitted).
    Eigen::VectorXd row_sums;
    TraceStats stats;
};

struct ExchangeResult {
    Band band = Band::IR;
    SparseF64 factors;
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
