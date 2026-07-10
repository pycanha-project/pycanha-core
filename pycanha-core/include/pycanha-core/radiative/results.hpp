#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>

#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/sparse.hpp"

namespace pycanha::radiative {

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

// Matrix rows/cols are face slots (global, both sides). 1 - row_sum of a VF
// row is the view factor to space; exchange matrices likewise leave "to
// space" implicit as the row deficit (the consumer materializes it as
// couplings to a space node).
struct VfResult {
    SparseF64 vf;
    Eigen::VectorXd row_sums;
    TraceStats stats;
};

struct ExchangeResult {
    Band band = Band::IR;
    SparseF64 factors;
    TraceStats stats;
};

// Per-face-slot vectors; the solar kernel is O(Nf) and needs no matrix.
struct SolarResult {
    Eigen::VectorXd direct;  // W/m^2 absorbed, direct illumination
    Eigen::VectorXd total;   // W/m^2 absorbed incl. reflections
    TraceStats stats;
};

}  // namespace pycanha::radiative
