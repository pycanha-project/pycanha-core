#pragma once

#include <cstdint>

namespace pycanha::radiative {

// Monte-Carlo trace control: a fixed ray count plus a seed make every run
// deterministic and reproducible. Accuracy-targeted convergence is
// intentionally NOT a C++ concept — the accumulate_* calls are
// batch-additive, so callers re-invoke until the reported statistics
// satisfy them.
struct TraceSettings {
    // Rays per emitting face, per accumulate() call.
    std::uint64_t rays_per_face = 10'000;
    // Deterministic seed; batches use seed + batch_index.
    std::uint32_t seed = 0;
    // MCRT ray-kill energy cutoff (exchange kernels).
    float energy_threshold = 1e-4F;
    // Hard safety bound on the bounce loop.
    std::uint32_t max_bounces = 64;
};

// Accumulator buffer layout. Kernels are identical in both layouts and cells
// accumulate as integers (integer adds are associative, unlike float adds),
// so Dense and Tiled results are bit-identical for the same seed.
enum class AccumLayout : std::uint8_t {
    Dense,  // one Nf x Nf buffer — the small-model fast path
    Tiled,  // tile_rows x Nf blocks streamed + CPU-sparsified — the big path
};

struct AccumConfig {
    AccumLayout layout = AccumLayout::Dense;
    // Tiled only; 0 = invalid, must be set (the Python policy derives it).
    std::uint32_t tile_rows = 0;
    // Drop entries <= threshold when sparsifying a readback block; row sums
    // and space deficits are computed BEFORE thresholding, so conservation
    // checks stay exact. 0 keeps any nonzero.
    double sparse_threshold = 0.0;
};

}  // namespace pycanha::radiative
