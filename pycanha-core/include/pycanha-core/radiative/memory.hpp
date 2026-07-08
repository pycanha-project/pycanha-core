#pragma once

#include <cstdint>

namespace pycanha::radiative {

// Sizing mechanism for the accumulator memory policy: the C++ side reports
// exact byte requirements, the caller (the Python layer) decides layout and
// tile_rows against Device::memory_budget() — so runs fail fast with a clear
// message instead of exhausting device memory mid-trace.
// TODO(radiative): add estimate_memory(scene, config) once RadiativeScene
// exists.
struct MemoryEstimate {
    // Nf * cell_width per tile row (+ fixed overhead).
    std::uint64_t gpu_bytes_per_tile_row = 0;
    // BLAS/TLAS/materials/geometry.
    std::uint64_t gpu_bytes_scene = 0;
    // One readback block (2x when double-buffered).
    std::uint64_t host_bytes_block = 0;
};

}  // namespace pycanha::radiative
