#pragma once

#include <cstdint>

namespace pycanha::radiative {

// Sizing mechanism for the accumulator memory policy (D29): the C++ side
// reports exact byte requirements, the Python side decides layout/tile_rows
// against Device::memory_budget(). estimate_memory(scene, config) lands with
// RadiativeScene.
struct MemoryEstimate {
    // Nf * cell_width per tile row (+ fixed overhead).
    std::uint64_t gpu_bytes_per_tile_row = 0;
    // BLAS/TLAS/materials/geometry.
    std::uint64_t gpu_bytes_scene = 0;
    // One readback block (2x when double-buffered).
    std::uint64_t host_bytes_block = 0;
};

}  // namespace pycanha::radiative
