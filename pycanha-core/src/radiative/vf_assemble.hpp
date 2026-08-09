#pragma once

// Backend-agnostic VF assembly. Turning accumulated integer counts into a
// VfResult is pure host arithmetic — no Vulkan, no Objective-C — so both
// backends call the same implementation instead of mirroring it.
//
// That matters more than ordinary de-duplication: the Metal backend cannot
// be compiled on a machine without a Mac, so a mistake mirrored into the
// .mm file produces no error, no warning and no failing test until someone
// with Apple hardware next builds the branch. Keeping the logic here means
// the only Metal-side code is a call.
//
// The traversal itself — the cache-blocked tile-pair walk, the sparse merge,
// the weight table, the CSR packing — is shared with the exchange assembly
// and lives in pair_walk.hpp. What stays here is only what knows that a cell
// is a hit count.

#include <cstdint>
#include <span>
#include <variant>

#include "pair_walk.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative::detail {

// Where the accumulated counts live, which is the only thing that differs
// between the two layouts:
//  - Dense: the mapped GPU cell block, rows * (rows + num_virtual_columns)
//    u32 cells in row-major order. The caller has already made it readable
//    on the host (Vulkan needs an explicit invalidate; Metal's unified
//    memory does not).
//  - Tiled: one host map per row. The GPU buffer only ever held one block
//    of scratch, which was drained into these maps after every dispatch.
using CountSource =
    std::variant<std::span<const std::uint32_t>, std::span<const HostCountRow>>;

// `areas` and `rays_per_row` are per face slot; the result matrix has one
// row per slot and num_virtual_columns extra bucket columns. Ray counts and
// trace timings that belong to the accumulator rather than the counts
// (total_rays, rays_per_face, gpu_time) are filled in by the caller.
[[nodiscard]] VfResult assemble_vf(CountSource counts,
                                   std::span<const double> areas,
                                   std::span<const std::uint64_t> rays_per_row,
                                   const AccumConfig& config,
                                   const AssemblyTuning& tuning = {});

}  // namespace pycanha::radiative::detail
