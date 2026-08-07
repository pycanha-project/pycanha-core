#pragma once

// Backend-agnostic exchange assembly. Turning accumulated fixed-point
// deposits into an ExchangeResult is pure host arithmetic — no Vulkan, no
// Objective-C — so both backends call the same implementation instead of
// mirroring it.
//
// That matters more than ordinary de-duplication: the Metal backend cannot
// be compiled on a machine without a Mac, so a mistake mirrored into the
// .mm file produces no error, no warning and no failing test until someone
// with Apple hardware next builds the branch. Keeping the logic here means
// the only Metal-side code is a call.

#include <cstdint>
#include <span>
#include <variant>

#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vf_assemble.hpp"

namespace pycanha::radiative::detail {

// Where the accumulated deposits live, which is the only thing that differs
// between the two layouts:
//  - Dense: the mapped GPU cell block, rows * (rows + num_virtual_columns)
//    u64 cells in row-major order. The caller has already made it readable
//    on the host (Vulkan needs an explicit invalidate; Metal's unified
//    memory does not).
//  - Tiled: one host map per row. The GPU buffer only ever held one block
//    of scratch, which was drained into these maps after every dispatch.
//
// Cells hold energy in fixed point: the integer count of 1/fp_scale units
// deposited. Integer adds are associative, so the two layouts accumulate to
// the same cells and every number derived from them is bit-identical.
using ExchangeCellSource =
    std::variant<std::span<const std::uint64_t>, std::span<const HostCountRow>>;

// `rays_per_row` is per face slot and defines the slot count; the result
// matrix has one row per slot and num_virtual_columns extra bucket columns.
// `fp_scale` is the power of two the accumulator fixed on its first batch (0
// when nothing was ever traced, which yields an empty result). Ray counts
// and trace timings that belong to the accumulator rather than the cells
// (total_rays, rays_per_face, gpu_time) are filled in by the caller.
[[nodiscard]] ExchangeResult assemble_exchange(
    ExchangeCellSource cells, std::span<const std::uint64_t> rays_per_row,
    double fp_scale, Band band, const AccumConfig& config);

// Largest deviation of a row's total deposits from the energy that row
// emitted, in fixed-point units. Everything wraps mod 2^64 — the identity
// the kernels maintain — so a correct trace returns exactly 0.
[[nodiscard]] std::uint64_t exchange_conservation_error(
    ExchangeCellSource cells, std::span<const std::uint64_t> rays_per_row,
    double fp_scale);

}  // namespace pycanha::radiative::detail
