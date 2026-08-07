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

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <variant>
#include <vector>

#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative::detail {

// Tiled-layout row storage: accumulated column -> count for one matrix row.
using HostCountRow = std::unordered_map<std::uint32_t, std::uint64_t>;

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

// Weight of the forward estimate when the two Monte-Carlo estimates of a
// face pair are combined:
//
//     X_i = 1/2 (1 + sign(Y) |Y|^n),   Y = (v - u) / (v + u)
//
// with u = A_i/N_i and v = A_j/N_j. |Y|^n is tabulated rather than handed to
// std::pow because the inner loop runs once per matrix cell: at 1e4 face
// slots a pow per pair costs roughly ten times the memory traffic of the
// entire pass, which would turn a streaming, bandwidth-bound assembly into a
// compute-bound one. It also removes the only transcendental from an
// otherwise IEEE-exact pipeline, which is what keeps results reproducible
// between toolchains.
//
// The tabulation splits t = m * 2^e with m in [1, 2), so t^n = m^n * 2^(n e),
// and tabulates the two factors separately. A uniform grid in t would be bad
// exactly where it matters: for n < 1 the derivative of t^n is unbounded as
// t -> 0, so the small view factors would carry large relative error. m^n
// over [1, 2) is smooth with bounded derivative, so linear interpolation on
// it is accurate to about 1e-8 at 4096 nodes.
//
// Every entry is rounded to 24 significand bits. Two platforms' std::pow
// agree to within an ulp or so, far finer than that quantum, so the rounded
// tables come out identical and the assembled values then match bit for bit;
// the 6e-8 relative error the rounding costs sits orders of magnitude below
// the Monte-Carlo noise the weight is applied to.
class WeightTable {
  public:
    // `exponent` must be finite and > 0. n = 0 would make the weight
    // winner-takes-all AND break the equal-density case, where 0^n must
    // vanish so that both estimates get exactly one half.
    explicit WeightTable(double exponent);

    // u and v are the two A/N ratios; both must be finite and >= 0.
    [[nodiscard]] double forward_weight(double u, double v) const noexcept;

    // Same weight computed straight from std::pow. Kept as the oracle the
    // tabulated path is tested against, and as the fallback the tuning knob
    // below can select.
    [[nodiscard]] static double naive_forward_weight(double u, double v,
                                                     double exponent) noexcept;

  private:
    [[nodiscard]] double unit_pow(double t) const noexcept;

    double _exponent;
    // m^n at 4096 + 1 nodes across m in [1, 2].
    std::vector<double> _mantissa;
    // 2^(n e) for every binary exponent a double in (0, 1) can carry.
    std::vector<double> _scale;
};

// Assembly knobs that are not part of the public API. They exist so a test
// can pin the fast path against the simple one: the blocked walk, the
// thread split and the tabulated weight are all written for speed, and each
// must produce exactly what its obvious counterpart produces.
struct AssemblyTuning {
    // Side of the square tile pairs the transposed walk holds in cache.
    // 0 selects the plain row-by-row walk instead (Dense only; the sparse
    // path is a linear merge in both cases).
    std::size_t tile = 128;
    // 0 asks for hardware_concurrency. Small matrices always run inline.
    unsigned threads = 0;
    // false selects WeightTable::naive_forward_weight.
    bool tabulated_weight = true;
};

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
