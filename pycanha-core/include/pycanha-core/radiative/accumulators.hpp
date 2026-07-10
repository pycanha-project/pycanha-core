#pragma once

#include <memory>

#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative {

class RadiativeScene;

namespace detail {
class VfAccumImpl;
}  // namespace detail

// Owns the GPU-side counting buffer plus the bookkeeping needed to turn raw
// counts into a VfResult. Batch-additive: successive accumulate_vf calls add
// up; reset() clears everything. Cells are integers, so results are
// bit-deterministic for a given seed regardless of dispatch shape.
class VfAccumulator {
  public:
    // Dense allocates the full num_slots x num_slots buffer up front (the
    // small-model fast path); Tiled bounds GPU memory to tile_rows x
    // num_slots and streams row blocks into host-side sparse storage. Both
    // produce bit-identical results for the same seed.
    explicit VfAccumulator(const RadiativeScene& scene,
                           AccumConfig config = {});
    ~VfAccumulator();
    VfAccumulator(VfAccumulator&&) noexcept;
    VfAccumulator& operator=(VfAccumulator&&) noexcept;
    VfAccumulator(const VfAccumulator&) = delete;
    VfAccumulator& operator=(const VfAccumulator&) = delete;

    void reset();
    // Readback + normalization + statistics. Callable repeatedly; each call
    // reflects everything accumulated so far.
    [[nodiscard]] VfResult result() const;

    [[nodiscard]] detail::VfAccumImpl& impl() noexcept;

  private:
    std::unique_ptr<detail::VfAccumImpl> _impl;
};

}  // namespace pycanha::radiative
