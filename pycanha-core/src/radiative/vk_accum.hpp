#pragma once

// src-private VF accumulator: a GPU counting buffer plus the host
// bookkeeping (rays emitted per row) needed to normalize counts into view
// factors and compute their statistics.

#include <cstdint>
#include <span>
#include <vector>

#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vk_scene.hpp"

namespace pycanha::radiative::detail {

class VfAccumImpl {
  public:
    // Dense layout only: one num_slots x num_slots u32 buffer, host-visible
    // so zeroing and readback are plain memory operations.
    // TODO(radiative): Tiled layout (streamed row blocks, CPU
    // sparsification) — currently rejected in the constructor.
    VfAccumImpl(SceneImpl& scene, AccumConfig config);
    ~VfAccumImpl();
    VfAccumImpl(const VfAccumImpl&) = delete;
    VfAccumImpl& operator=(const VfAccumImpl&) = delete;
    VfAccumImpl(VfAccumImpl&&) = delete;
    VfAccumImpl& operator=(VfAccumImpl&&) = delete;

    void reset();
    [[nodiscard]] VfResult build_result() const;

    // Bookkeeping done by SceneImpl::accumulate_vf after each batch.
    void record_batch(std::span<const std::uint32_t> emitters,
                      std::uint64_t rays_per_face);

    [[nodiscard]] VkBuffer buffer() const noexcept { return _counts.buffer; }
    [[nodiscard]] std::uint64_t rays_per_face() const noexcept {
        return _rays_per_face;
    }
    [[nodiscard]] SceneImpl& scene() const noexcept { return _scene; }

  private:
    SceneImpl& _scene;
    AccumConfig _config;
    GpuBuffer _counts;
    // Rays emitted per matrix row, accumulated across batches (rows in an
    // emitter subset differ from rows that never emitted).
    std::vector<std::uint64_t> _rays_per_row;
    std::uint64_t _rays_per_face = 0;
    std::uint64_t _total_rays = 0;
};

}  // namespace pycanha::radiative::detail
