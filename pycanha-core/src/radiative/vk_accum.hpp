#pragma once

// src-private VF accumulator: GPU counting cells plus the host bookkeeping
// (rays emitted per row) needed to normalize counts into view factors and
// compute their statistics. Two layouts, one result:
//  - Dense: one num_slots x num_slots buffer, counts stay on the GPU.
//  - Tiled: a tile_rows x num_slots scratch buffer reused per row block;
//    counts stream into per-row host maps after every block.
// Cells are integers, so both layouts produce bit-identical results for the
// same seed — a property the tests assert.

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vk_scene.hpp"

namespace pycanha::radiative::detail {

class VfAccumImpl {
  public:
    VfAccumImpl(SceneImpl& scene, AccumConfig config);
    ~VfAccumImpl();
    VfAccumImpl(const VfAccumImpl&) = delete;
    VfAccumImpl& operator=(const VfAccumImpl&) = delete;
    VfAccumImpl(VfAccumImpl&&) = delete;
    VfAccumImpl& operator=(VfAccumImpl&&) = delete;

    void reset();
    [[nodiscard]] VfResult build_result() const;

    // --- called by SceneImpl::accumulate_vf --------------------------------
    [[nodiscard]] AccumLayout layout() const noexcept { return _config.layout; }
    [[nodiscard]] std::uint32_t tile_rows() const noexcept {
        return _config.tile_rows;
    }
    // The GPU cell buffer: whole matrix (Dense) or block scratch (Tiled).
    [[nodiscard]] VkBuffer buffer() const noexcept { return _counts.buffer; }
    // Tiled only: zeroes the block scratch before a new row block.
    void clear_block_scratch();
    // Tiled only: reads the scratch rows of `block_emitters` (block-relative
    // row = slot - row_offset) and adds nonzero cells into the host maps.
    void absorb_block(std::span<const std::uint32_t> block_emitters,
                      std::uint32_t row_offset);
    void record_batch(std::span<const std::uint32_t> emitters,
                      std::uint64_t rays_per_face);
    [[nodiscard]] std::uint64_t rays_per_face() const noexcept {
        return _rays_per_face;
    }
    [[nodiscard]] SceneImpl& scene() const noexcept { return _scene; }

  private:
    // Integer count of cell (row, col) regardless of layout.
    [[nodiscard]] std::uint64_t count_at(std::size_t row,
                                         std::size_t col) const;

    SceneImpl& _scene;
    AccumConfig _config;
    GpuBuffer _counts;
    // Tiled layout: accumulated counts per row (column -> count). Dense
    // keeps everything in the GPU buffer instead.
    std::vector<std::unordered_map<std::uint32_t, std::uint64_t>> _host_rows;
    // Rays emitted per matrix row, accumulated across batches (rows in an
    // emitter subset differ from rows that never emitted).
    std::vector<std::uint64_t> _rays_per_row;
    std::uint64_t _rays_per_face = 0;
    std::uint64_t _total_rays = 0;
};

}  // namespace pycanha::radiative::detail
