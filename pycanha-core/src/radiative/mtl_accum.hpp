#pragma once

// src-private accumulators for the Metal backend: GPU integer cells plus the
// host bookkeeping needed to normalize them into results and compute
// statistics. Mirrors vk_accum.hpp one-to-one — same class and method names,
// so the shared scene.cpp compiles against either backend. Matrix
// accumulators (VF counts, exchange fixed-point energy) come in two layouts,
// one result:
//  - Dense: one num_slots x num_slots buffer, cells stay on the GPU.
//  - Tiled: a tile_rows x num_slots scratch buffer reused per row block;
//    cells stream into per-row host maps after every block.
// Cells are integers, so both layouts produce bit-identical results for the
// same seed — a property the tests assert. The solar accumulator is two
// per-face vectors and needs no layout machinery.
//
// The 64-bit cells are written by the kernels as a pair of 32-bit atomic
// counters (Metal has no 64-bit buffer atomics), but two adjacent u32 at an
// 8-byte stride ARE the little-endian image of the u64, so every size and
// every readback below is the same as on the Vulkan side.

#import <Metal/Metal.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "mtl_scene.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vf_assemble.hpp"

namespace pycanha::radiative::detail {

// Running per-entry statistics while scanning exchange rows into a CSR.
struct EntryStats {
    double stderr_sum = 0.0;
    double stderr_max = 0.0;
    std::size_t entries = 0;
    double lost_energy = 0.0;  // signed: Russian-roulette adjustments
};

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
    [[nodiscard]] id<MTLBuffer> buffer() const noexcept {
        return _counts.buffer;
    }
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
    SceneImpl& _scene;
    AccumConfig _config;
    GpuBuffer _counts;
    // Tiled layout: accumulated counts per row (column -> count). Dense
    // keeps everything in the GPU buffer instead.
    std::vector<HostCountRow> _host_rows;
    // Rays emitted per matrix row, accumulated across batches (rows in an
    // emitter subset differ from rows that never emitted).
    std::vector<std::uint64_t> _rays_per_row;
    std::uint64_t _rays_per_face = 0;
    std::uint64_t _total_rays = 0;
};

class ExchangeAccumImpl {
  public:
    ExchangeAccumImpl(SceneImpl& scene, Band band, AccumConfig config);
    ~ExchangeAccumImpl();
    ExchangeAccumImpl(const ExchangeAccumImpl&) = delete;
    ExchangeAccumImpl& operator=(const ExchangeAccumImpl&) = delete;
    ExchangeAccumImpl(ExchangeAccumImpl&&) = delete;
    ExchangeAccumImpl& operator=(ExchangeAccumImpl&&) = delete;

    void reset();
    [[nodiscard]] ExchangeResult build_result() const;
    [[nodiscard]] std::uint64_t conservation_error() const;

    // --- called by SceneImpl::accumulate_exchange ---------------------------
    [[nodiscard]] Band band() const noexcept { return _band; }
    [[nodiscard]] AccumLayout layout() const noexcept { return _config.layout; }
    [[nodiscard]] std::uint32_t tile_rows() const noexcept {
        return _config.tile_rows;
    }
    [[nodiscard]] id<MTLBuffer> buffer() const noexcept {
        return _cells.buffer;
    }
    // Fixes the fixed-point scale on the first batch (a power of two with
    // headroom for follow-up batches) and guards the cumulative ray budget
    // against cell overflow. Returns the scale for the push constants.
    [[nodiscard]] float prepare_batch(std::uint64_t rays_per_face);
    // Tiled only: zeroes the block scratch before a new row block.
    void clear_block_scratch();
    // Tiled only: adds the scratch rows of `block_emitters` into the host
    // maps (block-relative row = slot - row_offset).
    void absorb_block(std::span<const std::uint32_t> block_emitters,
                      std::uint32_t row_offset);
    void record_batch(std::span<const std::uint32_t> emitters,
                      std::uint64_t rays_per_face);
    [[nodiscard]] SceneImpl& scene() const noexcept { return _scene; }

  private:
    // Fixed-point value of cell (row, col) regardless of layout; columns
    // include the virtual space/inactive/lost buckets.
    [[nodiscard]] std::uint64_t cell_at(std::size_t row, std::size_t col) const;
    // Appends one row's thresholded entries and statistics (including the
    // signed lost-column energy).
    void scan_row(std::size_t row, std::uint64_t rays_row,
                  std::vector<SparseIndex>& indices,
                  std::vector<double>& values, EntryStats& stats) const;

    SceneImpl& _scene;
    Band _band;
    AccumConfig _config;
    GpuBuffer _cells;  // u64 fixed-point deposits, row stride slots + 3
    // Tiled layout: host-side accumulation (Dense reads the GPU buffer).
    std::vector<HostCountRow> _host_rows;
    std::vector<std::uint64_t> _rays_per_row;
    // Power of two chosen on the first batch; 0 = not chosen yet. Kept as
    // double for exact integer arithmetic on the host (f32 in the shader).
    double _fp_scale = 0.0;
    std::uint64_t _rays_per_face = 0;
    std::uint64_t _total_rays = 0;
};

class SolarAccumImpl {
  public:
    explicit SolarAccumImpl(SceneImpl& scene);
    ~SolarAccumImpl();
    SolarAccumImpl(const SolarAccumImpl&) = delete;
    SolarAccumImpl& operator=(const SolarAccumImpl&) = delete;
    SolarAccumImpl(SolarAccumImpl&&) = delete;
    SolarAccumImpl& operator=(SolarAccumImpl&&) = delete;

    void reset();
    [[nodiscard]] SolarResult build_result() const;

    // --- called by SceneImpl::accumulate_solar ------------------------------
    [[nodiscard]] id<MTLBuffer> direct_buffer() const noexcept {
        return _direct.buffer;
    }
    [[nodiscard]] id<MTLBuffer> total_buffer() const noexcept {
        return _total.buffer;
    }
    // Records the sun on the first batch (later batches must match — summing
    // different suns is meaningless), fixes the fixed-point scale and guards
    // the cumulative ray budget. Returns the normalized sun direction and
    // the scale for the push constants.
    struct BatchSetup {
        std::array<float, 3> sun_dir;
        float fp_scale;
    };
    [[nodiscard]] BatchSetup prepare_batch(const SolarState& sun,
                                           std::uint64_t rays_per_face);
    void record_batch(std::uint64_t rays_per_face, std::size_t num_emitters);
    [[nodiscard]] SceneImpl& scene() const noexcept { return _scene; }

  private:
    SceneImpl& _scene;
    GpuBuffer _direct;  // u64 fixed-point absorbed direct energy per slot
    GpuBuffer _total;   // u64 fixed-point absorbed total energy per slot
    SolarState _sun{};
    bool _sun_recorded = false;
    double _fp_scale = 0.0;
    // Deposits carry the emitting face's area, so the fixed-point budget
    // scales with the summed face area (rounded up, at least 1).
    std::uint64_t _area_units = 1;
    std::uint64_t _rays_per_face = 0;
    std::uint64_t _total_rays = 0;
};

}  // namespace pycanha::radiative::detail
