#include "vk_accum.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vk_device.hpp"
#include "vk_scene.hpp"

namespace pycanha::radiative::detail {

namespace {

// Largest power of two S with rays_per_face * S <= 2^58. Powers of two are
// exact in f32 and make to_fp(1.0) == S on the GPU, so a row's expected
// balance is the exact integer rays * S; the 2^58 budget leaves 32x
// cumulative-ray headroom before a u64 row balance could overflow, with a
// deposit resolution (1/S) still far below Monte-Carlo noise.
[[nodiscard]] double select_fp_scale(std::uint64_t rays_per_face) {
    constexpr int budget_bits = 58;
    const int rays_bits =
        rays_per_face > 1 ? std::bit_width(rays_per_face - 1) : 0;
    return std::ldexp(1.0, std::max(budget_bits - rays_bits, 0));
}

// Cumulative rays_per_face a fixed-point accumulator can absorb before its
// u64 balances could overflow.
[[nodiscard]] std::uint64_t max_cumulative_rays(double fp_scale) {
    return (std::uint64_t{1} << 63U) / static_cast<std::uint64_t>(fp_scale);
}

// Magnitude of a wrapping-u64 difference (conservation residuals are exact
// zeros when the kernel is right; a broken kernel may miss in either
// direction).
[[nodiscard]] std::uint64_t wrap_magnitude(std::uint64_t difference) {
    return std::min(difference, std::uint64_t{0} - difference);
}

void clear_host_visible(const SceneImpl& scene, const GpuBuffer& buffer) {
    std::memset(checked_mapped(buffer), 0, buffer.size);
    vmaFlushAllocation(scene.device().allocator, buffer.allocation, 0,
                       VK_WHOLE_SIZE);
}

void invalidate_host_visible(const SceneImpl& scene, const GpuBuffer& buffer) {
    vmaInvalidateAllocation(scene.device().allocator, buffer.allocation, 0,
                            VK_WHOLE_SIZE);
}

// Matrix row stride: the real face columns plus the virtual
// space/inactive/lost bucket columns.
[[nodiscard]] std::size_t matrix_columns(std::size_t slots) {
    return slots + static_cast<std::size_t>(num_virtual_columns);
}

}  // namespace

VfAccumImpl::VfAccumImpl(SceneImpl& scene, AccumConfig config)
    : _scene(scene), _config(config) {
    const std::uint64_t slots = _scene.num_face_slots();
    std::uint64_t buffer_rows = slots;
    if (_config.layout == AccumLayout::Tiled) {
        if (_config.tile_rows == 0) {
            throw std::invalid_argument(
                "pycanha::radiative: the Tiled layout needs tile_rows > 0");
        }
        _config.tile_rows = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(_config.tile_rows, slots));
        buffer_rows = _config.tile_rows;
        _host_rows.resize(slots);
    }
    _counts = _scene.create_buffer(
        buffer_rows * matrix_columns(slots) * sizeof(std::uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        /*host_visible=*/true);
    _rays_per_row.assign(slots, 0);
    reset();
}

VfAccumImpl::~VfAccumImpl() { _scene.destroy_buffer(_counts); }

void VfAccumImpl::reset() {
    clear_block_scratch();
    for (auto& row : _host_rows) {
        row.clear();
    }
    std::ranges::fill(_rays_per_row, 0);
    _rays_per_face = 0;
    _total_rays = 0;
}

void VfAccumImpl::clear_block_scratch() { clear_host_visible(_scene, _counts); }

void VfAccumImpl::absorb_block(std::span<const std::uint32_t> block_emitters,
                               std::uint32_t row_offset) {
    invalidate_host_visible(_scene, _counts);
    const std::size_t cols = matrix_columns(_scene.num_face_slots());
    const std::span<const std::uint32_t> scratch(
        static_cast<const std::uint32_t*>(checked_mapped(_counts)),
        static_cast<std::size_t>(_config.tile_rows) * cols);
    for (const std::uint32_t slot : block_emitters) {
        const std::size_t row = slot - row_offset;
        auto& host_row = _host_rows[slot];
        for (std::size_t col = 0; col < cols; ++col) {
            const std::uint32_t cell = scratch[(row * cols) + col];
            if (cell != 0) {
                host_row[static_cast<std::uint32_t>(col)] += cell;
            }
        }
    }
}

void VfAccumImpl::record_batch(std::span<const std::uint32_t> emitters,
                               std::uint64_t rays_per_face) {
    for (const std::uint32_t slot : emitters) {
        _rays_per_row[slot] += rays_per_face;
    }
    _rays_per_face += rays_per_face;
    _total_rays += rays_per_face * emitters.size();
}

std::uint64_t VfAccumImpl::count_at(std::size_t row, std::size_t col) const {
    if (_config.layout == AccumLayout::Dense) {
        const std::size_t slots = _scene.num_face_slots();
        const std::size_t cols = matrix_columns(slots);
        const std::span<const std::uint32_t> counts(
            static_cast<const std::uint32_t*>(checked_mapped(_counts)),
            slots * cols);
        return counts[(row * cols) + col];
    }
    const auto& host_row = _host_rows[row];
    const auto it = host_row.find(static_cast<std::uint32_t>(col));
    return it == host_row.end() ? 0 : it->second;
}

double VfAccumImpl::scan_row(std::size_t row, std::uint64_t rays_row,
                             std::vector<std::int32_t>& indices,
                             std::vector<double>& values,
                             EntryStats& stats) const {
    const std::size_t slots = _scene.num_face_slots();
    const std::size_t cols = matrix_columns(slots);
    double row_sum = 0.0;
    // Ascending column order in both layouts, so the CSR (and every derived
    // statistic) is bit-identical between Dense and Tiled.
    for (std::size_t col = 0; col < cols; ++col) {
        const std::uint64_t count = count_at(row, col);
        if (count == 0) {
            continue;
        }
        const double vf =
            static_cast<double>(count) / static_cast<double>(rays_row);
        // Row statistics come BEFORE thresholding, so dropping tiny entries
        // never corrupts the closure accounting.
        row_sum += vf;
        if (col < slots) {
            // Binomial standard error of the per-entry estimate (real face
            // columns only).
            const double entry_stderr = std::sqrt(
                vf * std::max(1.0 - vf, 0.0) / static_cast<double>(rays_row));
            stats.stderr_sum += entry_stderr;
            stats.stderr_max = std::max(stats.stderr_max, entry_stderr);
            ++stats.entries;
        }
        if (vf > _config.sparse_threshold) {
            indices.push_back(static_cast<std::int32_t>(col));
            values.push_back(vf);
        }
    }
    return row_sum;
}

VfResult VfAccumImpl::build_result() const {
    if (_config.layout == AccumLayout::Dense) {
        invalidate_host_visible(_scene, _counts);
    }
    const std::size_t slots = _scene.num_face_slots();
    const std::span<const double> areas = _scene.face_areas();

    VfResult result;
    result.vf.rows = static_cast<std::int64_t>(slots);
    result.vf.cols = static_cast<std::int64_t>(matrix_columns(slots));
    result.vf.indptr.resize(static_cast<Eigen::Index>(slots) + 1);
    result.row_sums = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(slots));

    std::vector<std::int32_t> indices;
    std::vector<double> values;
    EntryStats stats;

    result.vf.indptr(0) = 0;
    for (std::size_t row = 0; row < slots; ++row) {
        const std::uint64_t rays_row = _rays_per_row[row];
        if (rays_row > 0) {
            result.row_sums(static_cast<Eigen::Index>(row)) =
                scan_row(row, rays_row, indices, values, stats);
        }
        result.vf.indptr(static_cast<Eigen::Index>(row) + 1) =
            static_cast<std::int64_t>(values.size());
    }

    result.vf.indices = Eigen::Map<const Eigen::VectorX<std::int32_t>>(
        indices.data(), static_cast<Eigen::Index>(indices.size()));
    result.vf.values = Eigen::Map<const Eigen::VectorXd>(
        values.data(), static_cast<Eigen::Index>(values.size()));

    // Reciprocity residual max |Ai*Fij - Aj*Fji| (normalized by the larger
    // term) over pairs where both rows emitted — a winding/parity bug shows
    // up here long before it is visible in individual entries.
    double reciprocity = 0.0;
    for (std::size_t row = 0; row < slots; ++row) {
        if (_rays_per_row[row] == 0) {
            continue;
        }
        for (std::size_t col = row + 1; col < slots; ++col) {
            if (_rays_per_row[col] == 0) {
                continue;
            }
            const double forward = areas[row] *
                                   static_cast<double>(count_at(row, col)) /
                                   static_cast<double>(_rays_per_row[row]);
            // Transposed lookup: the reverse-direction view factor.
            const std::size_t transposed_row = col;
            const std::size_t transposed_col = row;
            const double backward =
                areas[col] *
                static_cast<double>(count_at(transposed_row, transposed_col)) /
                static_cast<double>(_rays_per_row[col]);
            const double larger = std::max(forward, backward);
            if (larger > 0.0) {
                reciprocity = std::max(reciprocity,
                                       std::abs(forward - backward) / larger);
            }
        }
    }

    result.stats.total_rays = _total_rays;
    result.stats.rays_per_face = _rays_per_face;
    result.stats.mean_stderr =
        stats.entries > 0
            ? stats.stderr_sum / static_cast<double>(stats.entries)
            : 0.0;
    result.stats.max_stderr = stats.stderr_max;
    result.stats.reciprocity_residual = reciprocity;
    // TODO(radiative): fill gpu_time from timestamp queries around each
    // dispatch chunk.
    return result;
}

ExchangeAccumImpl::ExchangeAccumImpl(SceneImpl& scene, Band band,
                                     AccumConfig config)
    : _scene(scene), _band(band), _config(config) {
    const std::uint64_t slots = _scene.num_face_slots();
    std::uint64_t buffer_rows = slots;
    if (_config.layout == AccumLayout::Tiled) {
        if (_config.tile_rows == 0) {
            throw std::invalid_argument(
                "pycanha::radiative: the Tiled layout needs tile_rows > 0");
        }
        _config.tile_rows = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(_config.tile_rows, slots));
        buffer_rows = _config.tile_rows;
        _host_rows.resize(slots);
    }
    _cells = _scene.create_buffer(
        buffer_rows * matrix_columns(slots) * sizeof(std::uint64_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        /*host_visible=*/true);
    _rays_per_row.assign(slots, 0);
    reset();
}

ExchangeAccumImpl::~ExchangeAccumImpl() { _scene.destroy_buffer(_cells); }

void ExchangeAccumImpl::reset() {
    clear_block_scratch();
    for (auto& row : _host_rows) {
        row.clear();
    }
    std::ranges::fill(_rays_per_row, 0);
    _fp_scale = 0.0;
    _rays_per_face = 0;
    _total_rays = 0;
}

void ExchangeAccumImpl::clear_block_scratch() {
    clear_host_visible(_scene, _cells);
}

void ExchangeAccumImpl::absorb_block(
    std::span<const std::uint32_t> block_emitters, std::uint32_t row_offset) {
    invalidate_host_visible(_scene, _cells);
    const std::size_t cols = matrix_columns(_scene.num_face_slots());
    const std::span<const std::uint64_t> scratch(
        static_cast<const std::uint64_t*>(checked_mapped(_cells)),
        static_cast<std::size_t>(_config.tile_rows) * cols);
    for (const std::uint32_t slot : block_emitters) {
        const std::size_t row = slot - row_offset;
        auto& host_row = _host_rows[slot];
        // Cells add with wrap: the lost column may carry negative (wrapped)
        // Russian-roulette adjustments.
        for (std::size_t col = 0; col < cols; ++col) {
            const std::uint64_t cell = scratch[(row * cols) + col];
            if (cell != 0) {
                host_row[static_cast<std::uint32_t>(col)] += cell;
            }
        }
    }
}

float ExchangeAccumImpl::prepare_batch(std::uint64_t rays_per_face) {
    if (_fp_scale <= 0.0) {
        _fp_scale = select_fp_scale(rays_per_face);
    }
    if (_rays_per_face + rays_per_face > max_cumulative_rays(_fp_scale)) {
        throw std::invalid_argument(
            "pycanha::radiative: cumulative rays_per_face exceeds the "
            "fixed-point accumulation range; reset the accumulator or use "
            "fewer rays");
    }
    return static_cast<float>(_fp_scale);
}

void ExchangeAccumImpl::record_batch(std::span<const std::uint32_t> emitters,
                                     std::uint64_t rays_per_face) {
    for (const std::uint32_t slot : emitters) {
        _rays_per_row[slot] += rays_per_face;
    }
    _rays_per_face += rays_per_face;
    _total_rays += rays_per_face * emitters.size();
}

std::uint64_t ExchangeAccumImpl::cell_at(std::size_t row,
                                         std::size_t col) const {
    if (_config.layout == AccumLayout::Dense) {
        const std::size_t slots = _scene.num_face_slots();
        const std::size_t cols = matrix_columns(slots);
        const std::span<const std::uint64_t> cells(
            static_cast<const std::uint64_t*>(checked_mapped(_cells)),
            slots * cols);
        return cells[(row * cols) + col];
    }
    const auto& host_row = _host_rows[row];
    const auto it = host_row.find(static_cast<std::uint32_t>(col));
    return it == host_row.end() ? 0 : it->second;
}

void ExchangeAccumImpl::scan_row(std::size_t row, std::uint64_t rays_row,
                                 std::vector<std::int32_t>& indices,
                                 std::vector<double>& values,
                                 EntryStats& stats) const {
    const std::size_t slots = _scene.num_face_slots();
    const std::size_t cols = matrix_columns(slots);
    const std::size_t lost_col =
        slots + static_cast<std::size_t>(lost_column_offset);
    const double inv_scale = 1.0 / _fp_scale;
    // Ascending column order in both layouts, so the CSR (and every derived
    // statistic) is bit-identical between Dense and Tiled.
    for (std::size_t col = 0; col < cols; ++col) {
        const std::uint64_t cell = cell_at(row, col);
        if (cell == 0) {
            continue;
        }
        // The lost column is signed: Russian-roulette boost withdrawals may
        // push it (slightly) negative. The fixed-point value converts FIRST
        // (exact for full deposits), THEN divides by the rays — the same
        // rounding path as the vf counts, so blackbody exchange factors are
        // bit-identical to view factors.
        const double energy =
            col == lost_col
                ? static_cast<double>(static_cast<std::int64_t>(cell)) *
                      inv_scale
                : static_cast<double>(cell) * inv_scale;
        const double factor = energy / static_cast<double>(rays_row);
        if (col == lost_col) {
            stats.lost_energy += energy;
        }
        if (col < slots) {
            // Conservative per-entry standard error: a ray's deposit into
            // one cell is in [0, 1], so the Bernoulli bound dominates the
            // true variance.
            const double entry_stderr =
                std::sqrt(factor * std::max(1.0 - factor, 0.0) /
                          static_cast<double>(rays_row));
            stats.stderr_sum += entry_stderr;
            stats.stderr_max = std::max(stats.stderr_max, entry_stderr);
            ++stats.entries;
        }
        // Row statistics come before thresholding; the threshold only
        // prunes what is stored.
        if (std::abs(factor) > _config.sparse_threshold) {
            indices.push_back(static_cast<std::int32_t>(col));
            values.push_back(factor);
        }
    }
}

ExchangeResult ExchangeAccumImpl::build_result() const {
    if (_config.layout == AccumLayout::Dense) {
        invalidate_host_visible(_scene, _cells);
    }
    const std::size_t slots = _scene.num_face_slots();

    ExchangeResult result;
    result.band = _band;
    result.factors.rows = static_cast<std::int64_t>(slots);
    result.factors.cols = static_cast<std::int64_t>(matrix_columns(slots));
    result.factors.indptr.resize(static_cast<Eigen::Index>(slots) + 1);

    std::vector<std::int32_t> indices;
    std::vector<double> values;
    EntryStats stats;
    double emitted_energy = 0.0;

    result.factors.indptr(0) = 0;
    for (std::size_t row = 0; row < slots; ++row) {
        const std::uint64_t rays_row = _rays_per_row[row];
        if (rays_row > 0 && _fp_scale > 0.0) {
            scan_row(row, rays_row, indices, values, stats);
            emitted_energy += static_cast<double>(rays_row);
        }
        result.factors.indptr(static_cast<Eigen::Index>(row) + 1) =
            static_cast<std::int64_t>(values.size());
    }

    result.factors.indices = Eigen::Map<const Eigen::VectorX<std::int32_t>>(
        indices.data(), static_cast<Eigen::Index>(indices.size()));
    result.factors.values = Eigen::Map<const Eigen::VectorXd>(
        values.data(), static_cast<Eigen::Index>(values.size()));

    result.stats.total_rays = _total_rays;
    result.stats.rays_per_face = _rays_per_face;
    result.stats.mean_stderr =
        stats.entries > 0
            ? stats.stderr_sum / static_cast<double>(stats.entries)
            : 0.0;
    result.stats.max_stderr = stats.stderr_max;
    result.stats.lost_energy_fraction =
        emitted_energy > 0.0 ? stats.lost_energy / emitted_energy : 0.0;
    return result;
}

std::uint64_t ExchangeAccumImpl::conservation_error() const {
    if (_fp_scale <= 0.0) {
        return 0;
    }
    if (_config.layout == AccumLayout::Dense) {
        invalidate_host_visible(_scene, _cells);
    }
    const std::size_t slots = _scene.num_face_slots();
    const std::size_t cols = matrix_columns(slots);
    const auto scale = static_cast<std::uint64_t>(_fp_scale);
    std::uint64_t max_error = 0;
    for (std::size_t row = 0; row < slots; ++row) {
        if (_rays_per_row[row] == 0) {
            continue;
        }
        // Everything wraps mod 2^64 — the identity the kernel maintains.
        // The virtual columns are part of the balance, so the full row
        // accounts for every parcel of emitted energy.
        std::uint64_t balance = 0;
        for (std::size_t col = 0; col < cols; ++col) {
            balance += cell_at(row, col);
        }
        const std::uint64_t expected = _rays_per_row[row] * scale;
        max_error = std::max(max_error, wrap_magnitude(balance - expected));
    }
    return max_error;
}

SolarAccumImpl::SolarAccumImpl(SceneImpl& scene) : _scene(scene) {
    const std::uint64_t slots = _scene.num_face_slots();
    _direct = _scene.create_buffer(slots * sizeof(std::uint64_t),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   /*host_visible=*/true);
    _total = _scene.create_buffer(slots * sizeof(std::uint64_t),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  /*host_visible=*/true);
    const std::span<const double> areas = _scene.face_areas();
    const double total_area = std::accumulate(areas.begin(), areas.end(), 0.0);
    _area_units = std::max<std::uint64_t>(
        1, static_cast<std::uint64_t>(std::ceil(total_area)));
    reset();
}

SolarAccumImpl::~SolarAccumImpl() {
    _scene.destroy_buffer(_direct);
    _scene.destroy_buffer(_total);
}

void SolarAccumImpl::reset() {
    clear_host_visible(_scene, _direct);
    clear_host_visible(_scene, _total);
    _sun_recorded = false;
    _fp_scale = 0.0;
    _rays_per_face = 0;
    _total_rays = 0;
}

SolarAccumImpl::BatchSetup SolarAccumImpl::prepare_batch(
    const SolarState& sun, std::uint64_t rays_per_face) {
    const double norm = sun.direction.norm();
    if (!(norm > 0.0)) {
        throw std::invalid_argument(
            "pycanha::radiative: the sun direction must be a nonzero vector");
    }
    if (sun.irradiance < 0.0) {
        throw std::invalid_argument(
            "pycanha::radiative: the solar irradiance cannot be negative");
    }
    const Vector3D direction = sun.direction / norm;
    if (!_sun_recorded) {
        _sun.direction = direction;
        _sun.irradiance = sun.irradiance;
        _sun_recorded = true;
    } else if (!(_sun.direction.array() == direction.array()).all() ||
               _sun.irradiance != sun.irradiance) {
        throw std::invalid_argument(
            "pycanha::radiative: all batches of a solar accumulator must use "
            "the same SolarState; use a fresh accumulator per sun snapshot");
    }
    if (_fp_scale <= 0.0) {
        _fp_scale = select_fp_scale(rays_per_face * _area_units);
    }
    if ((_rays_per_face + rays_per_face) * _area_units >
        max_cumulative_rays(_fp_scale)) {
        throw std::invalid_argument(
            "pycanha::radiative: cumulative rays_per_face exceeds the "
            "fixed-point accumulation range; reset the accumulator or use "
            "fewer rays");
    }
    return BatchSetup{.sun_dir = {static_cast<float>(direction.x()),
                                  static_cast<float>(direction.y()),
                                  static_cast<float>(direction.z())},
                      .fp_scale = static_cast<float>(_fp_scale)};
}

void SolarAccumImpl::record_batch(std::uint64_t rays_per_face,
                                  std::size_t num_emitters) {
    _rays_per_face += rays_per_face;
    _total_rays += rays_per_face * num_emitters;
}

SolarResult SolarAccumImpl::build_result() const {
    const auto slots = static_cast<Eigen::Index>(_scene.num_face_slots());
    SolarResult result;
    result.direct = Eigen::VectorXd::Zero(slots);
    result.total = Eigen::VectorXd::Zero(slots);
    result.stats.total_rays = _total_rays;
    result.stats.rays_per_face = _rays_per_face;
    if (_rays_per_face == 0 || _fp_scale <= 0.0) {
        return result;
    }

    invalidate_host_visible(_scene, _direct);
    invalidate_host_visible(_scene, _total);
    const std::span<const std::uint64_t> direct(
        static_cast<const std::uint64_t*>(checked_mapped(_direct)),
        static_cast<std::size_t>(slots));
    const std::span<const std::uint64_t> total(
        static_cast<const std::uint64_t*>(checked_mapped(_total)),
        static_cast<std::size_t>(slots));

    const std::span<const double> areas = _scene.face_areas();
    const auto rays = static_cast<double>(_rays_per_face);
    const double norm = (1.0 / _fp_scale) / rays;
    double stderr_sum = 0.0;
    double stderr_max = 0.0;
    std::size_t stderr_entries = 0;
    for (Eigen::Index slot = 0; slot < slots; ++slot) {
        const auto index = static_cast<std::size_t>(slot);
        const double area = areas[index];
        if (area <= 0.0) {
            continue;  // no geometry on this slot, nothing was deposited
        }
        // Deposits already carry the emitting face's area, so the
        // accumulated cells ARE the absorbed watts (per unit irradiance).
        const double direct_watts =
            _sun.irradiance * static_cast<double>(direct[index]) * norm;
        const double total_watts =
            _sun.irradiance * static_cast<double>(total[index]) * norm;
        result.direct(slot) = direct_watts;
        result.total(slot) = total_watts;
        if (total_watts > 0.0) {
            // Advisory precision estimate: the Bernoulli bound on the flux
            // fraction (exact only without concentration above 1 sun),
            // scaled back to watts.
            const double flux_fraction = total_watts / (_sun.irradiance * area);
            const double entry_stderr =
                _sun.irradiance * area *
                std::sqrt(std::min(flux_fraction, 1.0) *
                          std::max(1.0 - flux_fraction, 0.0) / rays);
            stderr_sum += entry_stderr;
            stderr_max = std::max(stderr_max, entry_stderr);
            ++stderr_entries;
        }
    }
    result.stats.mean_stderr =
        stderr_entries > 0 ? stderr_sum / static_cast<double>(stderr_entries)
                           : 0.0;
    result.stats.max_stderr = stderr_max;
    return result;
}

}  // namespace pycanha::radiative::detail
