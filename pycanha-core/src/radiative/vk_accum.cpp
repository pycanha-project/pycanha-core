#include "vk_accum.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vk_device.hpp"
#include "vk_scene.hpp"

namespace pycanha::radiative::detail {

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
    _counts = _scene.create_buffer(buffer_rows * slots * sizeof(std::uint32_t),
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

void VfAccumImpl::clear_block_scratch() {
    std::memset(checked_mapped(_counts), 0, _counts.size);
    vmaFlushAllocation(_scene.device().allocator, _counts.allocation, 0,
                       VK_WHOLE_SIZE);
}

void VfAccumImpl::absorb_block(std::span<const std::uint32_t> block_emitters,
                               std::uint32_t row_offset) {
    vmaInvalidateAllocation(_scene.device().allocator, _counts.allocation, 0,
                            VK_WHOLE_SIZE);
    const std::size_t slots = _scene.num_face_slots();
    const std::span<const std::uint32_t> scratch(
        static_cast<const std::uint32_t*>(checked_mapped(_counts)),
        static_cast<std::size_t>(_config.tile_rows) * slots);
    for (const std::uint32_t slot : block_emitters) {
        const std::size_t row = slot - row_offset;
        auto& host_row = _host_rows[slot];
        for (std::size_t col = 0; col < slots; ++col) {
            const std::uint32_t cell = scratch[(row * slots) + col];
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
        const std::span<const std::uint32_t> counts(
            static_cast<const std::uint32_t*>(checked_mapped(_counts)),
            slots * slots);
        return counts[(row * slots) + col];
    }
    const auto& host_row = _host_rows[row];
    const auto it = host_row.find(static_cast<std::uint32_t>(col));
    return it == host_row.end() ? 0 : it->second;
}

VfResult VfAccumImpl::build_result() const {
    if (_config.layout == AccumLayout::Dense) {
        vmaInvalidateAllocation(_scene.device().allocator, _counts.allocation,
                                0, VK_WHOLE_SIZE);
    }
    const std::size_t slots = _scene.num_face_slots();
    const std::span<const double> areas = _scene.face_areas();

    VfResult result;
    result.vf.rows = static_cast<std::int64_t>(slots);
    result.vf.cols = static_cast<std::int64_t>(slots);
    result.vf.indptr.resize(static_cast<Eigen::Index>(slots) + 1);
    result.row_sums = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(slots));

    std::vector<std::int32_t> indices;
    std::vector<double> values;
    double stderr_sum = 0.0;
    double stderr_max = 0.0;

    result.vf.indptr(0) = 0;
    for (std::size_t row = 0; row < slots; ++row) {
        const std::uint64_t rays_row = _rays_per_row[row];
        if (rays_row > 0) {
            double row_sum = 0.0;
            // Ascending column order in both layouts, so the CSR (and every
            // derived statistic) is bit-identical between Dense and Tiled.
            for (std::size_t col = 0; col < slots; ++col) {
                const std::uint64_t count = count_at(row, col);
                if (count == 0) {
                    continue;
                }
                const double vf =
                    static_cast<double>(count) / static_cast<double>(rays_row);
                indices.push_back(static_cast<std::int32_t>(col));
                values.push_back(vf);
                row_sum += vf;
                // Binomial standard error of the per-entry estimate.
                const double entry_stderr =
                    std::sqrt(vf * std::max(1.0 - vf, 0.0) /
                              static_cast<double>(rays_row));
                stderr_sum += entry_stderr;
                stderr_max = std::max(stderr_max, entry_stderr);
            }
            result.row_sums(static_cast<Eigen::Index>(row)) = row_sum;
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
        values.empty() ? 0.0 : stderr_sum / static_cast<double>(values.size());
    result.stats.max_stderr = stderr_max;
    result.stats.reciprocity_residual = reciprocity;
    // TODO(radiative): fill gpu_time from timestamp queries around each
    // dispatch chunk.
    return result;
}

}  // namespace pycanha::radiative::detail
