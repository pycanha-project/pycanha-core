#pragma once

// src-private scene implementation: geometry/material upload, emission
// tables, acceleration structures, the VF pipeline and dispatch.
// TODO(radiative): split into vk_accel / vk_pipeline / vk_dispatch once the
// exchange and solar kernels land and the shared shapes are clear.

#include <Eigen/Geometry>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/scene_part.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vk_device.hpp"

namespace pycanha::radiative::detail {

class VfAccumImpl;

// Minimal RAII-by-owner buffer: created/destroyed by SceneImpl helpers.
struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;  // non-null only for host-visible buffers
    VkDeviceSize size = 0;
};

// Host-visible buffers are created with the mapped flag; a null mapping
// would be a logic error, and checking makes that explicit.
[[nodiscard]] inline void* checked_mapped(const GpuBuffer& buffer) {
    if (buffer.mapped == nullptr) {
        throw std::logic_error("pycanha::radiative: buffer is not host-mapped");
    }
    return buffer.mapped;
}

// C++ mirror of the InstanceData struct in kernels/common.slang — must
// match byte-for-byte (scalar layout, 80 bytes). tf_rows are rotation rows
// with the translation in .w (world = R * p + t).
struct InstanceDataGpu {
    std::uint64_t vertices_addr;
    std::uint64_t indices_addr;
    std::uint64_t tri_face_addr;
    std::uint32_t part_id;
    std::uint32_t flags;
    std::array<std::array<float, 4>, 3> tf_rows;
};
static_assert(sizeof(InstanceDataGpu) == 80);

// C++ mirror of the Push struct in kernels/common.slang (scalar layout,
// 64 bytes).
struct PushConstants {
    std::uint32_t row_offset;
    std::uint32_t num_emitters;
    std::uint32_t rays_this_chunk;
    std::uint32_t ray_index_offset;
    std::uint32_t batch_seed;
    std::uint32_t max_bounces;
    std::uint32_t flags;
    std::uint32_t num_face_slots;
    float energy_threshold;
    float fp_scale;
    float inv_fp_scale;
    float ray_tmin_scale;
    std::array<float, 3> sun_dir;
    float pad;
};
static_assert(sizeof(PushConstants) == 64);

class SceneImpl {
  public:
    SceneImpl(DeviceImpl& device, std::vector<ScenePart> parts,
              MaterialTable materials);
    ~SceneImpl();
    SceneImpl(const SceneImpl&) = delete;
    SceneImpl& operator=(const SceneImpl&) = delete;
    SceneImpl(SceneImpl&&) = delete;
    SceneImpl& operator=(SceneImpl&&) = delete;

    void set_part_transform(std::uint32_t part_id,
                            const gmm::CoordinateTransformation& world_tf);
    void commit();

    void accumulate_vf(VfAccumImpl& acc, const TraceSettings& settings,
                       std::span<const std::uint32_t> emitters);

    [[nodiscard]] std::uint32_t num_face_slots() const noexcept {
        return _num_slots;
    }
    [[nodiscard]] const MaterialTable& materials() const noexcept {
        return _materials;
    }
    [[nodiscard]] std::span<const double> face_areas() const noexcept {
        return _face_areas;
    }
    [[nodiscard]] DeviceImpl& device() const noexcept { return _device; }

    // Buffer helpers shared with the accumulator implementations.
    [[nodiscard]] GpuBuffer create_buffer(VkDeviceSize size,
                                          VkBufferUsageFlags usage,
                                          bool host_visible) const;
    void destroy_buffer(GpuBuffer& buffer) const noexcept;
    // Creates a host-visible buffer, copies `bytes` of `data` into it and
    // flushes. Empty inputs get a minimal valid buffer (Vulkan forbids
    // zero-sized ones).
    [[nodiscard]] GpuBuffer upload_to_new_buffer(
        const void* data, std::size_t bytes, VkBufferUsageFlags usage) const;

  private:
    struct PartGpu {
        GpuBuffer vertices;
        GpuBuffer indices;
        GpuBuffer tri_face;
        GpuBuffer blas_storage;
        VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
        VkDeviceAddress blas_address = 0;
        std::uint32_t num_triangles = 0;
    };

    [[nodiscard]] VkDeviceAddress buffer_address(const GpuBuffer& buffer) const;
    // Records commands into a one-shot command buffer, submits on the
    // compute queue and waits for completion.
    template <class Record>
    void submit_once(Record&& record);

    void upload_geometry(const std::vector<ScenePart>& parts);
    void upload_part(const ScenePart& part, std::size_t index,
                     Eigen::AlignedBox3d& world_box);
    void build_face_tables(const std::vector<ScenePart>& parts);
    void build_emission_tables(const std::vector<ScenePart>& parts);
    void build_blas(const std::vector<ScenePart>& parts);
    void build_tlas_first();
    void rebuild_tlas();
    void write_instance_buffers();
    void create_pipeline();
    // Points descriptor `binding` of the scene's set at `buffer`. Only valid
    // while no submitted work uses the set (every dispatch here is waited).
    void write_storage_descriptor(std::uint32_t binding, VkBuffer buffer) const;
    // Traces settings.rays_per_face rays for `emitters` (all within
    // [row_offset, row_offset + accumulator rows)) into `acc_buffer`, split
    // into watchdog-safe chunks.
    void dispatch_vf_rows(VkBuffer acc_buffer,
                          std::span<const std::uint32_t> emitters,
                          std::uint32_t row_offset,
                          const TraceSettings& settings);

    DeviceImpl& _device;
    MaterialTable _materials;
    std::uint32_t _num_slots = 0;
    std::vector<double> _face_areas;
    std::vector<std::uint32_t> _default_emitters;
    float _ray_tmin_scale = 1e-4F;

    std::vector<PartGpu> _parts;
    std::vector<InstanceDataGpu> _instances_host;  // updated by transforms

    GpuBuffer _instance_ssbo;   // InstanceDataGpu[]
    GpuBuffer _tlas_instances;  // VkAccelerationStructureInstanceKHR[]
    GpuBuffer _tlas_storage;
    GpuBuffer _tlas_scratch;
    VkAccelerationStructureKHR _tlas = VK_NULL_HANDLE;

    GpuBuffer _materials_buf;
    GpuBuffer _face_material_buf;
    GpuBuffer _face_flags_buf;
    GpuBuffer _emit_tri_offset_buf;
    GpuBuffer _emit_tri_part_buf;
    GpuBuffer _emit_tri_prim_buf;
    GpuBuffer _emit_cum_area_buf;
    GpuBuffer _emitters_buf;  // grown on demand per accumulate call

    VkDescriptorSetLayout _set_layout = VK_NULL_HANDLE;
    VkPipelineLayout _pipeline_layout = VK_NULL_HANDLE;
    VkPipeline _vf_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool _descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet _descriptor_set = VK_NULL_HANDLE;

    VkCommandPool _command_pool = VK_NULL_HANDLE;
    VkFence _fence = VK_NULL_HANDLE;
};

}  // namespace pycanha::radiative::detail
