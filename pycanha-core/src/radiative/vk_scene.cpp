#include "vk_scene.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/radiative/kernels/vf_spv.h"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/scene_part.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "vk_accum.hpp"
#include "vk_device.hpp"

namespace pycanha::radiative::detail {

namespace {

// Keep single GPU submissions well under the ~2 s Windows watchdog (TDR)
// limit by bounding the rays per dispatch. TODO(radiative): replace the
// fixed bound with timestamp-based calibration targeting ~0.25 s chunks.
constexpr std::uint64_t max_rays_per_chunk_total = 1U << 22U;

constexpr std::uint32_t workgroup_size_x = 64;
constexpr std::uint32_t vf_num_bindings = 11;

// Enum names that do not fit the line-length limit where they are needed.
constexpr VkStructureType stype_triangles_data =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
constexpr VkStructureType stype_instances_data =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
constexpr VkStructureType stype_accel_address_info =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;

// Named usage combinations (the enum constants are signed, so combining
// them inline would mix signed operands in bitwise expressions).
constexpr auto geometry_usage =
    static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) |
    static_cast<VkBufferUsageFlags>(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) |
    static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
constexpr auto accel_storage_usage =
    static_cast<VkBufferUsageFlags>(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) |
    static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
constexpr auto scratch_usage =
    static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) |
    static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
constexpr auto instance_input_usage =
    static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) |
    static_cast<VkBufferUsageFlags>(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("pycanha::radiative: ") + what + " failed (VkResult " +
            std::to_string(static_cast<int>(result)) + ")");
    }
}

void write_transform_rows(const gmm::CoordinateTransformation& tf,
                          std::array<std::array<float, 4>, 3>& rows) {
    const Eigen::Matrix3d& rotation = tf.rotation();
    const Vector3D& translation = tf.translation();
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            rows.at(row).at(col) =
                static_cast<float>(rotation(static_cast<Eigen::Index>(row),
                                            static_cast<Eigen::Index>(col)));
        }
        rows.at(row).at(3) =
            static_cast<float>(translation(static_cast<Eigen::Index>(row)));
    }
}

VkTransformMatrixKHR to_vk_transform(const InstanceDataGpu& instance) {
    // VkTransformMatrixKHR is row-major 3x4 — identical to tf_rows.
    VkTransformMatrixKHR out;
    std::memcpy(&out.matrix, instance.tf_rows.data(), sizeof(out.matrix));
    return out;
}

}  // namespace

GpuBuffer SceneImpl::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                   bool host_visible) const {
    GpuBuffer out;
    out.size = size;
    const VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr};
    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    if (host_visible) {
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    VmaAllocationInfo result_info{};
    check(vmaCreateBuffer(_device.allocator, &buffer_info, &alloc_info,
                          &out.buffer, &out.allocation, &result_info),
          "buffer allocation");
    out.mapped = result_info.pMappedData;
    return out;
}

void SceneImpl::destroy_buffer(GpuBuffer& buffer) const noexcept {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(_device.allocator, buffer.buffer, buffer.allocation);
        buffer = GpuBuffer{};
    }
}

GpuBuffer SceneImpl::upload_to_new_buffer(const void* data, std::size_t bytes,
                                          VkBufferUsageFlags usage) const {
    GpuBuffer buffer = create_buffer(std::max<std::size_t>(bytes, 4), usage,
                                     /*host_visible=*/true);
    if (bytes > 0) {
        std::memcpy(checked_mapped(buffer), data, bytes);
    }
    vmaFlushAllocation(_device.allocator, buffer.allocation, 0, VK_WHOLE_SIZE);
    return buffer;
}

VkDeviceAddress SceneImpl::buffer_address(const GpuBuffer& buffer) const {
    const VkBufferDeviceAddressInfo info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = buffer.buffer};
    return vkGetBufferDeviceAddress(_device.device, &info);
}

template <class Record>
void SceneImpl::submit_once(Record&& record) {
    const VkCommandBufferAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = _command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(_device.device, &alloc_info, &cmd),
          "command buffer allocation");
    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr};
    check(vkBeginCommandBuffer(cmd, &begin_info), "command buffer begin");
    std::forward<Record>(record)(cmd);
    check(vkEndCommandBuffer(cmd), "command buffer end");

    const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                              .pNext = nullptr,
                              .waitSemaphoreCount = 0,
                              .pWaitSemaphores = nullptr,
                              .pWaitDstStageMask = nullptr,
                              .commandBufferCount = 1,
                              .pCommandBuffers = &cmd,
                              .signalSemaphoreCount = 0,
                              .pSignalSemaphores = nullptr};
    check(vkResetFences(_device.device, 1, &_fence), "fence reset");
    check(vkQueueSubmit(_device.compute_queue, 1, &submit, _fence),
          "queue submit");
    check(vkWaitForFences(_device.device, 1, &_fence, VK_TRUE,
                          std::numeric_limits<std::uint64_t>::max()),
          "fence wait");
    vkFreeCommandBuffers(_device.device, _command_pool, 1, &cmd);
}

SceneImpl::SceneImpl(DeviceImpl& device, std::vector<ScenePart> parts,
                     MaterialTable materials)
    : _device(device), _materials(std::move(materials)) {
    if (parts.empty()) {
        throw std::invalid_argument(
            "pycanha::radiative: a scene needs at least one part");
    }
    _num_slots = static_cast<std::uint32_t>(_materials.face_material.rows());
    if (_num_slots == 0 || (_num_slots % 2) != 0) {
        throw std::invalid_argument(
            "pycanha::radiative: material table has no face slots (build it "
            "from the same model as the parts)");
    }
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (parts[i].part_id != i) {
            throw std::invalid_argument(
                "pycanha::radiative: part_id must equal the part's position "
                "in the vector");
        }
        if (parts[i].mesh.nt() == 0) {
            throw std::invalid_argument("pycanha::radiative: part " +
                                        std::to_string(i) +
                                        " has an empty mesh");
        }
        if (static_cast<std::uint32_t>(parts[i].mesh.nf()) > _num_slots) {
            throw std::invalid_argument(
                "pycanha::radiative: part " + std::to_string(i) +
                " has face ids beyond the material table");
        }
    }

    const VkCommandPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = _device.compute_family};
    check(vkCreateCommandPool(_device.device, &pool_info, nullptr,
                              &_command_pool),
          "command pool creation");
    const VkFenceCreateInfo fence_info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0};
    check(vkCreateFence(_device.device, &fence_info, nullptr, &_fence),
          "fence creation");

    upload_geometry(parts);
    build_emission_tables(parts);
    build_blas(parts);
    write_instance_buffers();
    build_tlas_first();
    create_pipeline();

    SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                       "radiative: scene built ({} parts, {} face slots)",
                       parts.size(), _num_slots);
}

SceneImpl::~SceneImpl() {
    vkDeviceWaitIdle(_device.device);
    if (_vf_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(_device.device, _vf_pipeline, nullptr);
    }
    if (_pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(_device.device, _pipeline_layout, nullptr);
    }
    if (_descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(_device.device, _descriptor_pool, nullptr);
    }
    if (_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(_device.device, _set_layout, nullptr);
    }
    if (_tlas != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR(_device.device, _tlas, nullptr);
    }
    for (PartGpu& part : _parts) {
        if (part.blas != VK_NULL_HANDLE) {
            vkDestroyAccelerationStructureKHR(_device.device, part.blas,
                                              nullptr);
        }
        destroy_buffer(part.vertices);
        destroy_buffer(part.indices);
        destroy_buffer(part.tri_face);
        destroy_buffer(part.blas_storage);
    }
    destroy_buffer(_instance_ssbo);
    destroy_buffer(_tlas_instances);
    destroy_buffer(_tlas_storage);
    destroy_buffer(_tlas_scratch);
    destroy_buffer(_materials_buf);
    destroy_buffer(_face_material_buf);
    destroy_buffer(_face_flags_buf);
    destroy_buffer(_emit_tri_offset_buf);
    destroy_buffer(_emit_tri_part_buf);
    destroy_buffer(_emit_tri_prim_buf);
    destroy_buffer(_emit_cum_area_buf);
    destroy_buffer(_emitters_buf);
    if (_fence != VK_NULL_HANDLE) {
        vkDestroyFence(_device.device, _fence, nullptr);
    }
    if (_command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(_device.device, _command_pool, nullptr);
    }
}

void SceneImpl::upload_geometry(const std::vector<ScenePart>& parts) {
    _face_areas.assign(_num_slots, 0.0);
    _parts.resize(parts.size());
    _instances_host.resize(parts.size());

    // World bounding box of the non-celestial parts: the self-intersection
    // epsilon must follow the spacecraft scale, not the planet distance.
    Eigen::AlignedBox3d world_box;
    for (std::size_t p = 0; p < parts.size(); ++p) {
        upload_part(parts[p], p, world_box);
    }
    const double characteristic =
        world_box.isEmpty() ? 1.0 : world_box.diagonal().norm();
    _ray_tmin_scale = static_cast<float>(1e-4 * std::max(characteristic, 1e-6));

    build_face_tables(parts);
}

void SceneImpl::upload_part(const ScenePart& part, std::size_t index,
                            Eigen::AlignedBox3d& world_box) {
    const gmm::TriMeshF& mesh = part.mesh;
    const auto num_points = static_cast<Eigen::Index>(mesh.np());
    const auto num_triangles = static_cast<Eigen::Index>(mesh.nt());
    PartGpu& gpu = _parts[index];
    gpu.num_triangles = static_cast<std::uint32_t>(num_triangles);

    // Eigen matrices are column-major in memory: repack row by row into the
    // tightly-packed layouts the shaders index by pointer.
    std::vector<float> vertices(static_cast<std::size_t>(num_points) * 3);
    for (Eigen::Index v = 0; v < num_points; ++v) {
        const std::size_t base = static_cast<std::size_t>(v) * 3;
        vertices[base + 0] = mesh.vertices(v, 0);
        vertices[base + 1] = mesh.vertices(v, 1);
        vertices[base + 2] = mesh.vertices(v, 2);
    }
    std::vector<std::uint32_t> indices(static_cast<std::size_t>(num_triangles) *
                                       3);
    std::vector<std::uint32_t> tri_face(
        static_cast<std::size_t>(num_triangles));
    for (Eigen::Index t = 0; t < num_triangles; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * 3;
        indices[base + 0] = mesh.triangles(t, 0);
        indices[base + 1] = mesh.triangles(t, 1);
        indices[base + 2] = mesh.triangles(t, 2);
        tri_face[static_cast<std::size_t>(t)] = mesh.face_ids(t);
    }

    gpu.vertices = upload_to_new_buffer(
        vertices.data(), vertices.size() * sizeof(float), geometry_usage);
    gpu.indices = upload_to_new_buffer(
        indices.data(), indices.size() * sizeof(std::uint32_t), geometry_usage);
    gpu.tri_face = upload_to_new_buffer(tri_face.data(),
                                        tri_face.size() * sizeof(std::uint32_t),
                                        geometry_usage);

    // Rigid transforms preserve areas: part-local areas are world areas.
    const Eigen::VectorXd part_areas =
        gmm::mesh::ops::compute_face_slot_areas(mesh);
    for (Eigen::Index slot = 0; slot < part_areas.rows(); ++slot) {
        _face_areas[static_cast<std::size_t>(slot)] += part_areas(slot);
    }

    InstanceDataGpu& instance = _instances_host[index];
    instance.part_id = part.part_id;
    instance.flags = part.kind == PartKind::CelestialBody ? 1U : 0U;
    write_transform_rows(part.transform, instance.tf_rows);

    if (part.kind != PartKind::CelestialBody) {
        constexpr std::array<Eigen::AlignedBox3d::CornerType, 8> corners = {
            Eigen::AlignedBox3d::BottomLeftFloor,
            Eigen::AlignedBox3d::BottomRightFloor,
            Eigen::AlignedBox3d::TopLeftFloor,
            Eigen::AlignedBox3d::TopRightFloor,
            Eigen::AlignedBox3d::BottomLeftCeil,
            Eigen::AlignedBox3d::BottomRightCeil,
            Eigen::AlignedBox3d::TopLeftCeil,
            Eigen::AlignedBox3d::TopRightCeil};
        const Eigen::AlignedBox3d local_box =
            gmm::mesh::ops::bounding_box(mesh);
        for (const auto corner : corners) {
            world_box.extend(part.transform.apply(local_box.corner(corner)));
        }
    }
}

void SceneImpl::build_face_tables(const std::vector<ScenePart>& parts) {
    // Global per-slot tables. Missing material (-1) is tolerated (treated
    // as blackbody by the exchange kernels); inactive slots never emit.
    std::vector<float> material_rows(
        static_cast<std::size_t>(_materials.properties.rows()) * 6);
    for (Eigen::Index row = 0; row < _materials.properties.rows(); ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * 6;
        for (int dof = 0; dof < 6; ++dof) {
            material_rows[base + static_cast<std::size_t>(dof)] =
                _materials.properties(row, dof);
        }
    }
    std::vector<std::int32_t> face_material(_num_slots);
    std::vector<std::uint32_t> face_flags(_num_slots, 0);
    for (std::uint32_t slot = 0; slot < _num_slots; ++slot) {
        face_material[slot] = _materials.face_material(slot);
        if (_materials.face_active(slot)) {
            face_flags[slot] |= 1U;
        }
    }
    for (const ScenePart& part : parts) {
        if (part.kind != PartKind::CelestialBody) {
            continue;
        }
        const auto num_triangles = static_cast<Eigen::Index>(part.mesh.nt());
        for (Eigen::Index t = 0; t < num_triangles; ++t) {
            const auto base = part.mesh.face_ids(t);
            face_flags[base] |= 2U;
            face_flags[base + 1U] |= 2U;
        }
    }

    constexpr VkBufferUsageFlags table_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    _materials_buf =
        upload_to_new_buffer(material_rows.data(),
                             material_rows.size() * sizeof(float), table_usage);
    _face_material_buf = upload_to_new_buffer(
        face_material.data(), face_material.size() * sizeof(std::int32_t),
        table_usage);
    _face_flags_buf = upload_to_new_buffer(
        face_flags.data(), face_flags.size() * sizeof(std::uint32_t),
        table_usage);

    // Default emitter list: active, non-planet slots with geometry.
    _default_emitters.clear();
    for (std::uint32_t slot = 0; slot < _num_slots; ++slot) {
        if ((face_flags[slot] & 1U) != 0U && (face_flags[slot] & 2U) == 0U &&
            _face_areas[slot] > 0.0) {
            _default_emitters.push_back(slot);
        }
    }
}

void SceneImpl::build_emission_tables(const std::vector<ScenePart>& parts) {
    struct EmitTriangle {
        std::uint32_t pair_base;
        std::uint32_t part;
        std::uint32_t prim;
        double area;
    };
    std::vector<EmitTriangle> triangles;
    for (std::size_t p = 0; p < parts.size(); ++p) {
        const gmm::TriMeshF& mesh = parts[p].mesh;
        const Eigen::VectorXd areas = gmm::mesh::ops::compute_areas(mesh);
        const auto num_triangles = static_cast<Eigen::Index>(mesh.nt());
        for (Eigen::Index t = 0; t < num_triangles; ++t) {
            triangles.push_back(
                EmitTriangle{.pair_base = mesh.face_ids(t),
                             .part = static_cast<std::uint32_t>(p),
                             .prim = static_cast<std::uint32_t>(t),
                             .area = areas(t)});
        }
    }
    // Face slots partition across parts, so a stable sort by slot keeps the
    // per-part triangle order within each face.
    std::ranges::stable_sort(triangles, {}, &EmitTriangle::pair_base);

    // Per-slot offsets: even slot = first triangle of the pair, odd slot =
    // one past the last (both sides of a pair share the triangle list, so
    // the shader reads [pair_base] and [pair_base + 1]).
    std::vector<std::uint32_t> offsets(_num_slots, 0);
    std::vector<std::uint32_t> part_ids(triangles.size());
    std::vector<std::uint32_t> prim_ids(triangles.size());
    std::vector<float> cum_area(triangles.size());

    std::size_t index = 0;
    for (std::uint32_t pair = 0; pair < _num_slots; pair += 2) {
        offsets[pair] = static_cast<std::uint32_t>(index);
        const std::size_t begin = index;
        double total = 0.0;
        while (index < triangles.size() && triangles[index].pair_base == pair) {
            total += triangles[index].area;
            ++index;
        }
        double running = 0.0;
        for (std::size_t i = begin; i < index; ++i) {
            part_ids[i] = triangles[i].part;
            prim_ids[i] = triangles[i].prim;
            running += triangles[i].area;
            // Normalize to exactly 1.0 at the last triangle so the binary
            // search never falls off the end; zero-area pairs cannot be
            // emitters (filtered by area in the default list).
            cum_area[i] =
                total > 0.0 ? static_cast<float>(running / total) : 1.0F;
        }
        offsets[pair + 1U] = static_cast<std::uint32_t>(index);
    }

    constexpr VkBufferUsageFlags table_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    _emit_tri_offset_buf = upload_to_new_buffer(
        offsets.data(), offsets.size() * sizeof(std::uint32_t), table_usage);
    _emit_tri_part_buf = upload_to_new_buffer(
        part_ids.data(), part_ids.size() * sizeof(std::uint32_t), table_usage);
    _emit_tri_prim_buf = upload_to_new_buffer(
        prim_ids.data(), prim_ids.size() * sizeof(std::uint32_t), table_usage);
    _emit_cum_area_buf = upload_to_new_buffer(
        cum_area.data(), cum_area.size() * sizeof(float), table_usage);
}

void SceneImpl::build_blas(const std::vector<ScenePart>& parts) {
    for (std::size_t p = 0; p < parts.size(); ++p) {
        PartGpu& gpu = _parts[p];

        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        auto& triangles = geometry.geometry.triangles;
        triangles.sType = stype_triangles_data;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = buffer_address(gpu.vertices);
        triangles.vertexStride = 3 * sizeof(float);
        triangles.maxVertex = static_cast<std::uint32_t>(
            parts[p].mesh.np() > 0 ? parts[p].mesh.np() - 1 : 0);
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = buffer_address(gpu.indices);

        VkAccelerationStructureBuildGeometryInfoKHR build{};
        build.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build.geometryCount = 1;
        build.pGeometries = &geometry;

        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(
            _device.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &build, &gpu.num_triangles, &sizes);

        gpu.blas_storage =
            create_buffer(sizes.accelerationStructureSize, accel_storage_usage,
                          /*host_visible=*/false);
        GpuBuffer scratch = create_buffer(sizes.buildScratchSize, scratch_usage,
                                          /*host_visible=*/false);

        VkAccelerationStructureCreateInfoKHR create{};
        create.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create.buffer = gpu.blas_storage.buffer;
        create.size = sizes.accelerationStructureSize;
        create.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        check(vkCreateAccelerationStructureKHR(_device.device, &create, nullptr,
                                               &gpu.blas),
              "BLAS creation");

        build.dstAccelerationStructure = gpu.blas;
        build.scratchData.deviceAddress = buffer_address(scratch);
        const VkAccelerationStructureBuildRangeInfoKHR range{
            .primitiveCount = gpu.num_triangles,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0};
        const std::array<const VkAccelerationStructureBuildRangeInfoKHR*, 1>
            ranges = {&range};
        submit_once([&build, &ranges](VkCommandBuffer cmd) {
            vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, ranges.data());
        });
        destroy_buffer(scratch);

        const VkAccelerationStructureDeviceAddressInfoKHR address_info{
            .sType = stype_accel_address_info,
            .pNext = nullptr,
            .accelerationStructure = gpu.blas};
        gpu.blas_address = vkGetAccelerationStructureDeviceAddressKHR(
            _device.device, &address_info);
    }
}

void SceneImpl::write_instance_buffers() {
    const std::size_t count = _instances_host.size();
    if (_instance_ssbo.buffer == VK_NULL_HANDLE) {
        _instance_ssbo = create_buffer(count * sizeof(InstanceDataGpu),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       /*host_visible=*/true);
        _tlas_instances =
            create_buffer(count * sizeof(VkAccelerationStructureInstanceKHR),
                          instance_input_usage, /*host_visible=*/true);
    }
    const std::span<VkAccelerationStructureInstanceKHR> tlas_span(
        static_cast<VkAccelerationStructureInstanceKHR*>(
            checked_mapped(_tlas_instances)),
        count);
    for (std::size_t p = 0; p < count; ++p) {
        InstanceDataGpu& instance = _instances_host[p];
        instance.vertices_addr = buffer_address(_parts[p].vertices);
        instance.indices_addr = buffer_address(_parts[p].indices);
        instance.tri_face_addr = buffer_address(_parts[p].tri_face);

        VkAccelerationStructureInstanceKHR vk_instance{};
        vk_instance.transform = to_vk_transform(instance);
        // 24-bit field; part counts are far below the 2^24 limit.
        vk_instance.instanceCustomIndex =
            static_cast<std::uint32_t>(p) & 0xFFFFFFU;
        vk_instance.mask = 0xFF;
        vk_instance.instanceShaderBindingTableRecordOffset = 0;
        vk_instance.flags = 0;
        vk_instance.accelerationStructureReference = _parts[p].blas_address;
        tlas_span[p] = vk_instance;
    }
    std::memcpy(checked_mapped(_instance_ssbo), _instances_host.data(),
                count * sizeof(InstanceDataGpu));
    vmaFlushAllocation(_device.allocator, _instance_ssbo.allocation, 0,
                       VK_WHOLE_SIZE);
    vmaFlushAllocation(_device.allocator, _tlas_instances.allocation, 0,
                       VK_WHOLE_SIZE);
}

void SceneImpl::build_tlas_first() {
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = stype_instances_data;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress =
        buffer_address(_tlas_instances);

    VkAccelerationStructureBuildGeometryInfoKHR build{};
    build.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geometry;

    auto instance_count = static_cast<std::uint32_t>(_parts.size());
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        _device.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
        &instance_count, &sizes);

    _tlas_storage =
        create_buffer(sizes.accelerationStructureSize,
                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                      /*host_visible=*/false);
    _tlas_scratch =
        create_buffer(std::max(sizes.buildScratchSize, sizes.updateScratchSize),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                      /*host_visible=*/false);

    VkAccelerationStructureCreateInfoKHR create{};
    create.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    create.buffer = _tlas_storage.buffer;
    create.size = sizes.accelerationStructureSize;
    create.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    check(vkCreateAccelerationStructureKHR(_device.device, &create, nullptr,
                                           &_tlas),
          "TLAS creation");
    rebuild_tlas();
}

void SceneImpl::rebuild_tlas() {
    // Full rebuild, not refit: N instances is tiny and rebuilds keep
    // traversal quality under large rotations.
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = stype_instances_data;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress =
        buffer_address(_tlas_instances);

    VkAccelerationStructureBuildGeometryInfoKHR build{};
    build.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.dstAccelerationStructure = _tlas;
    build.geometryCount = 1;
    build.pGeometries = &geometry;
    build.scratchData.deviceAddress = buffer_address(_tlas_scratch);

    const VkAccelerationStructureBuildRangeInfoKHR range{
        .primitiveCount = static_cast<std::uint32_t>(_parts.size()),
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0};
    const std::array<const VkAccelerationStructureBuildRangeInfoKHR*, 1>
        ranges = {&range};
    submit_once([&build, &ranges](VkCommandBuffer cmd) {
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, ranges.data());
    });
}

void SceneImpl::set_part_transform(
    std::uint32_t part_id, const gmm::CoordinateTransformation& world_tf) {
    if (part_id >= _instances_host.size()) {
        throw std::invalid_argument("pycanha::radiative: unknown part_id " +
                                    std::to_string(part_id));
    }
    write_transform_rows(world_tf, _instances_host[part_id].tf_rows);
}

void SceneImpl::commit() {
    write_instance_buffers();
    rebuild_tlas();
}

void SceneImpl::create_pipeline() {
    std::array<VkDescriptorSetLayoutBinding, vf_num_bindings> bindings{};
    std::uint32_t binding_index = 0;
    for (VkDescriptorSetLayoutBinding& binding : bindings) {
        binding = VkDescriptorSetLayoutBinding{
            .binding = binding_index,
            .descriptorType =
                binding_index == 0
                    ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                    : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr};
        ++binding_index;
    }
    const VkDescriptorSetLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = vf_num_bindings,
        .pBindings = bindings.data()};
    check(vkCreateDescriptorSetLayout(_device.device, &layout_info, nullptr,
                                      &_set_layout),
          "descriptor set layout creation");

    const VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants)};
    const VkPipelineLayoutCreateInfo pipeline_layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range};
    check(vkCreatePipelineLayout(_device.device, &pipeline_layout_info, nullptr,
                                 &_pipeline_layout),
          "pipeline layout creation");

    const VkShaderModuleCreateInfo module_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = sizeof(kernels::vf_spv),
        .pCode = static_cast<const std::uint32_t*>(kernels::vf_spv)};
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(_device.device, &module_info, nullptr, &module),
          "VF shader module creation");

    const VkPipelineShaderStageCreateInfo stage_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = module,
        .pName = "main",  // slangc renames the entry point
        .pSpecializationInfo = nullptr};
    const VkComputePipelineCreateInfo pipeline_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = stage_info,
        .layout = _pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0};
    const VkResult pipeline_result =
        vkCreateComputePipelines(_device.device, VK_NULL_HANDLE, 1,
                                 &pipeline_info, nullptr, &_vf_pipeline);
    vkDestroyShaderModule(_device.device, module, nullptr);
    check(pipeline_result, "VF pipeline creation");

    const std::array<VkDescriptorPoolSize, 2> pool_sizes{
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .descriptorCount = 1},
        VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .descriptorCount = vf_num_bindings - 1}};
    const VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = 1,
        .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data()};
    check(vkCreateDescriptorPool(_device.device, &pool_info, nullptr,
                                 &_descriptor_pool),
          "descriptor pool creation");
    const VkDescriptorSetAllocateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = _descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &_set_layout};
    check(vkAllocateDescriptorSets(_device.device, &set_info, &_descriptor_set),
          "descriptor set allocation");

    const VkWriteDescriptorSetAccelerationStructureKHR tlas_write_info{
        .sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .pNext = nullptr,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &_tlas};
    VkWriteDescriptorSet tlas_write{};
    tlas_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    tlas_write.pNext = &tlas_write_info;
    tlas_write.dstSet = _descriptor_set;
    tlas_write.dstBinding = 0;
    tlas_write.descriptorCount = 1;
    tlas_write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    vkUpdateDescriptorSets(_device.device, 1, &tlas_write, 0, nullptr);

    // Static bindings (1-4, 6-9) are written once; the emitter list (5) and
    // the accumulator (10) are (re)written per accumulate call.
    write_storage_descriptor(1, _instance_ssbo.buffer);
    write_storage_descriptor(2, _materials_buf.buffer);
    write_storage_descriptor(3, _face_material_buf.buffer);
    write_storage_descriptor(4, _face_flags_buf.buffer);
    write_storage_descriptor(6, _emit_tri_offset_buf.buffer);
    write_storage_descriptor(7, _emit_tri_part_buf.buffer);
    write_storage_descriptor(8, _emit_tri_prim_buf.buffer);
    write_storage_descriptor(9, _emit_cum_area_buf.buffer);
}

void SceneImpl::write_storage_descriptor(std::uint32_t binding,
                                         VkBuffer buffer) const {
    const VkDescriptorBufferInfo info{
        .buffer = buffer, .offset = 0, .range = VK_WHOLE_SIZE};
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = _descriptor_set,
        .dstBinding = binding,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &info,
        .pTexelBufferView = nullptr};
    vkUpdateDescriptorSets(_device.device, 1, &write, 0, nullptr);
}

void SceneImpl::accumulate_vf(VfAccumImpl& acc, const TraceSettings& settings,
                              std::span<const std::uint32_t> emitters) {
    const std::vector<std::uint32_t>& all = _default_emitters;
    std::vector<std::uint32_t> list(emitters.begin(), emitters.end());
    if (list.empty()) {
        list = all;
    }
    if (list.empty() || settings.rays_per_face == 0) {
        return;
    }
    const std::uint32_t num_slots = _num_slots;
    if (std::ranges::any_of(list, [num_slots](const std::uint32_t slot) {
            return slot >= num_slots;
        })) {
        throw std::invalid_argument(
            "pycanha::radiative: emitter slot out of range");
    }
    // u32 counting cells: the CUMULATIVE per-face ray count must stay below
    // 2^31 or cells could overflow.
    if (acc.rays_per_face() + settings.rays_per_face > (1ULL << 31U)) {
        throw std::invalid_argument(
            "pycanha::radiative: cumulative rays_per_face exceeds the u32 "
            "counting range; reset the accumulator or use fewer rays");
    }

    // (Re)upload the emitter list, growing the buffer when needed.
    const VkDeviceSize needed = list.size() * sizeof(std::uint32_t);
    if (_emitters_buf.buffer == VK_NULL_HANDLE || _emitters_buf.size < needed) {
        destroy_buffer(_emitters_buf);
        _emitters_buf = create_buffer(
            needed, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, /*host_visible=*/true);
    }
    std::memcpy(checked_mapped(_emitters_buf), list.data(), needed);
    vmaFlushAllocation(_device.allocator, _emitters_buf.allocation, 0,
                       VK_WHOLE_SIZE);

    write_storage_descriptor(5, _emitters_buf.buffer);
    write_storage_descriptor(10, acc.buffer());

    const float ray_tmin_scale = _ray_tmin_scale;
    PushConstants push{
        .row_offset = 0,  // dense accumulator: rows are absolute slots
        .num_emitters = static_cast<std::uint32_t>(list.size()),
        .rays_this_chunk = 0,   // set per chunk below
        .ray_index_offset = 0,  // set per chunk below
        .batch_seed = settings.seed,
        .max_bounces = settings.max_bounces,
        .flags = 0,
        .num_face_slots = num_slots,
        .energy_threshold = settings.energy_threshold,
        .fp_scale = 1.0F,  // VF counts are unscaled integers
        .inv_fp_scale = 1.0F,
        .ray_tmin_scale = ray_tmin_scale,
        .sun_dir = {0.0F, 0.0F, 0.0F},
        .pad = 0.0F};

    const std::uint64_t rays_per_chunk =
        std::max<std::uint64_t>(1, max_rays_per_chunk_total / list.size());
    std::uint64_t done = 0;
    while (done < settings.rays_per_face) {
        const std::uint64_t chunk =
            std::min(rays_per_chunk, settings.rays_per_face - done);
        push.rays_this_chunk = static_cast<std::uint32_t>(chunk);
        push.ray_index_offset = static_cast<std::uint32_t>(done);
        const auto groups_x = static_cast<std::uint32_t>(
            (chunk + workgroup_size_x - 1) / workgroup_size_x);

        submit_once([this, &push, groups_x](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              _vf_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    _pipeline_layout, 0, 1, &_descriptor_set, 0,
                                    nullptr);
            vkCmdPushConstants(cmd, _pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(PushConstants), &push);
            vkCmdDispatch(cmd, groups_x, push.num_emitters, 1);
            // Make the atomic writes visible to the host mapping.
            const VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                                 nullptr, 0, nullptr);
        });
        done += chunk;
    }

    acc.record_batch(list, settings.rays_per_face);
}

}  // namespace pycanha::radiative::detail
