#pragma once

// src-private: volk / VMA / all Vk* usage stays below src/radiative (D3).
// Public headers must never include this file.

// This header is the module's Vulkan gateway: it exports the Vulkan API to
// every radiative TU. volk MUST precede any Vulkan-including header (it
// defines VK_NO_PROTOTYPES; a prototyped vulkan.h seen first is a hard
// error), so no other file may include vulkan/volk/VMA headers directly.
#include <volk.h>                // IWYU pragma: export
#include <vulkan/vulkan_core.h>  // IWYU pragma: export

// VMA's function-pointer sourcing (VMA_STATIC_VULKAN_FUNCTIONS=0 /
// VMA_DYNAMIC_VULKAN_FUNCTIONS=1, volk provides everything) is configured as
// target-wide compile definitions in CMake so every TU sees one config.
#include <vk_mem_alloc.h>  // IWYU pragma: export

#include <cstdint>
#include <optional>
#include <vector>

#include "pycanha-core/radiative/device.hpp"

namespace pycanha::radiative::detail {

// Process-wide loader + instance bootstrap. Created once on first use and
// kept alive for the process lifetime: volk's global dispatch table is
// loaded from this instance, so destroying/recreating instances would
// invalidate every loaded function pointer. Returns VK_NULL_HANDLE when no
// Vulkan driver is present (the D2 "no Vulkan" probe).
[[nodiscard]] VkInstance shared_instance();

// One physical device with its capability verdict. `info.ray_tracing` is
// true only when the device passes the FULL requirement set of the roadmap
// (09 §1): Vulkan >= 1.3; extensions acceleration_structure + ray_query +
// deferred_host_operations; features shaderInt64, bufferDeviceAddress,
// scalarBlockLayout, timelineSemaphore, shaderBufferInt64Atomics,
// synchronization2, maintenance4, accelerationStructure, rayQuery.
struct PhysicalDeviceCheck {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    DeviceInfo info;
    bool has_memory_budget = false;
};

// Empty when no driver / no devices. Order matches enumerate_devices().
[[nodiscard]] std::vector<PhysicalDeviceCheck> enumerate_physical_devices();

// Logical device + queues + VMA allocator over one capable physical device.
class DeviceImpl {
  public:
    // `picked` must have info.ray_tracing == true.
    explicit DeviceImpl(const PhysicalDeviceCheck& picked);
    ~DeviceImpl();
    DeviceImpl(const DeviceImpl&) = delete;
    DeviceImpl& operator=(const DeviceImpl&) = delete;
    DeviceImpl(DeviceImpl&&) = delete;
    DeviceImpl& operator=(DeviceImpl&&) = delete;

    [[nodiscard]] std::uint64_t memory_budget() const;

    DeviceInfo info;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t compute_family = 0;
    VkQueue compute_queue = VK_NULL_HANDLE;
    // Dedicated transfer family (TRANSFER without GRAPHICS|COMPUTE) when the
    // hardware has one; otherwise transfer_queue == compute_queue.
    std::optional<std::uint32_t> transfer_family;
    VkQueue transfer_queue = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    bool has_memory_budget = false;
};

}  // namespace pycanha::radiative::detail
