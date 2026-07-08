// Public Device / availability API over the src-private vk_device layer.

#include "pycanha-core/radiative/device.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vk_device.hpp"

namespace pycanha::radiative {

namespace {

using detail::PhysicalDeviceCheck;

// Selection order for the default pick: discrete > integrated > other
// hardware > software (a software rasterizer like lavapipe is a valid last
// resort — same SPIR-V, just slow).
[[nodiscard]] int selection_score(const PhysicalDeviceCheck& check) {
    if (check.info.software) {
        return 0;
    }
    VkPhysicalDeviceProperties2 props2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = nullptr,
        .properties = {}};
    vkGetPhysicalDeviceProperties2(check.physical_device, &props2);
    switch (props2.properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return 3;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return 2;
        default:
            return 1;
    }
}

}  // namespace

bool is_available() {
    const auto checks = detail::enumerate_physical_devices();
    return std::ranges::any_of(checks, [](const PhysicalDeviceCheck& check) {
        return check.info.ray_tracing;
    });
}

std::vector<DeviceInfo> enumerate_devices() {
    const auto checks = detail::enumerate_physical_devices();
    std::vector<DeviceInfo> infos;
    infos.reserve(checks.size());
    std::ranges::transform(
        checks, std::back_inserter(infos),
        [](const PhysicalDeviceCheck& check) { return check.info; });
    return infos;
}

Device Device::create(std::int32_t index) {
    const auto checks = detail::enumerate_physical_devices();
    if (checks.empty()) {
        throw std::runtime_error(
            "pycanha::radiative: no Vulkan driver found. Install a GPU "
            "driver with Vulkan support, or Mesa lavapipe for a software "
            "fallback.");
    }

    const PhysicalDeviceCheck* picked = nullptr;
    if (index >= 0) {
        const auto unsigned_index = static_cast<std::size_t>(index);
        if (unsigned_index >= checks.size()) {
            throw std::runtime_error("pycanha::radiative: device index " +
                                     std::to_string(index) + " out of range (" +
                                     std::to_string(checks.size()) +
                                     " Vulkan devices found)");
        }
        picked = &checks[unsigned_index];
        if (!picked->info.ray_tracing) {
            throw std::runtime_error(
                "pycanha::radiative: device '" + picked->info.name +
                "' does not support the required ray-tracing feature set");
        }
    } else {
        int best_score = -1;
        for (const PhysicalDeviceCheck& check : checks) {
            if (!check.info.ray_tracing) {
                continue;
            }
            const int score = selection_score(check);
            if (score > best_score) {
                best_score = score;
                picked = &check;
            }
        }
        if (picked == nullptr) {
            throw std::runtime_error(
                "pycanha::radiative: no ray-tracing-capable Vulkan device "
                "found (need acceleration structures, ray queries and "
                "64-bit atomics). Update the GPU driver or install Mesa "
                "lavapipe >= 26.");
        }
    }

    return Device(std::make_unique<detail::DeviceImpl>(*picked));
}

Device::Device(std::unique_ptr<detail::DeviceImpl> impl)
    : _impl(std::move(impl)) {}

Device::~Device() = default;
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;

const DeviceInfo& Device::info() const noexcept { return _impl->info; }

std::uint64_t Device::memory_budget() const { return _impl->memory_budget(); }

detail::DeviceImpl& Device::impl() const noexcept { return *_impl; }

}  // namespace pycanha::radiative
