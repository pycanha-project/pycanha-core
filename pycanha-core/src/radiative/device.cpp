// Public Device / availability API over the src-private backend device layer
// (Vulkan everywhere except macOS, where it is Metal). Everything here is
// backend-independent: the backend enumerates the devices, ranks them and
// reports whether each one can run the kernels.

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

#ifdef __APPLE__
#include "mtl_device.hpp"
#else
#include "vk_device.hpp"
#endif

namespace pycanha::radiative {

namespace {

using detail::PhysicalDeviceCheck;

// Wording of the "cannot run here" errors: the driver stack, the requirements
// and the way out are all backend-specific.
#ifdef __APPLE__
constexpr const char* no_devices_message =
    "pycanha::radiative: no Metal device found.";
constexpr const char* device_api_name = "Metal";
constexpr const char* no_capable_device_message =
    "pycanha::radiative: no ray-tracing-capable Metal device found (need an "
    "Apple9 GPU — M3 or newer — with ray-tracing support).";
#else
constexpr const char* no_devices_message =
    "pycanha::radiative: no Vulkan driver found. Install a GPU driver with "
    "Vulkan support, or Mesa lavapipe for a software fallback.";
constexpr const char* device_api_name = "Vulkan";
constexpr const char* no_capable_device_message =
    "pycanha::radiative: no ray-tracing-capable Vulkan device found (need "
    "acceleration structures, ray queries and 64-bit atomics). Update the GPU "
    "driver or install Mesa lavapipe >= 26.";
#endif

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
        throw std::runtime_error(no_devices_message);
    }

    const PhysicalDeviceCheck* picked = nullptr;
    if (index >= 0) {
        const auto unsigned_index = static_cast<std::size_t>(index);
        if (unsigned_index >= checks.size()) {
            throw std::runtime_error("pycanha::radiative: device index " +
                                     std::to_string(index) + " out of range (" +
                                     std::to_string(checks.size()) + " " +
                                     device_api_name + " devices found)");
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
            if (check.score > best_score) {
                best_score = check.score;
                picked = &check;
            }
        }
        if (picked == nullptr) {
            throw std::runtime_error(no_capable_device_message);
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
