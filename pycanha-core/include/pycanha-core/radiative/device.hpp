#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pycanha::radiative {

namespace detail {
class DeviceImpl;
}  // namespace detail

// One physical Vulkan device as seen by the radiative engine.
struct DeviceInfo {
    std::string name;
    // True when the device passes the full RT requirement set (Vulkan 1.3,
    // acceleration structures, ray queries, 64-bit buffer atomics, ...).
    bool ray_tracing = false;
    // Software implementation (lavapipe reports device type CPU).
    bool software = false;
    // Device-limit upper bound of rays per emitter in one dispatch.
    std::uint64_t max_dispatch_rays = 0;
    // Position in enumerate_devices(); the argument for Device::create().
    std::uint32_t index = 0;
};

// Cheap and safe on machines with no Vulkan driver at all: no driver means
// false / an empty list — never an exception (D2).
[[nodiscard]] bool is_available();
[[nodiscard]] std::vector<DeviceInfo> enumerate_devices();

// Owns the Vulkan instance/device/queues/allocator; one Device can serve
// many scenes. The ONLY place that touches driver discovery, so "no Vulkan"
// is a single, clear construction error. Public API exposes no Vulkan types.
class Device {
  public:
    // Creates the device at `index` (position in enumerate_devices()), or —
    // when `index` is negative — the best RT-capable device (discrete >
    // integrated > software). Throws std::runtime_error with an actionable
    // message when no RT-capable device (hardware or lavapipe) exists.
    [[nodiscard]] static Device create(std::int32_t index = -1);

    ~Device();
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] const DeviceInfo& info() const noexcept;

    // DEVICE_LOCAL bytes currently available for new allocations on the
    // largest device heap (VK_EXT_memory_budget when present, otherwise 80%
    // of the heap size). Input to the accumulator memory policy (D29).
    [[nodiscard]] std::uint64_t memory_budget() const;

    // Engine-internal accessor (opaque outside the library's own sources).
    [[nodiscard]] detail::DeviceImpl& impl() const noexcept;

  private:
    explicit Device(std::unique_ptr<detail::DeviceImpl> impl);

    std::unique_ptr<detail::DeviceImpl> _impl;
};

}  // namespace pycanha::radiative
