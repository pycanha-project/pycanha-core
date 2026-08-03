#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pycanha::radiative {

namespace detail {
class DeviceImpl;
}  // namespace detail

// One physical GPU as seen by the radiative engine (a Vulkan device, or a
// Metal device on macOS).
struct DeviceInfo {
    std::string name;
    // True when the device passes the full RT requirement set of its backend
    // (Vulkan: 1.3, acceleration structures, ray queries, 64-bit buffer
    // atomics, ...; Metal: hardware ray tracing on GPU family Apple9).
    bool ray_tracing = false;
    // Software implementation (lavapipe reports device type CPU). Always
    // false on Metal — it has no software rasterizer to fall back to.
    bool software = false;
    // Device-limit upper bound of rays per emitter in one dispatch.
    std::uint64_t max_dispatch_rays = 0;
    // Position in enumerate_devices(); the argument for Device::create().
    std::uint32_t index = 0;
};

// Cheap and safe on machines with no GPU driver at all: no driver means
// false / an empty list — never an exception. The rest of the library works
// without any GPU; only constructing radiative objects requires one.
[[nodiscard]] bool is_available();
[[nodiscard]] std::vector<DeviceInfo> enumerate_devices();

// Owns the backend's device and queues; one Device can serve many scenes.
// The ONLY place that touches driver discovery, so "no usable GPU" is a
// single, clear construction error. The public API exposes no backend types.
class Device {
  public:
    // Creates the device at `index` (position in enumerate_devices()), or —
    // when `index` is negative — the best RT-capable device (discrete >
    // integrated > software). Throws std::runtime_error with an actionable
    // message when no RT-capable device exists.
    [[nodiscard]] static Device create(std::int32_t index = -1);

    ~Device();
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] const DeviceInfo& info() const noexcept;

    // Device-local bytes currently available for new allocations: the
    // largest device heap minus what is in use where the driver reports it
    // (otherwise 80% of the heap size). On unified memory that heap is
    // system memory. Callers size accumulator buffers against this.
    [[nodiscard]] std::uint64_t memory_budget() const;

    // Engine-internal accessor (opaque outside the library's own sources).
    [[nodiscard]] detail::DeviceImpl& impl() const noexcept;

  private:
    explicit Device(std::unique_ptr<detail::DeviceImpl> impl);

    std::unique_ptr<detail::DeviceImpl> _impl;
};

}  // namespace pycanha::radiative
