#pragma once

// src-private: every Metal type stays below src/radiative so the public API
// exposes no Metal types (the Vulkan backend keeps the same rule with its own
// types). Public headers must never include this file.
//
// This header is Objective-C++: the sources that include it (mtl_*.mm plus
// the shared device.cpp / scene.cpp) are all compiled as OBJCXX with ARC on
// Apple, so the Metal objects below are ordinary strong references.

#import <Metal/Metal.h>

#include <cstdint>
#include <vector>

#include "pycanha-core/radiative/device.hpp"

namespace pycanha::radiative::detail {

// One Metal device with its capability verdict. `info.ray_tracing` is true
// only when the device can run the kernels: hardware ray tracing plus GPU
// family Apple9 (M3 / M4 / A17 Pro and newer), which is the agreed minimum.
// Older Apple Silicon takes the same graceful "no capable device" path as a
// machine without a ray-tracing driver.
struct PhysicalDeviceCheck {
    id<MTLDevice> device = nil;
    DeviceInfo info;
    // Preference when no device index is requested, highest wins. Ranking the
    // devices here rather than in the shared selection code keeps every
    // driver query inside the backend.
    int score = 0;
};

// Empty when the machine reports no Metal device at all. Order matches
// enumerate_devices().
[[nodiscard]] std::vector<PhysicalDeviceCheck> enumerate_physical_devices();

// Device + command queue over one capable Metal device. Metal has no queue
// families and no separate memory allocator: one queue serves compute,
// acceleration-structure builds and blits, and buffers come straight from the
// device.
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
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
};

}  // namespace pycanha::radiative::detail
