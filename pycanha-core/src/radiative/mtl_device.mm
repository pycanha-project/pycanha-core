#include "mtl_device.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <spdlog/spdlog.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "pycanha-core/utils/logger.hpp"

namespace pycanha::radiative::detail {

namespace {

[[nodiscard]] PhysicalDeviceCheck check_device(id<MTLDevice> device,
                                               std::uint32_t index) {
    PhysicalDeviceCheck check;
    check.device = device;
    check.info.name = std::string([[device name] UTF8String]);
    check.info.index = index;
    // Metal has no software rasterizer to fall back to (there is no lavapipe
    // equivalent): every device reported here is real hardware, or the
    // paravirtual device of a VM, which fails the capability check below.
    check.info.software = false;
    // Metal publishes no threadgroup-count limit the way Vulkan does. The
    // real bound is the u32 ray index carried in the push constants, and
    // dispatches are chunked far below it in any case.
    check.info.max_dispatch_rays = std::numeric_limits<std::uint32_t>::max();
    // Apple9 (M3 / M4 / A17 Pro) is the agreed minimum: it is the family
    // where the ray-tracing intrinsics the kernels use are available.
    check.info.ray_tracing =
        [device supportsRaytracing] &&
        [device supportsFamily:MTLGPUFamilyApple9];
    // Apple Silicon exposes exactly one GPU, but rank anyway so that a
    // machine reporting several picks deterministically.
    check.score = [device location] == MTLDeviceLocationBuiltIn ? 2 : 1;
    return check;
}

}  // namespace

std::vector<PhysicalDeviceCheck> enumerate_physical_devices() {
    std::vector<PhysicalDeviceCheck> checks;
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        checks.reserve([devices count]);
        std::uint32_t index = 0;
        for (id<MTLDevice> device in devices) {
            checks.push_back(check_device(device, index));
            ++index;
        }
    }
    return checks;
}

DeviceImpl::DeviceImpl(const PhysicalDeviceCheck& picked)
    : info(picked.info), device(picked.device) {
    queue = [device newCommandQueue];
    if (queue == nil) {
        throw std::runtime_error(
            "pycanha::radiative: Metal command queue creation failed for '" +
            info.name + "'");
    }
    SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                       "radiative: created device '{}' (unified memory={})",
                       info.name, static_cast<bool>([device hasUnifiedMemory]));
}

DeviceImpl::~DeviceImpl() = default;

std::uint64_t DeviceImpl::memory_budget() const {
    const auto recommended =
        static_cast<std::uint64_t>([device recommendedMaxWorkingSetSize]);
    if (recommended == 0) {
        // No working-set hint: fall back to 80 % of the machine's memory,
        // the same shape as the Vulkan path's heap-size fallback. Apple
        // Silicon is unified memory, so system memory IS the GPU heap.
        std::uint64_t physical = 0;
        @autoreleasepool {
            physical = static_cast<std::uint64_t>(
                [[NSProcessInfo processInfo] physicalMemory]);
        }
        return physical * 8 / 10;
    }
    const auto allocated =
        static_cast<std::uint64_t>([device currentAllocatedSize]);
    return recommended > allocated ? recommended - allocated : 0;
}

}  // namespace pycanha::radiative::detail
