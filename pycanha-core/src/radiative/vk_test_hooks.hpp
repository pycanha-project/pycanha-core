#pragma once

// Test-only access to Vulkan-backend state that has no public API. Free of
// Vulkan types so a test can include it without the backend's include paths.
// Defined in vk_device.cpp: the Metal backend and the stub build have none
// of this, so only a test built for the Vulkan backend may call it.

#include <cstdint>

#include "pycanha-core/radiative/device.hpp"

namespace pycanha::radiative::detail {

// Most emitting faces one kernel launch covers on `device`; longer emitter
// lists are traced in batches of this size.
[[nodiscard]] std::uint32_t max_dispatch_emitters(const Device& device);

// Lowers that limit, so the batching can be exercised on a GPU whose real
// limit no test scene comes near. Throws std::invalid_argument for 0 or for
// a value above the current limit: raising it past what the device reported
// would make the launches invalid.
void lower_max_dispatch_emitters(Device& device, std::uint32_t limit);

}  // namespace pycanha::radiative::detail
