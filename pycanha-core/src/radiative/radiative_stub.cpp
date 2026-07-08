// Built when PYCANHA_OPTION_RAYTRACING is OFF: the whole radiative module is
// excluded, but the availability probe and the Device symbols still exist so
// downstream code links against one stable surface (D2).

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/radiative/device.hpp"

namespace pycanha::radiative {

namespace detail {
class DeviceImpl {};  // never instantiated in a stub build
}  // namespace detail

bool is_available() { return false; }

std::vector<DeviceInfo> enumerate_devices() { return {}; }

Device Device::create(std::int32_t /*index*/) {
    throw std::runtime_error(
        "pycanha-core was built without raytracing support "
        "(PYCANHA_OPTION_RAYTRACING=OFF)");
}

Device::Device(std::unique_ptr<detail::DeviceImpl> impl)
    : _impl(std::move(impl)) {}

Device::~Device() = default;
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;

const DeviceInfo& Device::info() const noexcept {
    static const DeviceInfo none;
    return none;
}

std::uint64_t Device::memory_budget() const { return 0; }

detail::DeviceImpl& Device::impl() const noexcept { return *_impl; }

}  // namespace pycanha::radiative
