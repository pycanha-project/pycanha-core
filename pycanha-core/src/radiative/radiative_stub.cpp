// Built when PYCANHA_OPTION_RAYTRACING is OFF: the whole radiative module is
// excluded, but the availability probe and the Device symbols still exist so
// downstream code links against one stable surface on every platform.

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/scene.hpp"

namespace pycanha::radiative {

namespace {
[[noreturn]] void throw_unavailable() {
    throw std::runtime_error(
        "pycanha-core was built without raytracing support "
        "(PYCANHA_OPTION_RAYTRACING=OFF)");
}
}  // namespace

namespace detail {
// Never instantiated in a stub build; complete types are still needed for
// the unique_ptr members of the public classes.
class DeviceImpl {};
class SceneImpl {};
class VfAccumImpl {};
}  // namespace detail

bool is_available() { return false; }

std::vector<DeviceInfo> enumerate_devices() { return {}; }

Device Device::create(std::int32_t /*index*/) { throw_unavailable(); }

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

RadiativeScene::RadiativeScene(Device& /*device*/,
                               std::vector<ScenePart> /*parts*/,
                               MaterialTable /*materials*/) {
    throw_unavailable();
}
RadiativeScene::~RadiativeScene() = default;
RadiativeScene::RadiativeScene(RadiativeScene&&) noexcept = default;
RadiativeScene& RadiativeScene::operator=(RadiativeScene&&) noexcept = default;
void RadiativeScene::set_part_transform(std::uint32_t /*part_id*/,
                                        const gmm::CoordinateTransformation&
                                        /*world_tf*/) {
    throw_unavailable();
}
void RadiativeScene::commit() { throw_unavailable(); }
void RadiativeScene::accumulate_vf(VfAccumulator& /*acc*/,
                                   const TraceSettings& /*settings*/,
                                   std::span<const std::uint32_t>
                                   /*emitters*/) {
    throw_unavailable();
}
std::uint32_t RadiativeScene::num_face_slots() const noexcept { return 0; }
const MaterialTable& RadiativeScene::materials() const noexcept {
    static const MaterialTable none;
    return none;
}
std::span<const double> RadiativeScene::face_areas() const noexcept {
    return {};
}
detail::SceneImpl& RadiativeScene::impl() const noexcept { return *_impl; }

VfAccumulator::VfAccumulator(const RadiativeScene& /*scene*/,
                             AccumConfig /*config*/) {
    throw_unavailable();
}
VfAccumulator::~VfAccumulator() = default;
VfAccumulator::VfAccumulator(VfAccumulator&&) noexcept = default;
VfAccumulator& VfAccumulator::operator=(VfAccumulator&&) noexcept = default;
void VfAccumulator::reset() { throw_unavailable(); }
VfResult VfAccumulator::result() const { throw_unavailable(); }
detail::VfAccumImpl& VfAccumulator::impl() noexcept { return *_impl; }

}  // namespace pycanha::radiative
