// Public RadiativeScene / VfAccumulator API over the src-private vk_scene
// and vk_accum layers.

#include "pycanha-core/radiative/scene.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene_part.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vk_accum.hpp"
#include "vk_scene.hpp"

namespace pycanha::radiative {

RadiativeScene::RadiativeScene(Device& device, std::vector<ScenePart> parts,
                               MaterialTable materials)
    : _impl(std::make_unique<detail::SceneImpl>(device.impl(), std::move(parts),
                                                std::move(materials))) {}

RadiativeScene::~RadiativeScene() = default;
RadiativeScene::RadiativeScene(RadiativeScene&&) noexcept = default;
RadiativeScene& RadiativeScene::operator=(RadiativeScene&&) noexcept = default;

void RadiativeScene::set_part_transform(
    std::uint32_t part_id, const gmm::CoordinateTransformation& world_tf) {
    _impl->set_part_transform(part_id, world_tf);
}

void RadiativeScene::commit() { _impl->commit(); }

void RadiativeScene::accumulate_vf(VfAccumulator& acc,
                                   const TraceSettings& settings,
                                   std::span<const std::uint32_t> emitters) {
    _impl->accumulate_vf(acc.impl(), settings, emitters);
}

std::uint32_t RadiativeScene::num_face_slots() const noexcept {
    return _impl->num_face_slots();
}

const MaterialTable& RadiativeScene::materials() const noexcept {
    return _impl->materials();
}

std::span<const double> RadiativeScene::face_areas() const noexcept {
    return _impl->face_areas();
}

detail::SceneImpl& RadiativeScene::impl() const noexcept { return *_impl; }

VfAccumulator::VfAccumulator(const RadiativeScene& scene, AccumConfig config)
    : _impl(std::make_unique<detail::VfAccumImpl>(scene.impl(), config)) {}

VfAccumulator::~VfAccumulator() = default;
VfAccumulator::VfAccumulator(VfAccumulator&&) noexcept = default;
VfAccumulator& VfAccumulator::operator=(VfAccumulator&&) noexcept = default;

void VfAccumulator::reset() { _impl->reset(); }

VfResult VfAccumulator::result() const { return _impl->build_result(); }

detail::VfAccumImpl& VfAccumulator::impl() noexcept { return *_impl; }

}  // namespace pycanha::radiative
