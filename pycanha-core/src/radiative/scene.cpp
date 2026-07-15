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

void RadiativeScene::accumulate_exchange(
    ExchangeAccumulator& acc, const TraceSettings& settings,
    std::span<const std::uint32_t> emitters) {
    _impl->accumulate_exchange(acc.impl(), settings, emitters);
}

void RadiativeScene::accumulate_solar(const SolarState& sun,
                                      SolarAccumulator& acc,
                                      const TraceSettings& settings) {
    _impl->accumulate_solar(sun, acc.impl(), settings);
}

void RadiativeScene::update_materials(const MaterialTable& materials) {
    _impl->update_materials(materials);
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

ExchangeAccumulator::ExchangeAccumulator(const RadiativeScene& scene, Band band,
                                         AccumConfig config)
    : _impl(std::make_unique<detail::ExchangeAccumImpl>(scene.impl(), band,
                                                        config)) {}

ExchangeAccumulator::~ExchangeAccumulator() = default;
ExchangeAccumulator::ExchangeAccumulator(ExchangeAccumulator&&) noexcept =
    default;
ExchangeAccumulator& ExchangeAccumulator::operator=(
    ExchangeAccumulator&&) noexcept = default;

void ExchangeAccumulator::reset() { _impl->reset(); }

ExchangeResult ExchangeAccumulator::result() const {
    return _impl->build_result();
}

std::uint64_t ExchangeAccumulator::conservation_error() const {
    return _impl->conservation_error();
}

detail::ExchangeAccumImpl& ExchangeAccumulator::impl() noexcept {
    return *_impl;
}

SolarAccumulator::SolarAccumulator(const RadiativeScene& scene)
    : _impl(std::make_unique<detail::SolarAccumImpl>(scene.impl())) {}

SolarAccumulator::~SolarAccumulator() = default;
SolarAccumulator::SolarAccumulator(SolarAccumulator&&) noexcept = default;
SolarAccumulator& SolarAccumulator::operator=(SolarAccumulator&&) noexcept =
    default;

void SolarAccumulator::reset() { _impl->reset(); }

SolarResult SolarAccumulator::result() const { return _impl->build_result(); }

detail::SolarAccumImpl& SolarAccumulator::impl() noexcept { return *_impl; }

}  // namespace pycanha::radiative
