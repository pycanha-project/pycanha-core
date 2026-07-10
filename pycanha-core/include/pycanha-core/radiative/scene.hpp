#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/scene_part.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative {

class Device;
class VfAccumulator;

namespace detail {
class SceneImpl;
}  // namespace detail

// The stateful heart of the engine ("build once, compute many"):
// construction uploads geometry + materials, builds one bottom-level
// acceleration structure per rigid part and the initial top-level structure,
// and creates the compute pipelines. Tracing is batch-additive: each
// accumulate_* call adds settings.rays_per_face rays per emitting face into
// the given accumulator; repeat with increasing seeds to refine.
class RadiativeScene {
  public:
    // `device` must outlive the scene. Parts keep their global face ids, so
    // every matrix this scene produces is indexed by the model's face-slot
    // space. Throws std::invalid_argument on empty/inconsistent inputs.
    RadiativeScene(Device& device, std::vector<ScenePart> parts,
                   MaterialTable materials);
    ~RadiativeScene();
    RadiativeScene(RadiativeScene&&) noexcept;
    RadiativeScene& operator=(RadiativeScene&&) noexcept;
    RadiativeScene(const RadiativeScene&) = delete;
    RadiativeScene& operator=(const RadiativeScene&) = delete;

    // Articulation/motion: replace one part's placement, then commit() to
    // rebuild the instance structure (geometry stays untouched — moving a
    // part costs microseconds, not a scene rebuild).
    void set_part_transform(std::uint32_t part_id,
                            const gmm::CoordinateTransformation& world_tf);
    void commit();

    // Traces settings.rays_per_face rays per emitting face and ADDS the
    // first-hit counts into `acc`. `emitters` empty => all active
    // (non-planet) faces emit; otherwise only the listed face slots.
    void accumulate_vf(VfAccumulator& acc, const TraceSettings& settings,
                       std::span<const std::uint32_t> emitters = {});

    // Total face slots (rows/cols of every result matrix).
    [[nodiscard]] std::uint32_t num_face_slots() const noexcept;
    [[nodiscard]] const MaterialTable& materials() const noexcept;
    // Per-slot areas in the part-local mesh (sides share the pair area).
    [[nodiscard]] std::span<const double> face_areas() const noexcept;

    [[nodiscard]] detail::SceneImpl& impl() const noexcept;

  private:
    std::unique_ptr<detail::SceneImpl> _impl;
};

}  // namespace pycanha::radiative
