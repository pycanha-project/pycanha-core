#pragma once

#include <cstdint>

#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::radiative {

// Role of a part inside the raytraced scene. The GMM split defaults to
// Spacecraft for the remainder part and Articulated for named split groups;
// callers may reassign freely (plain data). CelestialBody parts (planet /
// moon spheres) are appended by the orchestration layer, never by the GMM.
enum class PartKind : std::uint8_t { Spacecraft, Articulated, CelestialBody };

// A rigid piece of the scene: one BLAS in the raytracer, one TLAS instance.
// Face identity is GLOBAL: `mesh.face_ids` keep the face ids of the model's
// unified mesh (GeometryModel::mesh()), so every part indexes the same
// face-slot space (nf = max_face_id + 2, even/odd = side 1/2) and result
// matrices need no per-part remapping.
struct ScenePart {
    // Part geometry in the part-local frame, float32 (the GMM RT mesh type).
    gmm::TriMeshF mesh;
    // Part -> world (initial placement; articulation replaces it per
    // snapshot via RadiativeScene::set_part_transform).
    gmm::CoordinateTransformation transform;
    PartKind kind = PartKind::Spacecraft;
    // Caller-chosen identifier echoed in results and instancing.
    std::uint32_t part_id = 0;
};

}  // namespace pycanha::radiative
