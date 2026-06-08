#pragma once

#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::gmm {

// Internal-only face side. Even FaceId == Front (side 1), odd == Back
// (side 2). This enum is intentionally NOT part of the ThermalMesh public
// surface; it lives here with the face_id_from_uv helper and is removed
// together with it.
enum class Side : unsigned char { Front = 0, Back = 1 };

}  // namespace pycanha::gmm

namespace pycanha::gmm::ops {

[[nodiscard]] FaceId face_id_from_uv(const Primitive& primitive,
                                     const ThermalMesh& thermal_mesh,
                                     const Point2D& uv,
                                     Side side = Side::Front);

}  // namespace pycanha::gmm::ops
