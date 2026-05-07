#pragma once

#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::gmm::ops {

[[nodiscard]] FaceId face_id_from_uv(const Primitive& primitive,
                                     const ThermalMesh& thermal_mesh,
                                     const Point2D& uv,
                                     Side side = Side::Front);

}  // namespace pycanha::gmm::ops
