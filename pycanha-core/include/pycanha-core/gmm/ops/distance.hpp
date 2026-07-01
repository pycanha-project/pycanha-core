#pragma once

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::gmm::ops {

[[nodiscard]] double distance(const Primitive& primitive, const Point3D& point);

}  // namespace pycanha::gmm::ops
