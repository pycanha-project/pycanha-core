#pragma once

#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::gmm::ops {

[[nodiscard]] Primitive transform(
    const Primitive& primitive, const CoordinateTransformation& transformation);

}  // namespace pycanha::gmm::ops
