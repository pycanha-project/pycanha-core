#pragma once

#include <manifold/manifold.h>

#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::gmm::cutting {

[[nodiscard]] manifold::Manifold build_cutter(
    const Primitive& cutter,
    const CoordinateTransformation& world_transform = {});

}  // namespace pycanha::gmm::cutting
