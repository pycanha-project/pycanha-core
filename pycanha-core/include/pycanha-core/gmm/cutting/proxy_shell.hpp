#pragma once

#include <cstdint>

#include <manifold/manifold.h>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::cutting {

struct ProxyMeta {
    GeometryId source_item_id;
    double thickness;
    std::uint32_t outer_original_id;
};

[[nodiscard]] manifold::Manifold build_primitive_proxy(
    const TriMesh& triangulated_primitive, const ProxyMeta& meta);

}  // namespace pycanha::gmm::cutting