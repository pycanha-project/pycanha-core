#pragma once

#include <manifold/manifold.h>

#include <cstdint>

#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::cutting {

struct ProxyMeta {
    GeometryId source_item_id;
    double thickness;
    std::uint32_t outer_original_id;
};

[[nodiscard]] manifold::Manifold build_primitive_proxy(
    const TriMeshD& triangulated_primitive, const ProxyMeta& meta);

}  // namespace pycanha::gmm::cutting
