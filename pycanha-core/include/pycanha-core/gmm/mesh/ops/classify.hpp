#pragma once

#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::gmm::mesh::ops {

[[nodiscard]] FaceId classify_triangle_by_centroid(
    const TriMesh& mesh, Eigen::Index triangle_index, const Primitive& primitive,
    const ThermalMesh& thermal_mesh);

}  // namespace pycanha::gmm::mesh::ops