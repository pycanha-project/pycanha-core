#pragma once

#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh::ops {

[[nodiscard]] bool is_watertight(const TriMesh& mesh);
[[nodiscard]] bool is_manifold(const TriMesh& mesh);
[[nodiscard]] bool is_watertight_on_curved_edges(const TriMesh& mesh);
[[nodiscard]] bool has_consistent_face_ids(const TriMesh& mesh);

}  // namespace pycanha::gmm::mesh::ops
