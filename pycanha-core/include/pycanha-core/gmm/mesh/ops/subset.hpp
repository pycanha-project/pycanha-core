#pragma once

#include <span>

#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh::ops {

[[nodiscard]] TriMesh extract_by_face_id(const TriMesh& mesh,
                                         std::span<const FaceId> face_ids);

}  // namespace pycanha::gmm::mesh::ops