#pragma once

#include <vector>

#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh::ops {

[[nodiscard]] std::vector<std::vector<Eigen::Index>> boundary_edge_loops(
    const TriMesh& mesh);

}  // namespace pycanha::gmm::mesh::ops
