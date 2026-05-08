#pragma once

#include <Eigen/Dense>

#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/mesh/unified_trimesh.hpp"

namespace pycanha::gmm::mesh::ops {

[[nodiscard]] Eigen::VectorXi permutation_by_face_id(const TriMesh& mesh);
void apply_permutation(TriMesh& mesh, const Eigen::VectorXi& permutation);
void apply_permutation(UnifiedTriMesh& mesh,
                       const Eigen::VectorXi& permutation);

}  // namespace pycanha::gmm::mesh::ops
