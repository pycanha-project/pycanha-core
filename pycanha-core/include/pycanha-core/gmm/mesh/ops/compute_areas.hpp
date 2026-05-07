#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh::ops {

[[nodiscard]] Eigen::VectorXd compute_areas(const TriMesh& mesh);
[[nodiscard]] Eigen::MatrixX3d compute_centroids(const TriMesh& mesh);
[[nodiscard]] Eigen::MatrixX3d compute_face_normals(const TriMesh& mesh);
[[nodiscard]] Eigen::AlignedBox3d bounding_box(const TriMesh& mesh);

}  // namespace pycanha::gmm::mesh::ops
