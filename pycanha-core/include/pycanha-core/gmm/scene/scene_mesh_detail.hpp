#pragma once

#include <Eigen/Dense>

#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::gmm::detail {

// Applies `transform` to every vertex of `mesh` in place (full-precision).
inline void apply_transform_in_place(
    TriMeshD& mesh, const CoordinateTransformation& transform) {
    if (transform.is_identity()) {
        return;
    }
    for (Eigen::Index vertex_idx = 0; vertex_idx < mesh.vertices.rows();
         ++vertex_idx) {
        mesh.vertices.row(vertex_idx) =
            transform.apply(mesh.vertices.row(vertex_idx).transpose())
                .transpose();
    }
}

}  // namespace pycanha::gmm::detail
