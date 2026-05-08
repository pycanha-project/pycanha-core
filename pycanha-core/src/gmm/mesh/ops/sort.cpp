#include "pycanha-core/gmm/mesh/ops/sort.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/mesh/unified_trimesh.hpp"

namespace pycanha::gmm::mesh::ops {

Eigen::VectorXi permutation_by_face_id(const TriMesh& mesh) {
    std::vector<int> indices(static_cast<std::size_t>(mesh.triangles.rows()));
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(indices.begin(), indices.end(), [&mesh](int lhs, int rhs) {
        return mesh.face_ids(lhs) < mesh.face_ids(rhs);
    });

    Eigen::VectorXi permutation(mesh.triangles.rows());
    for (Index tri_idx = 0; tri_idx < permutation.size(); ++tri_idx) {
        permutation(tri_idx) = indices[static_cast<std::size_t>(tri_idx)];
    }
    return permutation;
}

void apply_permutation(TriMesh& mesh, const Eigen::VectorXi& permutation) {
    PYCANHA_ASSERT(permutation.size() == mesh.triangles.rows(),
                   "Triangle permutation size mismatch");
    PYCANHA_ASSERT(mesh.face_ids.rows() == mesh.triangles.rows(),
                   "Each triangle must have one face id");

    Eigen::MatrixX3i triangles(mesh.triangles.rows(), 3);
    FaceIdVector face_ids(mesh.face_ids.rows());
    for (Index tri_idx = 0; tri_idx < permutation.size(); ++tri_idx) {
        const Index source_idx = permutation(tri_idx);
        triangles.row(tri_idx) = mesh.triangles.row(source_idx);
        face_ids(tri_idx) = mesh.face_ids(source_idx);
    }

    mesh.triangles = std::move(triangles);
    mesh.face_ids = std::move(face_ids);
}

void apply_permutation(UnifiedTriMesh& mesh,
                       const Eigen::VectorXi& permutation) {
    PYCANHA_ASSERT(mesh.geometry_ids.rows() == mesh.triangles.rows(),
                   "Each triangle must have one geometry id");

    Eigen::Matrix<std::uint64_t, Eigen::Dynamic, 1> geometry_ids(
        mesh.geometry_ids.rows());
    for (Index tri_idx = 0; tri_idx < permutation.size(); ++tri_idx) {
        geometry_ids(tri_idx) = mesh.geometry_ids(permutation(tri_idx));
    }

    apply_permutation(static_cast<TriMesh&>(mesh), permutation);
    mesh.geometry_ids = std::move(geometry_ids);
}

}  // namespace pycanha::gmm::mesh::ops
