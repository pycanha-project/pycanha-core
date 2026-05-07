#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh::ops {
namespace {

[[nodiscard]] Vector3D triangle_normal_unnormalized(const TriMesh& mesh,
                                                    Index tri_idx) {
    const Eigen::Vector3i tri = mesh.triangles.row(tri_idx);
    const Vector3D p0 = mesh.vertices.row(tri.x());
    const Vector3D p1 = mesh.vertices.row(tri.y());
    const Vector3D p2 = mesh.vertices.row(tri.z());
    return (p1 - p0).cross(p2 - p0);
}

}  // namespace

Eigen::VectorXd compute_areas(const TriMesh& mesh) {
    Eigen::VectorXd areas(mesh.triangles.rows());
    for (Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        areas[tri_idx] =
            0.5 * triangle_normal_unnormalized(mesh, tri_idx).norm();
    }
    return areas;
}

Eigen::MatrixX3d compute_centroids(const TriMesh& mesh) {
    Eigen::MatrixX3d centroids(mesh.triangles.rows(), 3);
    for (Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const Eigen::Vector3i tri = mesh.triangles.row(tri_idx);
        centroids.row(tri_idx) =
            (mesh.vertices.row(tri.x()) + mesh.vertices.row(tri.y()) +
             mesh.vertices.row(tri.z())) /
            3.0;
    }
    return centroids;
}

Eigen::MatrixX3d compute_face_normals(const TriMesh& mesh) {
    Eigen::MatrixX3d normals(mesh.triangles.rows(), 3);
    for (Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        normals.row(tri_idx) =
            triangle_normal_unnormalized(mesh, tri_idx).normalized();
    }
    return normals;
}

Eigen::AlignedBox3d bounding_box(const TriMesh& mesh) {
    Eigen::AlignedBox3d box;
    for (Index vertex_idx = 0; vertex_idx < mesh.vertices.rows();
         ++vertex_idx) {
        box.extend(mesh.vertices.row(vertex_idx).transpose());
    }
    return box;
}

}  // namespace pycanha::gmm::mesh::ops
