#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh::ops {

namespace detail {

template <class Scalar>
[[nodiscard]] Vector3D triangle_normal_unnormalized(const TriMesh<Scalar>& mesh,
                                                    Index tri_idx) {
    const auto tri = mesh.triangles.row(tri_idx);
    const Vector3D p0 =
        mesh.vertices.row(static_cast<Index>(tri(0))).template cast<double>();
    const Vector3D p1 =
        mesh.vertices.row(static_cast<Index>(tri(1))).template cast<double>();
    const Vector3D p2 =
        mesh.vertices.row(static_cast<Index>(tri(2))).template cast<double>();
    return (p1 - p0).cross(p2 - p0);
}

}  // namespace detail

template <class Scalar>
[[nodiscard]] Eigen::VectorXd compute_areas(const TriMesh<Scalar>& mesh) {
    Eigen::VectorXd areas(mesh.triangles.rows());
    for (Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        areas[tri_idx] =
            0.5 * detail::triangle_normal_unnormalized(mesh, tri_idx).norm();
    }
    return areas;
}

template <class Scalar>
[[nodiscard]] Eigen::MatrixX3d compute_centroids(const TriMesh<Scalar>& mesh) {
    Eigen::MatrixX3d centroids(mesh.triangles.rows(), 3);
    for (Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const auto tri = mesh.triangles.row(tri_idx);
        centroids.row(tri_idx) = (mesh.vertices.row(static_cast<Index>(tri(0)))
                                      .template cast<double>() +
                                  mesh.vertices.row(static_cast<Index>(tri(1)))
                                      .template cast<double>() +
                                  mesh.vertices.row(static_cast<Index>(tri(2)))
                                      .template cast<double>()) /
                                 3.0;
    }
    return centroids;
}

template <class Scalar>
[[nodiscard]] Eigen::MatrixX3d compute_face_normals(
    const TriMesh<Scalar>& mesh) {
    Eigen::MatrixX3d normals(mesh.triangles.rows(), 3);
    for (Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        normals.row(tri_idx) =
            detail::triangle_normal_unnormalized(mesh, tri_idx).normalized();
    }
    return normals;
}

template <class Scalar>
[[nodiscard]] Eigen::AlignedBox3d bounding_box(const TriMesh<Scalar>& mesh) {
    Eigen::AlignedBox3d box;
    for (Index vertex_idx = 0; vertex_idx < mesh.vertices.rows();
         ++vertex_idx) {
        box.extend(
            mesh.vertices.row(vertex_idx).template cast<double>().transpose());
    }
    return box;
}

}  // namespace pycanha::gmm::mesh::ops
