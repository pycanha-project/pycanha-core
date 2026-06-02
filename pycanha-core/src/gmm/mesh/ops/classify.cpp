#include "pycanha-core/gmm/mesh/ops/classify.hpp"

#include <variant>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/ops/face_id_from_uv.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::gmm::mesh::ops {

FaceId classify_triangle_by_centroid(const TriMesh& mesh,
                                     Eigen::Index triangle_index,
                                     const Primitive& primitive,
                                     const ThermalMesh& thermal_mesh) {
    const Eigen::Vector3i triangle = mesh.triangles.row(triangle_index);
    const Point3D p0 = mesh.vertices.row(triangle[0]).transpose();
    const Point3D p1 = mesh.vertices.row(triangle[1]).transpose();
    const Point3D p2 = mesh.vertices.row(triangle[2]).transpose();
    const Point3D centroid = (p0 + p1 + p2) / 3.0;

    const Point2D uv = std::visit(
        [&centroid](const auto& concrete_primitive) {
            return concrete_primitive.to_uv(centroid);
        },
        primitive);
    Vector3D triangle_normal = (p1 - p0).cross(p2 - p0);
    if (triangle_normal.norm() <= LENGTH_TOL) {
        triangle_normal = std::visit(
            [&uv](const auto& concrete_primitive) {
                return concrete_primitive.normal_at_uv(uv);
            },
            primitive);
    } else {
        triangle_normal.normalize();
    }

    const Vector3D primitive_normal = std::visit(
        [&uv](const auto& concrete_primitive) {
            return concrete_primitive.normal_at_uv(uv);
        },
        primitive);
    const Side side =
        triangle_normal.dot(primitive_normal) >= 0.0 ? Side::Front : Side::Back;
    return pycanha::gmm::ops::face_id_from_uv(primitive, thermal_mesh, uv,
                                              side);
}

}  // namespace pycanha::gmm::mesh::ops
