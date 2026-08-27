#include "pycanha-core/gmm/cutting/cutter_proxy.hpp"

#include <manifold/common.h>
#include <manifold/manifold.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ops/transform.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangular_prism.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::gmm::cutting {
namespace {

[[nodiscard]] manifold::vec3 to_manifold_vec(const Vector3D& vector) {
    return {vector.x(), vector.y(), vector.z()};
}

[[nodiscard]] manifold::mat3 to_manifold_rotation(
    const Eigen::Matrix3d& rotation) {
    return {manifold::vec3(rotation(0, 0), rotation(1, 0), rotation(2, 0)),
            manifold::vec3(rotation(0, 1), rotation(1, 1), rotation(2, 1)),
            manifold::vec3(rotation(0, 2), rotation(1, 2), rotation(2, 2))};
}

[[nodiscard]] manifold::mat3x4 to_manifold_transform(
    const Eigen::Matrix3d& rotation, const Point3D& translation) {
    return {to_manifold_rotation(rotation),
            manifold::vec3(translation.x(), translation.y(), translation.z())};
}

[[nodiscard]] bool spans_full_turn(double start_angle, double end_angle) {
    return std::abs((end_angle - start_angle) - (2.0 * std::numbers::pi)) <=
           1.0e-6;
}

[[nodiscard]] Eigen::Matrix3d frame_from_axis_and_reference(const Point3D& p1,
                                                            const Point3D& p2,
                                                            const Point3D& p3) {
    const Vector3D axis = (p2 - p1).normalized();
    Vector3D radial = p3 - p1;
    radial -= radial.dot(axis) * axis;
    if (radial.norm() <= LENGTH_TOL) {
        radial = axis.unitOrthogonal();
    }
    radial.normalize();
    const Vector3D tangent = axis.cross(radial).normalized();

    Eigen::Matrix3d rotation;
    rotation.col(0) = radial;
    rotation.col(1) = tangent;
    rotation.col(2) = axis;
    return rotation;
}

[[nodiscard]] manifold::Manifold build_cylinder(const Cylinder& cylinder) {
    if (!spans_full_turn(cylinder.start_angle(), cylinder.end_angle())) {
        throw std::logic_error("Cylinder cutters must be closed solids");
    }

    const double height = (cylinder.p2() - cylinder.p1()).norm();
    const Eigen::Matrix3d rotation = frame_from_axis_and_reference(
        cylinder.p1(), cylinder.p2(), cylinder.p3());
    const Point3D center = 0.5 * (cylinder.p1() + cylinder.p2());
    return manifold::Manifold::Cylinder(height, cylinder.radius(),
                                        cylinder.radius(), 64,
                                        /*center=*/true)
        .Transform(to_manifold_transform(rotation, center));
}

[[nodiscard]] manifold::Manifold build_cone(const Cone& cone) {
    if (!spans_full_turn(cone.start_angle(), cone.end_angle())) {
        throw std::logic_error("Cone cutters must be closed solids");
    }

    const double height = (cone.p2() - cone.p1()).norm();
    const Eigen::Matrix3d rotation =
        frame_from_axis_and_reference(cone.p1(), cone.p2(), cone.p3());
    const Point3D center = 0.5 * (cone.p1() + cone.p2());
    return manifold::Manifold::Cylinder(height, cone.radius1(), cone.radius2(),
                                        64, /*center=*/true)
        .Transform(to_manifold_transform(rotation, center));
}

[[nodiscard]] manifold::Manifold build_sphere(const Sphere& sphere) {
    if (!spans_full_turn(sphere.start_angle(), sphere.end_angle()) ||
        (std::abs(sphere.base_truncation() + sphere.radius()) > 1.0e-6) ||
        (std::abs(sphere.apex_truncation() - sphere.radius()) > 1.0e-6)) {
        throw std::logic_error("Sphere cutters must be closed solids");
    }

    return manifold::Manifold::Sphere(sphere.radius(), 96)
        .Translate(to_manifold_vec(sphere.p1()));
}

[[nodiscard]] manifold::Manifold build_triangular_prism(
    const TriangularPrism& prism) {
    if (!prism.is_valid()) {
        throw std::logic_error(
            "TriangularPrism cutters must enclose a volume: the base must be a "
            "real triangle and the extrusion must leave its plane");
    }

    // Vertices 0-2 are the base, 3-5 the extruded copy in the same order.
    const Vector3D extrusion = prism.height();
    const std::array<Point3D, 6> corners{prism.p1(),
                                         prism.p2(),
                                         prism.p3(),
                                         Point3D{prism.p1() + extrusion},
                                         Point3D{prism.p2() + extrusion},
                                         Point3D{prism.p3() + extrusion}};

    // The base is ordered so (p2 - p1) x (p3 - p1) points ALONG the extrusion,
    // so the base triangle faces into the solid and is wound backwards here;
    // the top keeps the base order. Each wall is the quad between one base
    // edge and its extruded copy, split into two triangles.
    std::vector<std::uint64_t> tri_verts{
        0U, 2U, 1U,  // base, reversed to face outward
        3U, 4U, 5U,  // top
    };
    for (std::uint64_t edge = 0U; edge < 3U; ++edge) {
        const std::uint64_t next = (edge + 1U) % 3U;
        tri_verts.insert(tri_verts.end(), {edge, next, next + 3U});
        tri_verts.insert(tri_verts.end(), {edge, next + 3U, edge + 3U});
    }

    manifold::MeshGL64 mesh;
    mesh.numProp = 3U;
    mesh.vertProperties.reserve(corners.size() * 3U);
    for (const Point3D& corner : corners) {
        mesh.vertProperties.insert(mesh.vertProperties.end(),
                                   {corner.x(), corner.y(), corner.z()});
    }
    mesh.triVerts = std::move(tri_verts);

    const manifold::Manifold solid(mesh);
    if (solid.Status() != manifold::Manifold::Error::NoError) {
        throw std::logic_error("TriangularPrism cutter is not a closed solid");
    }
    return solid;
}

[[nodiscard]] manifold::Manifold build_cube(const Cube& cube) {
    return manifold::Manifold::Cube(to_manifold_vec(cube.extent()),
                                    /*center=*/true)
        .Transform(to_manifold_transform(cube.orientation().toRotationMatrix(),
                                         cube.center()));
}

}  // namespace

manifold::Manifold build_cutter(
    const Primitive& cutter, const CoordinateTransformation& world_transform) {
    const Primitive world_cutter = ops::transform(cutter, world_transform);
    return std::visit(
        [](const auto& concrete_cutter) -> manifold::Manifold {
            using T = std::decay_t<decltype(concrete_cutter)>;
            if constexpr (std::is_same_v<T, Sphere>) {
                return build_sphere(concrete_cutter);
            } else if constexpr (std::is_same_v<T, Cylinder>) {
                return build_cylinder(concrete_cutter);
            } else if constexpr (std::is_same_v<T, Cone>) {
                return build_cone(concrete_cutter);
            } else if constexpr (std::is_same_v<T, Cube>) {
                return build_cube(concrete_cutter);
            } else if constexpr (std::is_same_v<T, TriangularPrism>) {
                return build_triangular_prism(concrete_cutter);
            } else {
                throw std::logic_error(
                    "CutGroup cutters must be closed solid primitives");
            }
        },
        world_cutter);
}

}  // namespace pycanha::gmm::cutting
