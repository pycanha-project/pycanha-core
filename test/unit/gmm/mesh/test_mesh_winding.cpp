#include <Eigen/Dense>
#include <algorithm>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/mesh/uv_mesher.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"

namespace {

using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::Cone;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Disc;
using pycanha::gmm::MeshOptions;
using pycanha::gmm::Paraboloid;
using pycanha::gmm::Primitive;
using pycanha::gmm::Quadrilateral;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Sphere;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::Triangle;
using pycanha::gmm::UvMesher;

constexpr double full_turn = 2.0 * std::numbers::pi;

// The winding of a triangle IS the definition of side 1: the raytracer reads
// the front face as the pair's even (side-1) face and negates it for side 2,
// while the readers, the conduction builder and the cut classifier all reason
// from normal_at_uv. If the two disagree for a primitive, every model using it
// has that primitive's two sides swapped -- node numbers, thermo-optical
// properties and activity flags all land on the wrong physical face.
//
// Nothing asserted this before, which is how the Disc shipped wound against
// its own axis.
[[nodiscard]] double smallest_winding_agreement(const Primitive& primitive) {
    const UvMesher mesher;
    const ThermalMesh thermal_mesh{{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}};
    const auto mesh = mesher.mesh(primitive, thermal_mesh, MeshOptions{});
    REQUIRE(mesh.triangles.rows() > 0);

    double worst = 1.0;
    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const Point3D p0 =
            mesh.vertices.row(mesh.triangles(tri_idx, 0)).transpose();
        const Point3D p1 =
            mesh.vertices.row(mesh.triangles(tri_idx, 1)).transpose();
        const Point3D p2 =
            mesh.vertices.row(mesh.triangles(tri_idx, 2)).transpose();

        const Vector3D winding_normal = (p1 - p0).cross(p2 - p0).normalized();
        const Point3D centroid = (p0 + p1 + p2) / 3.0;
        const Vector3D surface_normal = std::visit(
            [&centroid](const auto& concrete) {
                return concrete.normal_at_uv(concrete.to_uv(centroid));
            },
            primitive);

        worst = std::min(worst, winding_normal.dot(surface_normal));
    }
    return worst;
}

}  // namespace

TEST_CASE("Every primitive's triangulation winds along its own normal",
          "[gmm][mesh][winding]") {
    const std::vector<std::pair<std::string, Primitive>> primitives{
        {"rectangle",
         Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.5, 0.0})},
        {"triangle",
         Triangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.5, 0.0})},
        {"quadrilateral", Quadrilateral({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
                                        {1.7, 1.2, 0.0}, {0.0, 1.5, 0.0})},
        {"disc", Disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 0.0,
                      1.0, 0.0, full_turn)},
        {"annulus", Disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 0.4,
                         1.0, 0.0, full_turn)},
        {"cylinder", Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {1.0, 0.0, 0.0},
                              1.0, 0.0, full_turn)},
        {"cone", Cone({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {1.0, 0.0, 0.0}, 1.0,
                      0.4, 0.0, full_turn)},
        {"sphere", Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0},
                          1.0, -1.0, 1.0, 0.0, full_turn)},
        {"paraboloid", Paraboloid({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
                                  {1.0, 0.0, 0.0}, 1.0, 0.0, full_turn)},
    };

    for (const auto& [name, primitive] : primitives) {
        INFO("primitive: " << name);
        // Chords of a curved surface sit slightly off the surface normal, so
        // the invariant is the sign, not unity.
        REQUIRE(smallest_winding_agreement(primitive) > 0.0);
    }
}

TEST_CASE("A disc's triangulation faces the way its axis points",
          "[gmm][mesh][winding]") {
    // Disc::normal_at_uv is (p2 - p1) normalized, so side 1 is the +z face
    // here. Pinning the direction, not just the axis: the mesher used to wind
    // the other way, which swapped side 1 and side 2 for every disc.
    const Disc disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 0.0, 1.0,
                    0.0, full_turn);
    const UvMesher mesher;
    const auto mesh =
        mesher.mesh(disc, ThermalMesh{{0.0, 1.0}, {0.0, 1.0}}, MeshOptions{});

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const Point3D p0 =
            mesh.vertices.row(mesh.triangles(tri_idx, 0)).transpose();
        const Point3D p1 =
            mesh.vertices.row(mesh.triangles(tri_idx, 1)).transpose();
        const Point3D p2 =
            mesh.vertices.row(mesh.triangles(tri_idx, 2)).transpose();
        const Vector3D winding_normal = (p1 - p0).cross(p2 - p0).normalized();
        REQUIRE(winding_normal.z() > 0.99);
    }
}
