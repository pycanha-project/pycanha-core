#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/validate.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/mesh/uv_mesher.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"
#include "test_uv_mesher_helpers.hpp"

namespace {

using pycanha::gmm::MeshOptions;
using pycanha::gmm::Quadrilateral;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::Triangle;
using pycanha::gmm::UvMesher;
namespace mesh_ops = pycanha::gmm::mesh::ops;
namespace gmm_test = pycanha::gmm::test;

}  // namespace

// Meshed areas are checked against the shape's own closed form, never against
// primitive.surface_area(). Asserting the mesher agrees with the primitive
// passes whenever both are wrong in the same way, which is how a
// quadrilateral shipped for years as a rectangle.
TEST_CASE("UvMesher minimally triangulates rectangles", "[gmm][mesh]") {
    const Rectangle rectangle({0.0, 0.0, 0.0}, {4.0, 0.0, 0.0},
                              {0.0, 3.0, 0.0});
    const ThermalMesh thermal_mesh({0.0, 0.3, 1.0}, {0.0, 0.25, 0.75, 1.0});
    const UvMesher mesher;

    const auto mesh = mesher.mesh(rectangle, thermal_mesh, MeshOptions{});

    REQUIRE(mesh.vertices.rows() == 12);
    REQUIRE(mesh.triangles.rows() == 12);
    REQUIRE(mesh_ops::has_consistent_face_ids(mesh));
    REQUIRE_FALSE(mesh_ops::is_watertight(mesh));
    REQUIRE(gmm_test::face_ids_cover_all_face_pairs(mesh, thermal_mesh));
}

TEST_CASE("UvMesher covers a rectangle's own area and corners", "[gmm][mesh]") {
    const Rectangle rectangle({0.0, 0.0, 0.0}, {4.0, 0.0, 0.0},
                              {0.0, 3.0, 0.0});
    const UvMesher mesher;
    const auto mesh = mesher.mesh(
        rectangle, ThermalMesh({0.0, 0.3, 1.0}, {0.0, 0.25, 0.75, 1.0}),
        MeshOptions{});

    // 4 m x 3 m.
    REQUIRE(gmm_test::sum_triangle_areas(mesh) == Catch::Approx(12.0));
    const std::array<pycanha::Point3D, 3> corners{
        rectangle.p1(), rectangle.p2(), rectangle.p3()};
    REQUIRE(gmm_test::corners_present(mesh, corners, pycanha::LENGTH_TOL));
}

TEST_CASE("UvMesher minimally triangulates quadrilaterals", "[gmm][mesh]") {
    // A symmetric trapezoid: 4 m and 2 m parallel edges, 2 m apart, so
    // (4 + 2) / 2 * 2 = 6.
    const Quadrilateral quadrilateral({0.0, 0.0, 0.0}, {4.0, 0.0, 0.0},
                                      {3.0, 2.0, 0.0}, {1.0, 2.0, 0.0});
    const ThermalMesh thermal_mesh({0.0, 0.5, 1.0}, {0.0, 0.5, 1.0});
    const UvMesher mesher;

    const auto mesh = mesher.mesh(quadrilateral, thermal_mesh, MeshOptions{});

    REQUIRE(mesh.vertices.rows() == 9);
    REQUIRE(mesh.triangles.rows() == 8);
    REQUIRE(mesh_ops::has_consistent_face_ids(mesh));
    REQUIRE(gmm_test::face_ids_cover_all_face_pairs(mesh, thermal_mesh));
    REQUIRE(gmm_test::sum_triangle_areas(mesh) == Catch::Approx(6.0));
    // All four corners are in the mesh: p3 is a corner of the shape, not a
    // hint about it.
    const std::array<pycanha::Point3D, 4> corners{
        quadrilateral.p1(), quadrilateral.p2(), quadrilateral.p3(),
        quadrilateral.p4()};
    REQUIRE(gmm_test::corners_present(mesh, corners, pycanha::LENGTH_TOL));
}

TEST_CASE("UvMesher collapses the triangle apex edge into a fan",
          "[gmm][mesh]") {
    const Triangle triangle({0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {0.0, 2.0, 0.0});
    const ThermalMesh thermal_mesh({0.0, 0.5, 1.0}, {0.0, 0.5, 1.0});
    const UvMesher mesher;

    const auto mesh = mesher.mesh(triangle, thermal_mesh, MeshOptions{});

    REQUIRE(mesh.vertices.rows() == 7);
    REQUIRE(mesh.triangles.rows() == 6);
    REQUIRE(gmm_test::count_vertices_near(mesh, triangle.p1(),
                                          pycanha::LENGTH_TOL) == 1U);
    REQUIRE(gmm_test::face_ids_cover_all_face_pairs(mesh, thermal_mesh));
    // Half of 3 m x 2 m.
    REQUIRE(gmm_test::sum_triangle_areas(mesh) == Catch::Approx(3.0));
}
