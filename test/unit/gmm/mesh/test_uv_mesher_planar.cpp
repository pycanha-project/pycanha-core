#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
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
    REQUIRE(gmm_test::face_ids_cover_all_cells(mesh, thermal_mesh));
    REQUIRE(gmm_test::sum_triangle_areas(mesh) ==
            Catch::Approx(rectangle.surface_area()));
}

TEST_CASE("UvMesher minimally triangulates quadrilaterals", "[gmm][mesh]") {
    const Quadrilateral quadrilateral({0.0, 0.0, 0.0}, {2.0, 0.5, 0.0},
                                      {3.0, 2.5, 0.0}, {1.0, 2.0, 0.0});
    const ThermalMesh thermal_mesh({0.0, 0.5, 1.0}, {0.0, 0.5, 1.0});
    const UvMesher mesher;

    const auto mesh = mesher.mesh(quadrilateral, thermal_mesh, MeshOptions{});

    REQUIRE(mesh.vertices.rows() == 9);
    REQUIRE(mesh.triangles.rows() == 8);
    REQUIRE(mesh_ops::has_consistent_face_ids(mesh));
    REQUIRE(gmm_test::face_ids_cover_all_cells(mesh, thermal_mesh));
    REQUIRE(gmm_test::sum_triangle_areas(mesh) ==
            Catch::Approx(quadrilateral.surface_area()));
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
    REQUIRE(gmm_test::face_ids_cover_all_cells(mesh, thermal_mesh));
    REQUIRE(gmm_test::sum_triangle_areas(mesh) ==
            Catch::Approx(triangle.surface_area()));
}
