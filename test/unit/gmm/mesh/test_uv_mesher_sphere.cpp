#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/validate.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/uv_mesher.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "test_uv_mesher_helpers.hpp"

namespace {

using pycanha::Point3D;
using pycanha::gmm::MeshOptions;
using pycanha::gmm::Sphere;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::UvMesher;
namespace mesh_ops = pycanha::gmm::mesh::ops;
namespace gmm_test = pycanha::gmm::test;

}  // namespace

TEST_CASE("UvMesher builds watertight spheres with pole collapse",
          "[gmm][mesh]") {
    const Sphere sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 1.0,
                        -1.0, 1.0, 0.0, 2.0 * std::numbers::pi);
    const ThermalMesh thermal_mesh({0.0, 0.25, 0.5, 0.75, 1.0},
                                   {0.0, 0.2, 0.5, 0.8, 1.0});
    const UvMesher mesher;

    const auto mesh = mesher.mesh(sphere, thermal_mesh, MeshOptions{1e-3});

    REQUIRE(mesh_ops::is_watertight(mesh));
    REQUIRE(mesh_ops::is_watertight_on_curved_edges(mesh));
    REQUIRE(gmm_test::face_ids_cover_all_cells(mesh, thermal_mesh));
    REQUIRE(gmm_test::count_vertices_near(mesh, Point3D(0.0, 0.0, 1.0),
                                          pycanha::LENGTH_TOL) == 1U);
    REQUIRE(gmm_test::count_vertices_near(mesh, Point3D(0.0, 0.0, -1.0),
                                          pycanha::LENGTH_TOL) == 1U);
    REQUIRE(gmm_test::has_no_degenerate_triangles(mesh, 1e-12));
    REQUIRE(gmm_test::sum_triangle_areas(mesh) ==
            Catch::Approx(4.0 * std::numbers::pi).epsilon(0.001));
}

TEST_CASE("UvMesher refines polar-cap rows independently of equator rows",
          "[gmm][mesh]") {
    const Sphere sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 1.0,
                        -1.0, 1.0, 0.0, 2.0 * std::numbers::pi);
    const ThermalMesh thermal_mesh({0.0, 0.2, 0.7, 1.0},
                                   {0.0, 0.08, 0.35, 1.0});
    const UvMesher mesher;

    const auto coarse_mesh =
        mesher.mesh(sphere, thermal_mesh, MeshOptions{0.1});
    const auto fine_mesh = mesher.mesh(sphere, thermal_mesh, MeshOptions{0.01});

    REQUIRE(fine_mesh.vertices.rows() > coarse_mesh.vertices.rows());
    REQUIRE(fine_mesh.triangles.rows() > coarse_mesh.triangles.rows());
    REQUIRE(gmm_test::absolute_area_error(fine_mesh, sphere.surface_area()) <
            gmm_test::absolute_area_error(coarse_mesh, sphere.surface_area()));
    REQUIRE(gmm_test::count_vertices_near(fine_mesh, Point3D(0.0, 0.0, 1.0),
                                          pycanha::LENGTH_TOL) == 1U);
}
