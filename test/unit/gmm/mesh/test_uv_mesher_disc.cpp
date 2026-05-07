#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/validate.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/uv_mesher.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "test_uv_mesher_helpers.hpp"

namespace {

using pycanha::Point3D;
using pycanha::gmm::Disc;
using pycanha::gmm::MeshOptions;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::UvMesher;
namespace mesh_ops = pycanha::gmm::mesh::ops;
namespace gmm_test = pycanha::gmm::test;

}  // namespace

TEST_CASE("UvMesher refines annular disc boundaries independently",
          "[gmm][mesh]") {
    const Disc disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.2, 0.0, 0.0}, 0.4, 1.2,
                    0.0, 2.0 * std::numbers::pi);
    const ThermalMesh thermal_mesh({0.0, 0.25, 0.5, 1.0}, {0.0, 0.5, 1.0});
    const UvMesher mesher;

    const auto coarse_mesh = mesher.mesh(disc, thermal_mesh, MeshOptions{0.2});
    const auto fine_mesh = mesher.mesh(disc, thermal_mesh, MeshOptions{0.01});

    REQUIRE(fine_mesh.vertices.rows() > coarse_mesh.vertices.rows());
    REQUIRE(fine_mesh.triangles.rows() > coarse_mesh.triangles.rows());
    REQUIRE(gmm_test::absolute_area_error(fine_mesh, disc.surface_area()) <
            gmm_test::absolute_area_error(coarse_mesh, disc.surface_area()));
    REQUIRE(gmm_test::face_ids_cover_all_cells(fine_mesh, thermal_mesh));
    REQUIRE(mesh_ops::is_watertight_on_curved_edges(fine_mesh));
}

TEST_CASE("UvMesher collapses the full-disc center into one fan vertex",
          "[gmm][mesh]") {
    const Disc disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 0.0, 1.0,
                    0.0, 2.0 * std::numbers::pi);
    const ThermalMesh thermal_mesh({0.0, 0.2, 0.6, 1.0},
                                   {0.0, 0.33, 0.66, 1.0});
    const UvMesher mesher;

    const auto mesh = mesher.mesh(disc, thermal_mesh, MeshOptions{1e-3});
    const auto center_vertex_count =
        gmm_test::count_vertices_near(mesh, disc.p1(), pycanha::LENGTH_TOL);

    REQUIRE(center_vertex_count == 1U);
    REQUIRE(gmm_test::face_ids_cover_all_cells(mesh, thermal_mesh));
    REQUIRE(mesh_ops::is_watertight_on_curved_edges(mesh));
    REQUIRE(gmm_test::sum_triangle_areas(mesh) ==
            Catch::Approx(disc.surface_area()).epsilon(0.002));
}
