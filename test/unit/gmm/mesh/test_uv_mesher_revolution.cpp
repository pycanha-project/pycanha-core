#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/validate.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/uv_mesher.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "test_uv_mesher_helpers.hpp"

namespace {

using pycanha::Point3D;
using pycanha::gmm::Cone;
using pycanha::gmm::Cylinder;
using pycanha::gmm::MeshOptions;
using pycanha::gmm::Paraboloid;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::UvMesher;
namespace mesh_ops = pycanha::gmm::mesh::ops;
namespace gmm_test = pycanha::gmm::test;

template <typename PrimitiveType>
void require_refinement_improves_area(const PrimitiveType& primitive,
                                      const ThermalMesh& thermal_mesh,
                                      double coarse_tolerance,
                                      double fine_tolerance) {
    const UvMesher mesher;
    const auto coarse_mesh =
        mesher.mesh(primitive, thermal_mesh, MeshOptions{coarse_tolerance});
    const auto fine_mesh =
        mesher.mesh(primitive, thermal_mesh, MeshOptions{fine_tolerance});

    REQUIRE(fine_mesh.vertices.rows() > coarse_mesh.vertices.rows());
    REQUIRE(fine_mesh.triangles.rows() > coarse_mesh.triangles.rows());
    REQUIRE(
        gmm_test::absolute_area_error(fine_mesh, primitive.surface_area()) <
        gmm_test::absolute_area_error(coarse_mesh, primitive.surface_area()));
    REQUIRE(gmm_test::face_ids_cover_all_cells(fine_mesh, thermal_mesh));
}

}  // namespace

TEST_CASE("UvMesher refines cylinders by deviation tolerance", "[gmm][mesh]") {
    const Cylinder cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 2.5}, {1.2, 0.0, 0.0},
                            1.2, 0.0, 2.0 * std::numbers::pi);
    const ThermalMesh thermal_mesh({0.0, 0.2, 0.7, 1.0}, {0.0, 0.4, 1.0});

    require_refinement_improves_area(cylinder, thermal_mesh, 0.25, 0.02);
}

TEST_CASE("UvMesher refines cones and collapses the apex", "[gmm][mesh]") {
    const Cone cone({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {1.5, 0.0, 0.0}, 0.0, 1.5,
                    0.0, 2.0 * std::numbers::pi);
    const ThermalMesh thermal_mesh({0.0, 0.33, 0.66, 1.0},
                                   {0.0, 0.2, 0.7, 1.0});
    const UvMesher mesher;

    const auto fine_mesh = mesher.mesh(cone, thermal_mesh, MeshOptions{0.01});

    REQUIRE(gmm_test::count_vertices_near(fine_mesh, cone.p1(),
                                          pycanha::LENGTH_TOL) == 1U);
    REQUIRE(gmm_test::face_ids_cover_all_cells(fine_mesh, thermal_mesh));
    require_refinement_improves_area(cone, thermal_mesh, 0.2, 0.01);
}

TEST_CASE("UvMesher refines paraboloids and collapses the apex",
          "[gmm][mesh]") {
    const Paraboloid paraboloid({0.0, 0.0, 0.0}, {0.0, 0.0, 3.0},
                                {1.8, 0.0, 0.0}, 1.8, 0.0,
                                2.0 * std::numbers::pi);
    const ThermalMesh thermal_mesh({0.0, 0.25, 0.5, 1.0},
                                   {0.0, 0.15, 0.55, 1.0});
    const UvMesher mesher;

    const auto fine_mesh =
        mesher.mesh(paraboloid, thermal_mesh, MeshOptions{0.01});

    REQUIRE(gmm_test::count_vertices_near(fine_mesh, paraboloid.p1(),
                                          pycanha::LENGTH_TOL) == 1U);
    REQUIRE(gmm_test::has_no_degenerate_triangles(fine_mesh, 1e-12));
    REQUIRE(gmm_test::face_ids_cover_all_cells(fine_mesh, thermal_mesh));
    require_refinement_improves_area(paraboloid, thermal_mesh, 0.25, 0.01);
}
