#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/cutting/cutter_proxy.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/uv_mesher.hpp"
#include "pycanha-core/gmm/ops/distance.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/triangular_prism.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::Point3D;
using pycanha::gmm::Geometry;
using pycanha::gmm::GeometryGroupCutted;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::MeshOptions;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriangularPrism;
using pycanha::gmm::UvMesher;
namespace mesh_ops = pycanha::gmm::mesh::ops;

// A right-angled wedge: the base is the half-square with legs of 1 m in the
// z = 0 plane, extruded 2 m along +z. Volume 1, so it is a real solid.
[[nodiscard]] TriangularPrism make_wedge() {
    return {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 2.0}};
}

// Worst absolute error of to_uv(to_cartesian(uv)) over an interior grid of all
// five faces, reported as one number so the assertion stays one deep. Face
// borders are skipped: a point on an edge belongs to both faces that meet
// there, so which one to_uv picks is arbitrary and not part of the contract.
[[nodiscard]] double worst_round_trip_error(const TriangularPrism& prism) {
    constexpr int faces = 5;
    constexpr int steps = 5;
    double worst = 0.0;
    for (int face = 0; face < faces; ++face) {
        for (int first = 1; first < steps; ++first) {
            for (int second = 1; second < steps; ++second) {
                const pycanha::Point2D uv{
                    face + (static_cast<double>(first) / steps),
                    static_cast<double>(second) / steps};
                const pycanha::Point2D back =
                    prism.to_uv(prism.to_cartesian(uv));
                worst = std::max({worst, std::abs(back.x() - uv.x()),
                                  std::abs(back.y() - uv.y())});
            }
        }
    }
    return worst;
}

}  // namespace

TEST_CASE("TriangularPrism describes a closed solid",
          "[gmm][primitive][prism]") {
    const TriangularPrism wedge = make_wedge();

    SECTION("validity needs a real base and an extrusion out of its plane") {
        REQUIRE(wedge.is_valid());
        // Degenerate base.
        REQUIRE_FALSE(TriangularPrism({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                      {2.0, 0.0, 0.0}, {0.0, 0.0, 1.0})
                          .is_valid());
        // Extrusion inside the base plane: no volume to subtract.
        REQUIRE_FALSE(TriangularPrism({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0})
                          .is_valid());
    }

    SECTION("surface area counts both bases and all three walls") {
        // Two 0.5 bases, two 1 x 2 walls and the sqrt(2) x 2 hypotenuse wall.
        REQUIRE(wedge.surface_area() ==
                Catch::Approx(1.0 + 2.0 + 2.0 + (2.0 * std::numbers::sqrt2)));
    }

    SECTION("distance is zero on the surface and positive outside") {
        REQUIRE(pycanha::gmm::ops::distance(wedge, Point3D(0.5, 0.0, 1.0)) ==
                Catch::Approx(0.0).margin(1e-12));
        REQUIRE(pycanha::gmm::ops::distance(wedge, Point3D(0.0, 0.0, 5.0)) ==
                Catch::Approx(3.0));
        REQUIRE(pycanha::gmm::ops::distance(wedge, Point3D(-2.0, 0.5, 1.0)) ==
                Catch::Approx(2.0));
    }
}

TEST_CASE("TriangularPrism is cutter-only, like Cube",
          "[gmm][primitive][prism]") {
    const UvMesher mesher;
    REQUIRE_THROWS_AS(mesher.mesh(make_wedge(), ThermalMesh{}, MeshOptions{}),
                      std::logic_error);
}

TEST_CASE("TriangularPrism parametrises its five faces",
          "[gmm][primitive][prism]") {
    const TriangularPrism wedge = make_wedge();

    SECTION("uv round-trips on every face") {
        REQUIRE(worst_round_trip_error(wedge) < 1e-9);
    }

    SECTION("normals point out of the solid") {
        // Walls face away from the base's third corner; the base faces -z and
        // the top +z, since the extrusion runs along +z here.
        REQUIRE(wedge.normal_at_uv({0.5, 0.5})
                    .isApprox(pycanha::Vector3D(0.0, -1.0, 0.0)));
        REQUIRE(wedge.normal_at_uv({2.5, 0.5})
                    .isApprox(pycanha::Vector3D(-1.0, 0.0, 0.0)));
        REQUIRE(wedge.normal_at_uv({3.5, 0.5})
                    .isApprox(pycanha::Vector3D(0.0, 0.0, -1.0)));
        REQUIRE(wedge.normal_at_uv({4.5, 0.5})
                    .isApprox(pycanha::Vector3D(0.0, 0.0, 1.0)));
    }
}

TEST_CASE("A prism cutter removes the volume it covers", "[gmm][cutting]") {
    const auto prism = pycanha::gmm::cutting::build_cutter(make_wedge());
    REQUIRE(prism.Status() == manifold::Manifold::Error::NoError);
    // Half of a 1 x 1 x 2 box.
    REQUIRE(prism.Volume() == Catch::Approx(1.0));
}

TEST_CASE("A prism chamfers the corner of a plate", "[gmm][cutting]") {
    // A 2 m x 2 m plate in the z = 0 plane, with the wedge straddling it so
    // the cut takes the triangular corner between (0,0) and the (1,0)-(0,1)
    // diagonal: an area of 0.5 out of 4.
    auto plate = std::make_shared<GeometryItem>(
        "plate", Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}),
        ThermalMesh{});
    auto cutter = std::make_shared<GeometryItem>(
        "chamfer",
        TriangularPrism({0.0, 0.0, -1.0}, {1.0, 0.0, -1.0}, {0.0, 1.0, -1.0},
                        {0.0, 0.0, 1.0}),
        ThermalMesh{});

    GeometryGroupCutted cut("cut",
                            std::vector<std::shared_ptr<Geometry>>{plate},
                            std::vector<std::shared_ptr<GeometryItem>>{cutter});

    REQUIRE(mesh_ops::compute_areas(cut.mesh()).sum() ==
            Catch::Approx(3.5).margin(1e-6));
}
