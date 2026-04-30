#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "test_helpers.hpp"

namespace {

using pycanha::LENGTH_TOL;
using pycanha::Point2D;
using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::Cube;
namespace tests = pycanha::gmm::tests;

}  // namespace

TEST_CASE("Cube exposes a deterministic face atlas", "[gmm][primitive][cube]") {
    Cube cube({0.0, 0.0, 0.0}, {2.0, 4.0, 6.0});

    SECTION("constructor and setters keep values") {
        REQUIRE(cube.center().isApprox(Point3D(0.0, 0.0, 0.0), LENGTH_TOL));
        REQUIRE(cube.extent().isApprox(Vector3D(2.0, 4.0, 6.0), LENGTH_TOL));

        cube.set_center({1.0, 2.0, 3.0});
        cube.set_extent({4.0, 6.0, 8.0});

        REQUIRE(cube.center().isApprox(Point3D(1.0, 2.0, 3.0), LENGTH_TOL));
        REQUIRE(cube.extent().isApprox(Vector3D(4.0, 6.0, 8.0), LENGTH_TOL));
    }

    SECTION("validity requires positive extents") {
        REQUIRE(cube.is_valid());
        REQUIRE_FALSE(Cube({0.0, 0.0, 0.0}, {0.0, 4.0, 6.0}).is_valid());
    }

    SECTION("uv conversion round-trips representative face points") {
        tests::require_round_trip(
            cube, std::array<Point3D, 6>{
                      Point3D(1.0, 0.0, 0.0), Point3D(-1.0, 1.0, 2.0),
                      Point3D(0.0, 2.0, -1.0), Point3D(0.0, -2.0, 1.0),
                      Point3D(0.5, -1.0, 3.0), Point3D(-0.5, 1.0, -3.0)});
    }

    SECTION("surface area matches the box formula") {
        REQUIRE(cube.surface_area() == Catch::Approx(88.0));
    }

    SECTION("normal follows the selected atlas face") {
        const Point2D uv = cube.to_uv(Point3D(1.0, 0.0, 0.0));
        tests::require_parallel(cube.normal_at_uv(uv), Vector3D(1.0, 0.0, 0.0));
    }
}
