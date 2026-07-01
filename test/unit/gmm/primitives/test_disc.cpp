#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "test_helpers.hpp"

namespace {

using pycanha::LENGTH_TOL;
using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::Disc;
namespace tests = pycanha::gmm::tests;

}  // namespace

TEST_CASE("Disc supports annular sector geometry operations",
          "[gmm][primitive][disc]") {
    using std::numbers::pi;

    Disc disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {2.0, 0.0, 0.0}, 1.0, 3.0, 0.0,
              pi);

    SECTION("constructor and setters keep values") {
        REQUIRE(disc.p1().isApprox(Point3D(0.0, 0.0, 0.0), LENGTH_TOL));
        REQUIRE(disc.p2().isApprox(Point3D(0.0, 0.0, 1.0), LENGTH_TOL));
        REQUIRE(disc.p3().isApprox(Point3D(2.0, 0.0, 0.0), LENGTH_TOL));
        REQUIRE(disc.inner_radius() == Catch::Approx(1.0));
        REQUIRE(disc.outer_radius() == Catch::Approx(3.0));

        disc.set_inner_radius(0.5);
        disc.set_outer_radius(2.5);
        disc.set_start_angle(0.25);
        disc.set_end_angle(2.75);

        REQUIRE(disc.inner_radius() == Catch::Approx(0.5));
        REQUIRE(disc.outer_radius() == Catch::Approx(2.5));
        REQUIRE(disc.start_angle() == Catch::Approx(0.25));
        REQUIRE(disc.end_angle() == Catch::Approx(2.75));
    }

    SECTION("validity enforces orthogonality and radii ordering") {
        REQUIRE(disc.is_valid());
        REQUIRE_FALSE(Disc({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
                           1.0, 3.0, 0.0, pi)
                          .is_valid());
        REQUIRE_FALSE(Disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {2.0, 0.0, 0.0},
                           2.0, 2.0, 0.0, pi)
                          .is_valid());
    }

    SECTION("uv conversion round-trips representative points") {
        const double sqrt_half = std::sqrt(0.5);
        tests::require_round_trip(
            disc, std::array<Point3D, 5>{
                      Point3D(1.0, 0.0, 0.0),
                      Point3D(1.5 * sqrt_half, 1.5 * sqrt_half, 0.0),
                      Point3D(0.0, 2.0, 0.0),
                      Point3D(2.5 * sqrt_half, 2.5 * sqrt_half, 0.0),
                      Point3D(0.0, 3.0, 0.0)});
    }

    SECTION("surface area matches the annular sector formula") {
        REQUIRE(disc.surface_area() == Catch::Approx(4.0 * pi));
    }

    SECTION("normal follows the axis direction") {
        tests::require_parallel(disc.normal_at_uv({0.0, 1.0}),
                                Vector3D(0.0, 0.0, 1.0));
    }
}
