#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "test_helpers.hpp"

namespace {

using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::Paraboloid;
namespace tests = pycanha::gmm::tests;

}  // namespace

TEST_CASE("Paraboloid supports intrinsic shell operations",
          "[gmm][primitive][paraboloid]") {
    using std::numbers::pi;

    Paraboloid paraboloid({0.0, 0.0, 0.0}, {0.0, 0.0, 4.0}, {1.0, 0.0, 0.0},
                          2.0, 0.0, 2.0 * pi);

    SECTION("constructor and setters keep values") {
        REQUIRE(paraboloid.radius() == Catch::Approx(2.0));
        paraboloid.set_radius(1.5);
        paraboloid.set_start_angle(0.25);
        paraboloid.set_end_angle(5.75);
        REQUIRE(paraboloid.radius() == Catch::Approx(1.5));
        REQUIRE(paraboloid.start_angle() == Catch::Approx(0.25));
        REQUIRE(paraboloid.end_angle() == Catch::Approx(5.75));
    }

    SECTION("validity enforces non-degenerate axis and radius") {
        REQUIRE(paraboloid.is_valid());
        REQUIRE_FALSE(Paraboloid({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                 {1.0, 0.0, 0.0}, 2.0, 0.0, 2.0 * pi)
                          .is_valid());
        REQUIRE_FALSE(Paraboloid({0.0, 0.0, 0.0}, {0.0, 0.0, 4.0},
                                 {1.0, 0.0, 0.0}, 0.0, 0.0, 2.0 * pi)
                          .is_valid());
    }

    SECTION("uv conversion round-trips representative points") {
        using std::numbers::sqrt2;
        using std::numbers::sqrt3;

        tests::require_round_trip(
            paraboloid, std::array<Point3D, 5>{
                            Point3D(1.0, 0.0, 1.0), Point3D(0.0, sqrt2, 2.0),
                            Point3D(-sqrt3, 0.0, 3.0), Point3D(0.0, -2.0, 4.0),
                            Point3D(0.5, 0.0, 0.25)});
    }

    SECTION("surface area matches the analytic formula") {
        const double expected = pi * 2.0 * ((std::pow(68.0, 1.5)) - 8.0) / 48.0;
        REQUIRE(paraboloid.surface_area() == Catch::Approx(expected));
    }

    SECTION("normal points outward") {
        tests::require_parallel(paraboloid.normal_at_uv({0.0, 4.0}),
                                Vector3D(4.0, 0.0, -1.0));
    }
}
