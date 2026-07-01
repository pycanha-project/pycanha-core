#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "test_helpers.hpp"

namespace {

using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::Cone;
namespace tests = pycanha::gmm::tests;

}  // namespace

TEST_CASE("Cone supports frustum surface operations",
          "[gmm][primitive][cone]") {
    using std::numbers::pi;

    Cone cone({0.0, 0.0, 0.0}, {0.0, 0.0, 4.0}, {1.0, 0.0, 0.0}, 1.0, 3.0, 0.0,
              2.0 * pi);

    SECTION("constructor and setters keep values") {
        REQUIRE(cone.radius1() == Catch::Approx(1.0));
        REQUIRE(cone.radius2() == Catch::Approx(3.0));
        cone.set_radius1(0.5);
        cone.set_radius2(2.5);
        REQUIRE(cone.radius1() == Catch::Approx(0.5));
        REQUIRE(cone.radius2() == Catch::Approx(2.5));
    }

    SECTION("validity enforces non-degenerate profile") {
        REQUIRE(cone.is_valid());
        REQUIRE_FALSE(Cone({0.0, 0.0, 0.0}, {0.0, 0.0, 4.0}, {1.0, 0.0, 0.0},
                           0.0, 0.0, 0.0, 2.0 * pi)
                          .is_valid());
        REQUIRE_FALSE(Cone({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                           1.0, 3.0, 0.0, 2.0 * pi)
                          .is_valid());
    }

    SECTION("uv conversion round-trips representative points") {
        tests::require_round_trip(
            cone, std::array<Point3D, 5>{
                      Point3D(1.0, 0.0, 0.0), Point3D(0.0, 1.5, 1.0),
                      Point3D(-2.0, 0.0, 2.0), Point3D(0.0, -2.5, 3.0),
                      Point3D(3.0, 0.0, 4.0)});
    }

    SECTION("surface area matches the analytic formula") {
        REQUIRE(cone.surface_area() ==
                Catch::Approx(8.0 * pi * std::sqrt(5.0)));
    }

    SECTION("normal points outward") {
        tests::require_parallel(cone.normal_at_uv({0.0, 0.0}),
                                Vector3D(4.0, 0.0, -2.0));
    }
}
