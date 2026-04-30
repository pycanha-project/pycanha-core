#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "test_helpers.hpp"

namespace {

using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::Cylinder;
namespace tests = pycanha::gmm::tests;

}  // namespace

TEST_CASE("Cylinder supports intrinsic surface operations",
          "[gmm][primitive][cylinder]") {
    using std::numbers::pi;

    Cylinder cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 4.0}, {1.0, 0.0, 0.0}, 2.0,
                      0.0, 2.0 * pi);

    SECTION("constructor and setters keep values") {
        REQUIRE(cylinder.radius() == Catch::Approx(2.0));
        cylinder.set_radius(1.5);
        cylinder.set_start_angle(0.5);
        cylinder.set_end_angle(5.5);
        REQUIRE(cylinder.radius() == Catch::Approx(1.5));
        REQUIRE(cylinder.start_angle() == Catch::Approx(0.5));
        REQUIRE(cylinder.end_angle() == Catch::Approx(5.5));
    }

    SECTION("validity enforces orthogonality and positive radius") {
        REQUIRE(cylinder.is_valid());
        REQUIRE_FALSE(Cylinder({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                               {1.0, 0.0, 0.0}, 2.0, 0.0, 2.0 * pi)
                          .is_valid());
        REQUIRE_FALSE(Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 4.0},
                               {1.0, 0.0, 0.0}, 0.0, 0.0, 2.0 * pi)
                          .is_valid());
    }

    SECTION("uv conversion round-trips representative points") {
        tests::require_round_trip(
            cylinder, std::array<Point3D, 5>{
                          Point3D(2.0, 0.0, 0.0), Point3D(0.0, 2.0, 1.0),
                          Point3D(-2.0, 0.0, 2.0), Point3D(0.0, -2.0, 3.0),
                          Point3D(2.0, 0.0, 4.0)});
    }

    SECTION("surface area matches the analytic formula") {
        REQUIRE(cylinder.surface_area() == Catch::Approx(16.0 * pi));
    }

    SECTION("normal is radial") {
        tests::require_parallel(cylinder.normal_at_uv({0.0, 1.0}),
                                Vector3D(1.0, 0.0, 0.0));
    }
}
