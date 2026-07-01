#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "test_helpers.hpp"

namespace {

using pycanha::LENGTH_TOL;
using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::Rectangle;
namespace tests = pycanha::gmm::tests;

}  // namespace

TEST_CASE("Rectangle exposes intrinsic geometry operations",
          "[gmm][primitive][rectangle]") {
    Rectangle rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 3.0, 0.0});

    SECTION("constructor and setters keep values") {
        REQUIRE(rectangle.p1().isApprox(Point3D(0.0, 0.0, 0.0), LENGTH_TOL));
        REQUIRE(rectangle.p2().isApprox(Point3D(2.0, 0.0, 0.0), LENGTH_TOL));
        REQUIRE(rectangle.p3().isApprox(Point3D(0.0, 3.0, 0.0), LENGTH_TOL));

        rectangle.set_p1({1.0, 1.0, 0.0});
        rectangle.set_p2({3.0, 1.0, 0.0});
        rectangle.set_p3({1.0, 4.0, 0.0});

        REQUIRE(rectangle.p1().isApprox(Point3D(1.0, 1.0, 0.0), LENGTH_TOL));
        REQUIRE(rectangle.p2().isApprox(Point3D(3.0, 1.0, 0.0), LENGTH_TOL));
        REQUIRE(rectangle.p3().isApprox(Point3D(1.0, 4.0, 0.0), LENGTH_TOL));
    }

    SECTION("validity matches orthogonal edge constraints") {
        REQUIRE(rectangle.is_valid());
        REQUIRE_FALSE(
            Rectangle({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 3.0, 0.0})
                .is_valid());
        REQUIRE_FALSE(
            Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {1.0, 3.0, 0.0})
                .is_valid());
    }

    SECTION("uv conversion round-trips representative points") {
        tests::require_round_trip(
            rectangle, std::array<Point3D, 5>{
                           Point3D(0.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0),
                           Point3D(0.0, 3.0, 0.0), Point3D(1.0, 1.5, 0.0),
                           Point3D(0.5, 2.0, 0.0)});
    }

    SECTION("surface area matches analytic formula") {
        REQUIRE(rectangle.surface_area() == Catch::Approx(6.0));
    }

    SECTION("normal follows the positive orientation") {
        tests::require_parallel(rectangle.normal_at_uv({0.1, 0.1}),
                                Vector3D(0.0, 0.0, 1.0));
    }
}
