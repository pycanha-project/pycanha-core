#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "test_helpers.hpp"

namespace {

using pycanha::LENGTH_TOL;
using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::Quadrilateral;
namespace tests = pycanha::gmm::tests;

}  // namespace

TEST_CASE("Quadrilateral preserves planar value semantics",
          "[gmm][primitive][quadrilateral]") {
    Quadrilateral quadrilateral({0.0, 0.0, 0.0}, {2.0, 1.0, 0.0},
                                {3.0, 3.0, 0.0}, {1.0, 2.0, 0.0});

    SECTION("constructor and setters keep values") {
        REQUIRE(
            quadrilateral.p1().isApprox(Point3D(0.0, 0.0, 0.0), LENGTH_TOL));
        REQUIRE(
            quadrilateral.p2().isApprox(Point3D(2.0, 1.0, 0.0), LENGTH_TOL));
        REQUIRE(
            quadrilateral.p3().isApprox(Point3D(3.0, 3.0, 0.0), LENGTH_TOL));
        REQUIRE(
            quadrilateral.p4().isApprox(Point3D(1.0, 2.0, 0.0), LENGTH_TOL));

        quadrilateral.set_p1({1.0, 0.0, 0.0});
        quadrilateral.set_p2({3.0, 1.0, 0.0});
        quadrilateral.set_p3({4.0, 3.0, 0.0});
        quadrilateral.set_p4({2.0, 2.0, 0.0});

        REQUIRE(
            quadrilateral.p1().isApprox(Point3D(1.0, 0.0, 0.0), LENGTH_TOL));
        REQUIRE(
            quadrilateral.p2().isApprox(Point3D(3.0, 1.0, 0.0), LENGTH_TOL));
        REQUIRE(
            quadrilateral.p3().isApprox(Point3D(4.0, 3.0, 0.0), LENGTH_TOL));
        REQUIRE(
            quadrilateral.p4().isApprox(Point3D(2.0, 2.0, 0.0), LENGTH_TOL));
    }

    SECTION("validity rejects degenerate and non-planar cases") {
        REQUIRE(quadrilateral.is_valid());
        REQUIRE_FALSE(Quadrilateral({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
                                    {3.0, 3.0, 0.0}, {1.0, 2.0, 0.0})
                          .is_valid());
        REQUIRE_FALSE(Quadrilateral({0.0, 0.0, 0.0}, {2.0, 1.0, 0.0},
                                    {3.0, 3.0, 1.0}, {1.0, 2.0, 0.0})
                          .is_valid());
    }

    SECTION("uv conversion round-trips representative points") {
        tests::require_round_trip(
            quadrilateral, std::array<Point3D, 5>{
                               Point3D(0.0, 0.0, 0.0), Point3D(2.0, 1.0, 0.0),
                               Point3D(1.0, 2.0, 0.0), Point3D(1.5, 1.0, 0.0),
                               Point3D(2.25, 1.75, 0.0)});
    }

    SECTION("surface area matches the spanning parallelogram") {
        REQUIRE(quadrilateral.surface_area() == Catch::Approx(3.0));
    }

    SECTION("normal follows the positive orientation") {
        tests::require_parallel(quadrilateral.normal_at_uv({0.1, 0.1}),
                                Vector3D(0.0, 0.0, 1.0));
    }
}
