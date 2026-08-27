#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "test_helpers.hpp"

namespace {

using pycanha::LENGTH_TOL;
using pycanha::Point2D;
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

    SECTION("surface area of a parallelogram") {
        // These four corners happen to form a parallelogram, where the true
        // area and the old p3-free formula agree -- which is why this fixture
        // never caught the defect.
        REQUIRE(quadrilateral.surface_area() == Catch::Approx(3.0));
    }

    SECTION("normal follows the positive orientation") {
        tests::require_parallel(quadrilateral.normal_at_uv({0.1, 0.1}),
                                Vector3D(0.0, 0.0, 1.0));
    }
}

// A symmetric trapezoid: parallel edges of 4 m and 2 m, 2 m apart. Its area is
// (a + b) / 2 * h = 6 by elementary geometry -- an independent number, not the
// primitive's own other half.
namespace {

[[nodiscard]] Quadrilateral make_trapezoid() {
    return {{0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {3.0, 2.0, 0.0}, {1.0, 2.0, 0.0}};
}

// Strongly skewed but still convex and planar, to exercise the quadratic
// branch of the inverse map rather than its parallelogram fallback.
[[nodiscard]] Quadrilateral make_skewed() {
    return {{0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {1.0, 2.0, 0.0}, {0.0, 1.5, 0.0}};
}

// Worst absolute error of to_uv(to_cartesian(uv)) over a grid of the whole
// parameter square, reported as one number so the assertion stays one deep.
[[nodiscard]] double worst_round_trip_error(
    const Quadrilateral& quadrilateral) {
    constexpr int steps = 10;
    double worst = 0.0;
    for (int u_step = 0; u_step <= steps; ++u_step) {
        for (int v_step = 0; v_step <= steps; ++v_step) {
            const Point2D uv{static_cast<double>(u_step) / steps,
                             static_cast<double>(v_step) / steps};
            const Point2D round_tripped =
                quadrilateral.to_uv(quadrilateral.to_cartesian(uv));
            worst = std::max({worst, std::abs(round_tripped.x() - uv.x()),
                              std::abs(round_tripped.y() - uv.y())});
        }
    }
    return worst;
}

}  // namespace

TEST_CASE("Quadrilateral is a bilinear patch on all four corners",
          "[gmm][primitive][quadrilateral]") {
    const Quadrilateral trapezoid = make_trapezoid();

    SECTION("area is the trapezoid's, not the spanning parallelogram's") {
        REQUIRE(trapezoid.surface_area() == Catch::Approx(6.0));
        // The parallelogram on p2 - p1 and p4 - p1 would be 4 x 2 = 8, a third
        // too much: p3 has to enter the area.
        REQUIRE(trapezoid.surface_area() !=
                Catch::Approx((trapezoid.p2() - trapezoid.p1())
                                  .cross(trapezoid.p4() - trapezoid.p1())
                                  .norm()));
    }

    SECTION("the corners are the corners of the uv square") {
        REQUIRE(trapezoid.to_cartesian({0.0, 0.0})
                    .isApprox(trapezoid.p1(), LENGTH_TOL));
        REQUIRE(trapezoid.to_cartesian({1.0, 0.0})
                    .isApprox(trapezoid.p2(), LENGTH_TOL));
        REQUIRE(trapezoid.to_cartesian({1.0, 1.0})
                    .isApprox(trapezoid.p3(), LENGTH_TOL));
        REQUIRE(trapezoid.to_cartesian({0.0, 1.0})
                    .isApprox(trapezoid.p4(), LENGTH_TOL));
    }

    SECTION("uv round-trips over the whole patch") {
        REQUIRE(trapezoid.is_valid());
        REQUIRE(worst_round_trip_error(trapezoid) < 1.0e-9);
    }

    SECTION("uv round-trips on a strongly skewed patch") {
        const Quadrilateral skewed = make_skewed();
        REQUIRE(skewed.is_valid());
        REQUIRE(worst_round_trip_error(skewed) < 1.0e-9);
    }
}
