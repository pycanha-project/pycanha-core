#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"

namespace {

using pycanha::LENGTH_TOL;
using pycanha::Point2D;
using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::Triangle;

constexpr double validation_tolerance = 1e-10;

[[nodiscard]] Triangle make_triangle() {
    return {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 3.0, 0.0}};
}

}  // namespace

TEST_CASE("Triangle keeps owned copies", "[gmm][primitive][triangle]") {
    Triangle triangle = make_triangle();
    const Point3D new_p1(1.0, 0.0, 0.0);

    REQUIRE(triangle.p1().isApprox(Point3D(0.0, 0.0, 0.0), LENGTH_TOL));
    REQUIRE(triangle.p2().isApprox(Point3D(2.0, 0.0, 0.0), LENGTH_TOL));
    REQUIRE(triangle.p3().isApprox(Point3D(0.0, 3.0, 0.0), LENGTH_TOL));

    triangle.set_p1(new_p1);
    triangle.set_p2({3.0, 0.0, 0.0});
    triangle.set_p3({1.0, 4.0, 0.0});

    REQUIRE(triangle.p1().isApprox(new_p1, LENGTH_TOL));
    REQUIRE(triangle.p2().isApprox(Point3D(3.0, 0.0, 0.0), LENGTH_TOL));
    REQUIRE(triangle.p3().isApprox(Point3D(1.0, 4.0, 0.0), LENGTH_TOL));
}

TEST_CASE("Triangle rejects degenerate inputs", "[gmm][primitive][triangle]") {
    const Triangle triangle = make_triangle();

    REQUIRE(triangle.is_valid());
    REQUIRE_FALSE(
        Triangle({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 3.0, 0.0}).is_valid());
    REQUIRE_FALSE(
        Triangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {4.0, 0.0, 0.0}).is_valid());
}

TEST_CASE("Triangle uv conversion round-trips points",
          "[gmm][primitive][triangle]") {
    const Triangle triangle = make_triangle();
    const std::array<Point3D, 5> points = {
        Point3D(0.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0), Point3D(0.0, 3.0, 0.0),
        Point3D(1.0, 1.0, 0.0), Point3D(0.5, 1.5, 0.0),
    };

    for (const Point3D& point : points) {
        const Point2D uv = triangle.to_uv(point);
        REQUIRE(triangle.to_cartesian(uv).isApprox(point, LENGTH_TOL));
    }
}

TEST_CASE("Triangle surface area matches the analytic formula",
          "[gmm][primitive][triangle]") {
    const Triangle triangle = make_triangle();

    REQUIRE(triangle.surface_area() == Catch::Approx(3.0));
}

TEST_CASE("Triangle normal follows positive orientation",
          "[gmm][primitive][triangle]") {
    const Triangle triangle = make_triangle();
    const Vector3D normal = triangle.normal_at_uv({0.25, 0.25});

    REQUIRE(normal.isApprox(Vector3D(0.0, 0.0, 1.0), validation_tolerance));
}
