#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <numbers>

#include "pycanha-core/gmm/primitives.hpp"
#include "pycanha-core/parameters.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

using pycanha::Point3D;

// NOLINTBEGIN(bugprone-chained-comparison)
TEST_CASE("Triangle Primitive", "[gmm][primitive][triangle]") {
    using pycanha::gmm::Triangle;

    SECTION("Check constructor and set/get methods") {
        Point3D p1(0.0, 0.0, 0.0);
        Point3D p2(1.0, 0.0, 0.0);
        Point3D p3(1.0, 1.0, 0.0);
        Triangle tri(p1, p2, p3);

        REQUIRE(tri.get_p1() == p1);
        REQUIRE(tri.get_p2() == p2);
        REQUIRE(tri.get_p3() == p3);

        p1.x() += 1.0;
        p2.x() += 1.0;
        p3.x() += 1.0;

        REQUIRE(tri.get_p1() != p1);
        REQUIRE(tri.get_p2() != p2);
        REQUIRE(tri.get_p3() != p3);

        tri.set_p1(p1);
        tri.set_p2(p2);
        tri.set_p3(p3);

        REQUIRE(tri.get_p1() == p1);
        REQUIRE(tri.get_p2() == p2);
        REQUIRE(tri.get_p3() == p3);
    }

    SECTION("Check valid triangle") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(1.0, 0.0, 0.0);
        const Point3D p3(1.0, 1.0, 0.0);
        const Triangle tri(p1, p2, p3);

        REQUIRE_THAT(tri.v1().norm(),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));

        REQUIRE_THAT(tri.v2().norm(),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));

        REQUIRE(tri.is_valid());
    }

    SECTION("Invalid Tri 1: p1 == p2") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(0.0, 0.0, 0.0);
        const Point3D p3(0.0, 1.0, 0.0);
        const Triangle tri(p1, p2, p3);

        REQUIRE(!tri.is_valid());
    }

    SECTION("Invalid Tri 2: Colinear") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(1.0, 0.0, 0.0);
        const Point3D p3(-2.0, 0.0, 0.0);
        const Triangle tri(p1, p2, p3);

        REQUIRE(!tri.is_valid());
    }

    SECTION("Triangle - Point distances") {
        const Point3D p1(0.0, 0.0, 1.0);
        const Point3D p2(1.0, 0.0, 1.0);
        const Point3D p3(0.0, 1.0, 1.0);
        const Triangle tri(p1, p2, p3);

        // Points at the vertices
        REQUIRE_THAT(tri.distance(Point3D(0.0, 0.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(tri.distance(Point3D(1.0, 0.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(tri.distance(Point3D(0.0, 1.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));

        // Points on the edges
        REQUIRE_THAT(tri.distance(Point3D(0.25, 0.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(tri.distance(Point3D(0.0, 0.25, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(tri.distance(Point3D(0.5, 0.5, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));

        // Point inside the triangle
        REQUIRE_THAT(tri.distance(Point3D(0.25, 0.25, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));

        // Points outside the triangle - same plane
        REQUIRE_THAT(tri.distance(Point3D(0.5, -0.25, 1.0)),
                     Catch::Matchers::WithinAbs(0.25, LENGTH_TOL));
        REQUIRE_THAT(tri.distance(Point3D(-0.25, 0.5, 1.0)),
                     Catch::Matchers::WithinAbs(0.25, LENGTH_TOL));
        REQUIRE_THAT(
            tri.distance(Point3D(1.0, 1.0, 1.0)),
            Catch::Matchers::WithinAbs(std::sqrt(2.0) / 2.0, LENGTH_TOL));

        // Points outside the triangle - out of plane
        REQUIRE_THAT(tri.distance(Point3D(0.5, -1.0, 0.0)),
                     Catch::Matchers::WithinAbs(std::sqrt(2.0), LENGTH_TOL));
        REQUIRE_THAT(tri.distance(Point3D(-1.0, -1.0, 0.0)),
                     Catch::Matchers::WithinAbs(std::sqrt(3.0), LENGTH_TOL));
        REQUIRE_THAT(tri.distance(Point3D(0.5, 0.5, 0.0)),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));
    }

    SECTION("Triangle - Point distance as cutted plane") {
        const Point3D p1(0.0, 0.0, 1.0);
        const Point3D p2(1.0, 0.0, 1.0);
        const Point3D p3(0.0, 1.0, 1.0);
        const Triangle tri(p1, p2, p3);

        // Vector3D n = (p2 - p1).cross(p3 - p1).normalized();

        // Points on the plane
    }

    SECTION("Triangle - 2D 3D transformations") {
        const Point3D p1(0.0, 0.0, 1.0);
        const Point3D p2(1.0, 0.0, 1.0);
        const Point3D p3(0.0, 1.0, 1.0);
        const Triangle tri(p1, p2, p3);

        auto p1_2d = tri.from_3d_to_2d(p1);
        auto p2_2d = tri.from_3d_to_2d(p2);
        auto p3_2d = tri.from_3d_to_2d(p3);

        // Check that the triangle vertices are converted correctly
        REQUIRE(p1_2d.isApprox(Point2D(0.0, 0.0), LENGTH_TOL));
        REQUIRE(p2_2d.isApprox(Point2D(1.0, 0.0), LENGTH_TOL));
        REQUIRE(p3_2d.isApprox(Point2D(0.0, 1.0), LENGTH_TOL));

        // Check that the backwards conversion also works
        REQUIRE(tri.from_2d_to_3d(p1_2d).isApprox(p1, LENGTH_TOL));
        REQUIRE(tri.from_2d_to_3d(p2_2d).isApprox(p2, LENGTH_TOL));
        REQUIRE(tri.from_2d_to_3d(p3_2d).isApprox(p3, LENGTH_TOL));
    }
}

TEST_CASE("Rectangle Primitive", "[gmm][primitive][rectangle]") {
    using pycanha::gmm::Rectangle;

    SECTION("Check constructor and set/get methods") {
        Point3D p1(0.0, 0.0, 0.0);
        Point3D p2(1.0, 0.0, 0.0);
        Point3D p3(0.0, 1.0, 0.0);
        Rectangle rect(p1, p2, p3);

        REQUIRE(rect.get_p1() == p1);
        REQUIRE(rect.get_p2() == p2);
        REQUIRE(rect.get_p3() == p3);

        p1.x() += 1.0;
        p2.x() += 1.0;
        p3.x() += 1.0;

        REQUIRE(rect.get_p1() != p1);
        REQUIRE(rect.get_p2() != p2);
        REQUIRE(rect.get_p3() != p3);

        rect.set_p1(p1);
        rect.set_p2(p2);
        rect.set_p3(p3);

        REQUIRE(rect.get_p1() == p1);
        REQUIRE(rect.get_p2() == p2);
        REQUIRE(rect.get_p3() == p3);
    }

    SECTION("Check valid rectangle") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(1.0, 0.0, 0.0);
        const Point3D p3(0.0, 1.0, 0.0);
        const Rectangle rect(p1, p2, p3);

        REQUIRE_THAT(rect.v1().norm(),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));

        REQUIRE_THAT(rect.v2().norm(),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));

        REQUIRE(rect.is_valid());
    }

    SECTION("Invalid Rect 1: p1 == p2") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(0.0, 0.0, 0.0);
        const Point3D p3(0.0, 1.0, 0.0);
        const Rectangle rect(p1, p2, p3);

        REQUIRE(!rect.is_valid());
    }

    SECTION("Invalid Rect 2: Non perpendicular") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(1.0, 0.0, 0.0);
        const Point3D p3(0.05, 1.0, 0.0);
        const Rectangle rect(p1, p2, p3);

        REQUIRE(!rect.is_valid());
    }

    SECTION("Rectangle - Point distances") {
        const Point3D p1(0.0, 0.0, 1.0);
        const Point3D p2(1.0, 0.0, 1.0);
        const Point3D p3(0.0, 1.0, 1.0);
        const Rectangle rect(p1, p2, p3);

        // Points at the vertices
        REQUIRE_THAT(rect.distance(Point3D(0.0, 0.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(1.0, 0.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(0.0, 1.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(1.0, 1.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));

        // Points on the edges
        REQUIRE_THAT(rect.distance(Point3D(0.5, 0.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(0.0, 0.5, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(0.5, 1.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(1.0, 0.5, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));

        // Point inside the rectangle
        REQUIRE_THAT(rect.distance(Point3D(0.5, 0.5, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));

        // Points outside the rectangle - same plane
        REQUIRE_THAT(rect.distance(Point3D(0.5, -0.25, 1.0)),
                     Catch::Matchers::WithinAbs(0.25, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(-0.25, 0.5, 1.0)),
                     Catch::Matchers::WithinAbs(0.25, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(0.5, 1.25, 1.0)),
                     Catch::Matchers::WithinAbs(0.25, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(1.25, 0.5, 1.0)),
                     Catch::Matchers::WithinAbs(0.25, LENGTH_TOL));

        REQUIRE_THAT(rect.distance(Point3D(1.25, 0.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.25, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(-0.25, 0.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.25, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(1.0, 1.25, 1.0)),
                     Catch::Matchers::WithinAbs(0.25, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(1.0, -0.25, 1.0)),
                     Catch::Matchers::WithinAbs(0.25, LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(1.25, 1.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.25, LENGTH_TOL));

        REQUIRE_THAT(rect.distance(Point3D(3.0, 2.0, 1.0)),
                     Catch::Matchers::WithinAbs(std::sqrt(5.0), LENGTH_TOL));

        // Points outside the rectangle - out of plane
        REQUIRE_THAT(rect.distance(Point3D(0.5, -1.0, 0.0)),
                     Catch::Matchers::WithinAbs(std::sqrt(2.0), LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(-1.0, -1.0, 0.0)),
                     Catch::Matchers::WithinAbs(std::sqrt(3.0), LENGTH_TOL));
        REQUIRE_THAT(rect.distance(Point3D(0.5, 0.5, 0.0)),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));
    }

    SECTION("Rectangle - 2D 3D transformations") {
        const Point3D p1(0.0, 0.0, 1.0);
        const Point3D p2(1.0, 0.0, 1.0);
        const Point3D p3(0.0, 1.0, 1.0);
        const Rectangle rect(p1, p2, p3);

        auto p1_2d = rect.from_3d_to_2d(p1);
        auto p2_2d = rect.from_3d_to_2d(p2);
        auto p3_2d = rect.from_3d_to_2d(p3);

        // Check that the rectangle vertices are converted correctly
        REQUIRE(p1_2d.isApprox(Point2D(0.0, 0.0), LENGTH_TOL));
        REQUIRE(p2_2d.isApprox(Point2D(1.0, 0.0), LENGTH_TOL));
        REQUIRE(p3_2d.isApprox(Point2D(0.0, 1.0), LENGTH_TOL));
        REQUIRE(rect.from_3d_to_2d(Point3D(1.0, 1.0, 1.0))
                    .isApprox(Point2D(1.0, 1.0), LENGTH_TOL));

        // Check that the backwards conversion also works
        REQUIRE(rect.from_2d_to_3d(p1_2d).isApprox(p1, LENGTH_TOL));
        REQUIRE(rect.from_2d_to_3d(p2_2d).isApprox(p2, LENGTH_TOL));
        REQUIRE(rect.from_2d_to_3d(p3_2d).isApprox(p3, LENGTH_TOL));
    }
}

TEST_CASE("Quadrilateral Primitive", "[gmm][primitive][quadrilateral]") {
    using pycanha::gmm::Quadrilateral;
    SECTION("Check constructor and set/get methods") {
        Point3D p1(0.0, 0.0, 0.0);
        Point3D p2(1.0, 0.0, 0.0);
        Point3D p3(1.0, 1.0, 0.0);
        Point3D p4(0.0, 1.0, 0.0);
        Quadrilateral quad(p1, p2, p3, p4);

        REQUIRE(quad.get_p1() == p1);
        REQUIRE(quad.get_p2() == p2);
        REQUIRE(quad.get_p3() == p3);
        REQUIRE(quad.get_p4() == p4);

        p1.x() += 1.0;
        p2.x() += 1.0;
        p3.x() += 1.0;
        p4.x() += 1.0;

        REQUIRE(quad.get_p1() != p1);
        REQUIRE(quad.get_p2() != p2);
        REQUIRE(quad.get_p3() != p3);
        REQUIRE(quad.get_p4() != p4);

        quad.set_p1(p1);
        quad.set_p2(p2);
        quad.set_p3(p3);
        quad.set_p4(p4);

        REQUIRE(quad.get_p1() == p1);
        REQUIRE(quad.get_p2() == p2);
        REQUIRE(quad.get_p3() == p3);
        REQUIRE(quad.get_p4() == p4);
    }
    SECTION("Check valid Quad") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(1.0, 0.0, 0.0);
        const Point3D p3(1.0, 1.0, 0.0);
        const Point3D p4(0.0, 1.0, 0.0);
        const Quadrilateral quad(p1, p2, p3, p4);

        REQUIRE_THAT(quad.v1().norm(),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));

        REQUIRE_THAT(quad.v2().norm(),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));

        REQUIRE_THAT(quad.v1().dot(quad.v2()),
                     Catch::Matchers::WithinAbs(0.0, ANGLE_TOL));

        REQUIRE(quad.is_valid());
    }
    SECTION("Invalid Quad 1: p1 == p2") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(0.0, 0.0, 0.0);
        const Point3D p3(0.0, 1.0, 0.0);
        const Point3D p4(1.0, 0.0, 0.0);
        const Quadrilateral quad(p1, p2, p3, p4);

        REQUIRE(!quad.is_valid());
    }
    SECTION("Invalid Quad 2: Degenerated to a triangle") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(1.0, 0.0, 0.0);
        const Point3D p3(2.0, 0.0, 0.0);
        const Point3D p4(1.0, 1.0, 0.0);
        const Quadrilateral quad(p1, p2, p3, p4);

        REQUIRE(!quad.is_valid());
    }
    SECTION("Invalid Quad 3: Out-of-plane") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(1.0, 0.0, 0.0);
        const Point3D p3(1.0, 1.0, 1.0);
        const Point3D p4(0.0, 1.0, 0.0);
        const Quadrilateral quad(p1, p2, p3, p4);

        REQUIRE(!quad.is_valid());
    }
    SECTION("Invalid Quad 4: Two lines coincide") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(1.0, 0.0, 0.0);
        const Point3D p3(0.0, 0.5, 0.0);
        const Point3D p4(0.0, 1.0, 0.0);
        const Quadrilateral quad(p1, p2, p3, p4);

        REQUIRE(!quad.is_valid());
    }
    SECTION("Quadrilateral - Point distances") {
        const Point3D p1(0.0, 0.0, 1.0);
        const Point3D p2(2.0, 1.0, 1.0);
        const Point3D p3(3.0, 3.0, 1.0);
        const Point3D p4(1.0, 2.0, 1.0);
        const Quadrilateral quad(p1, p2, p3, p4);

        // Points at the vertices
        REQUIRE_THAT(quad.distance(p1),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(quad.distance(p2),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(quad.distance(p3),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(quad.distance(p4),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));

        // Points on the edges
        REQUIRE_THAT(quad.distance(p1 + (p2 - p1) * 0.5),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(quad.distance(p2 + (p3 - p2) * 0.5),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(quad.distance(p3 + (p4 - p3) * 0.5),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(quad.distance(p4 + (p1 - p4) * 0.5),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));

        // Point inside the quadrilateral
        REQUIRE_THAT(quad.distance(Point3D(2.0, 2.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));

        // Points outside the quadrilateral - same plane
        REQUIRE_THAT(quad.distance(Point3D(3.0, -1.0, 1.0)),
                     Catch::Matchers::WithinAbs(sqrt(5), LENGTH_TOL));
        REQUIRE_THAT(quad.distance(Point3D(-1.0, 3.0, 1.0)),
                     Catch::Matchers::WithinAbs(sqrt(5), LENGTH_TOL));
        REQUIRE_THAT(quad.distance(Point3D(-1.0, -1.0, 1.0)),
                     Catch::Matchers::WithinAbs(sqrt(2.0), LENGTH_TOL));
        REQUIRE_THAT(quad.distance(Point3D(4, 4, 1.0)),
                     Catch::Matchers::WithinAbs(sqrt(2), LENGTH_TOL));

        REQUIRE_THAT(quad.distance(Point3D(1.0, 0.0, 1.0)),
                     Catch::Matchers::WithinAbs(1.0 / sqrt(5), LENGTH_TOL));
        REQUIRE_THAT(quad.distance(Point3D(0.0, 1.0, 1.0)),
                     Catch::Matchers::WithinAbs(1.0 / sqrt(5), LENGTH_TOL));
        REQUIRE_THAT(quad.distance(Point3D(2.0, 3.0, 1.0)),
                     Catch::Matchers::WithinAbs(1.0 / sqrt(5), LENGTH_TOL));
        REQUIRE_THAT(quad.distance(Point3D(3.0, 2.0, 1.0)),
                     Catch::Matchers::WithinAbs(1.0 / sqrt(5), LENGTH_TOL));

        // Points outside the quadrilateral - out of plane
        REQUIRE_THAT(quad.distance(p1 + Vector3D(0.0, 0.0, -1.0)),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));
        REQUIRE_THAT(quad.distance(p2 + Vector3D(0.0, 0.0, -1.0)),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));
        REQUIRE_THAT(quad.distance(p3 + Vector3D(0.0, 0.0, -1.0)),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));
        REQUIRE_THAT(quad.distance(p4 + Vector3D(0.0, 0.0, -1.0)),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));

        REQUIRE_THAT(quad.distance(Point3D(1.0, 0.0, 0.0)),
                     Catch::Matchers::WithinAbs(sqrt(6.0 / 5.0), LENGTH_TOL));
        REQUIRE_THAT(quad.distance(Point3D(0.0, 1.0, 0.0)),
                     Catch::Matchers::WithinAbs(sqrt(6.0 / 5.0), LENGTH_TOL));
        REQUIRE_THAT(quad.distance(Point3D(2.0, 3.0, 0.0)),
                     Catch::Matchers::WithinAbs(sqrt(6.0 / 5.0), LENGTH_TOL));
        REQUIRE_THAT(quad.distance(Point3D(3.0, 2.0, 0.0)),
                     Catch::Matchers::WithinAbs(sqrt(6.0 / 5.0), LENGTH_TOL));
    }
    SECTION("Quadrilateral - 2D 3D transformations") {
        const Point3D p1(1.0, 2.0, 1.0);
        const Point3D p2(3.0, 3.0, 1.0);
        const Point3D p3(4.0, 5.0, 1.0);
        const Point3D p4(2.0, 4.0, 1.0);
        const Quadrilateral quad(p1, p2, p3, p4);

        auto p1_2d = quad.from_3d_to_2d(p1);
        auto p2_2d = quad.from_3d_to_2d(p2);
        auto p3_2d = quad.from_3d_to_2d(p3);
        auto p4_2d = quad.from_3d_to_2d(p4);

        // Expected 2D points
        const double theta = std::atan2(1.0, 2.0);
        const Eigen::Rotation2D<double> rot(-theta);  // NOLINT

        auto p1_2d_expected = rot * Point2D(0.0, 0.0);
        auto p2_2d_expected = rot * Point2D(2.0, 1.0);
        auto p3_2d_expected = rot * Point2D(3.0, 3.0);
        auto p4_2d_expected = rot * Point2D(1.0, 2.0);

        // Check that the quadrilateral vertices are converted correctly
        REQUIRE(p1_2d.isApprox(p1_2d_expected, LENGTH_TOL));
        REQUIRE(p2_2d.isApprox(p2_2d_expected, LENGTH_TOL));
        REQUIRE(p3_2d.isApprox(p3_2d_expected, LENGTH_TOL));
        REQUIRE(p4_2d.isApprox(p4_2d_expected, LENGTH_TOL));

        // Check that the backwards conversion also works
        REQUIRE(quad.from_2d_to_3d(p1_2d).isApprox(p1, LENGTH_TOL));
        REQUIRE(quad.from_2d_to_3d(p2_2d).isApprox(p2, LENGTH_TOL));
        REQUIRE(quad.from_2d_to_3d(p3_2d).isApprox(p3, LENGTH_TOL));
        REQUIRE(quad.from_2d_to_3d(p4_2d).isApprox(p4, LENGTH_TOL));
    }
}

TEST_CASE("Cylinder Primitive", "[gmm][primitive][cylinder]") {
    using pycanha::gmm::Cylinder;
    using std::numbers::pi;

    SECTION("Check constructor and set/get methods") {
        Point3D p1(0.0, 0.0, 0.0);
        Point3D p2(0.0, 0.0, 1.0);
        Point3D p3(1.0, 0.0, 0.0);
        const double radius = 1.0;
        const double start_angle = 0.0;
        const double end_angle = pi;
        Cylinder cyl(p1, p2, p3, radius, start_angle, end_angle);

        REQUIRE(cyl.get_p1() == p1);
        REQUIRE(cyl.get_p2() == p2);
        REQUIRE(cyl.get_p3() == p3);
        REQUIRE(cyl.get_radius() == radius);
        REQUIRE(cyl.get_start_angle() == start_angle);
        REQUIRE(cyl.get_end_angle() == end_angle);

        p1.z() += 1.0;
        p2.z() += 1.0;
        p3.x() += 1.0;

        REQUIRE(cyl.get_p1() != p1);
        REQUIRE(cyl.get_p2() != p2);
        REQUIRE(cyl.get_p3() != p3);

        cyl.set_p1(p1);
        cyl.set_p2(p2);
        cyl.set_p3(p3);
        cyl.set_radius(radius + 1.0);
        cyl.set_start_angle(start_angle + pi / 2.0);
        cyl.set_end_angle(end_angle + pi / 2.0);

        REQUIRE(cyl.get_p1() == p1);
        REQUIRE(cyl.get_p2() == p2);
        REQUIRE(cyl.get_p3() == p3);
        REQUIRE(cyl.get_radius() == radius + 1.0);
        REQUIRE(cyl.get_start_angle() == start_angle + pi / 2.0);
        REQUIRE(cyl.get_end_angle() == end_angle + pi / 2.0);
    }
    SECTION("Check valid Cylinder") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(1.0, 1.0, 0.0);
        const Point3D p3(1.0, -1.0, 0.0);
        const double radius = 1.0;
        const double start_angle = 0.0;
        const double end_angle = 2.0 * pi;
        const Cylinder cyl(p1, p2, p3, radius, start_angle, end_angle);

        REQUIRE(cyl.is_valid());
    }
    SECTION("Check invalid Cylinder - Coincidence points") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(0.0, 0.0, 0.0);
        const Point3D p3(1.0, 0.0, 0.0);
        const double radius = 1.0;
        const double start_angle = 0.0;
        const double end_angle = 2.0 * pi;
        const Cylinder cyl(p1, p2, p3, radius, start_angle, end_angle);

        REQUIRE(!cyl.is_valid());
    }

    SECTION("Check invalid Cylinder - Non-orthogonal directions") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(1.0, 1.0, 0.0);
        const Point3D p3(1.0, 0.0, 0.0);
        const double radius = 1.0;
        const double start_angle = 0.0;
        const double end_angle = 2.0 * pi;
        const Cylinder cyl(p1, p2, p3, radius, start_angle, end_angle);

        REQUIRE(!cyl.is_valid());
    }

    SECTION("Check invalid Cylinder - Invalid radius") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(1.0, 1.0, 0.0);
        const Point3D p3(1.0, 0.0, 0.0);
        const double invalid_radius_1 = 0.0;
        const double invalid_radius_2 = -1.0;
        const double start_angle = 0.0;
        const double end_angle = 2.0 * pi;
        const Cylinder cyl1(p1, p2, p3, invalid_radius_1, start_angle,
                            end_angle);
        const Cylinder cyl2(p1, p2, p3, invalid_radius_2, start_angle,
                            end_angle);

        REQUIRE(!cyl1.is_valid());
        REQUIRE(!cyl2.is_valid());
    }

    SECTION("Cylinder - Point distances") {
        const Point3D p1(0.0, 0.0, 1.0);
        const Point3D p2(0.0, 0.0, 5.0);
        const Point3D p3(0.1, 0.0, 1.0);
        const double radius = 1.0;
        const double start_angle = 0.0;
        const double end_angle = 2.0 * pi;
        const Cylinder cyl(p1, p2, p3, radius, start_angle, end_angle);

        // Points on the cylinder surface
        REQUIRE_THAT(cyl.distance(Point3D(1.0, 0.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(cyl.distance(Point3D(-1.0, 0.0, 1.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(cyl.distance(Point3D(1.0, 0.0, 5.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));
        REQUIRE_THAT(cyl.distance(Point3D(-1.0, 0.0, 5.0)),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));

        // Points on the cylinder axis
        REQUIRE_THAT(cyl.distance(p1),
                     Catch::Matchers::WithinAbs(radius, LENGTH_TOL));
        REQUIRE_THAT(cyl.distance(p2),
                     Catch::Matchers::WithinAbs(radius, LENGTH_TOL));
        REQUIRE_THAT(cyl.distance(p1 + (p2 - p1) * 0.5),
                     Catch::Matchers::WithinAbs(radius, LENGTH_TOL));
        REQUIRE_THAT(cyl.distance(p2 + (p2 - p1).normalized()),
                     Catch::Matchers::WithinAbs(std::sqrt(radius * radius + 1),
                                                LENGTH_TOL));
        REQUIRE_THAT(cyl.distance(p1 - (p2 - p1).normalized()),
                     Catch::Matchers::WithinAbs(std::sqrt(radius * radius + 1),
                                                LENGTH_TOL));
        // Points outside
        REQUIRE_THAT(cyl.distance(Point3D(0.0, 0.5, 0.0)),
                     Catch::Matchers::WithinAbs(std::sqrt(1.25), LENGTH_TOL));
        REQUIRE_THAT(cyl.distance(Point3D(0.5, 0.0, 0.0)),
                     Catch::Matchers::WithinAbs(std::sqrt(1.25), LENGTH_TOL));
        REQUIRE_THAT(cyl.distance(Point3D(2.0, 0.0, 0.0)),
                     Catch::Matchers::WithinAbs(std::sqrt(2), LENGTH_TOL));
        REQUIRE_THAT(cyl.distance(Point3D(0.0, 2.0, 3.0)),
                     Catch::Matchers::WithinAbs(1.0, LENGTH_TOL));
    }
    SECTION("Cylinder - 2D 3D transformations") {
        const Point3D p1_cyl(0.0, 0.0, 1.0);
        const Point3D p2_cyl(0.0, 0.0, 5.0);
        const Point3D p3_cyl(0.1, 0.0, 1.0);
        const double radius = 2.5;
        const double start_angle = 0.0;
        const double end_angle = 2.0 * pi;
        const Cylinder cyl(p1_cyl, p2_cyl, p3_cyl, radius, start_angle,
                           end_angle);

        const Point3D p1_3d(1.0 * radius, 0.0 * radius, 2.0);
        const Point3D p2_3d(0.0 * radius, 1.0 * radius, 3.0);
        const Point3D p3_3d(-1.0 * radius, 0.0 * radius, 4.0);
        const Point3D p4_3d(0.0 * radius, -1.0 * radius, 5.0);

        auto p1_2d = cyl.from_3d_to_2d(p1_3d);
        auto p2_2d = cyl.from_3d_to_2d(p2_3d);
        auto p3_2d = cyl.from_3d_to_2d(p3_3d);
        auto p4_2d = cyl.from_3d_to_2d(p4_3d);

        // Expected 2D points
        const Point2D p1_2d_expected(0.0, 1.0);
        const Point2D p2_2d_expected(radius * pi / 2.0, 2.0);
        const Point2D p3_2d_expected(radius * pi, 3.0);
        const Point2D p4_2d_expected(3.0 * radius * pi / 2.0, 4.0);

        // Check that the cylinder vertices are converted correctly
        REQUIRE(p1_2d.isApprox(p1_2d_expected, LENGTH_TOL));
        REQUIRE(p2_2d.isApprox(p2_2d_expected, LENGTH_TOL));
        REQUIRE(p3_2d.isApprox(p3_2d_expected, LENGTH_TOL));
        REQUIRE(p4_2d.isApprox(p4_2d_expected, LENGTH_TOL));

        // Check that the backwards conversion also works
        REQUIRE(cyl.from_2d_to_3d(p1_2d).isApprox(p1_3d, LENGTH_TOL));
        REQUIRE(cyl.from_2d_to_3d(p2_2d).isApprox(p2_3d, LENGTH_TOL));
        REQUIRE(cyl.from_2d_to_3d(p3_2d).isApprox(p3_3d, LENGTH_TOL));
        REQUIRE(cyl.from_2d_to_3d(p4_2d).isApprox(p4_3d, LENGTH_TOL));
    }
}

// NOLINTEND(bugprone-chained-comparison)
