#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "pycanha-core/gmm/primitives.hpp"

using pycanha::Point3D;
using pycanha::gmm::Quadrilateral;

TEST_CASE("Quadrilateral Primitive", "[gmm]") {
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
}
