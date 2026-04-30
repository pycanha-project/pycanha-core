#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "test_helpers.hpp"

namespace {

using pycanha::Point2D;
using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::Sphere;
namespace tests = pycanha::gmm::tests;

}  // namespace

TEST_CASE("Sphere supports truncated surface properties",
          "[gmm][primitive][sphere]") {
    using std::numbers::pi;

    Sphere sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 2.0, -1.0,
                  1.0, 0.0, 2.0 * pi);

    SECTION("constructor and setters keep values") {
        REQUIRE(sphere.radius() == Catch::Approx(2.0));
        REQUIRE(sphere.base_truncation() == Catch::Approx(-1.0));
        REQUIRE(sphere.apex_truncation() == Catch::Approx(1.0));

        sphere.set_radius(1.5);
        sphere.set_base_truncation(-0.5);
        sphere.set_apex_truncation(0.5);

        REQUIRE(sphere.radius() == Catch::Approx(1.5));
        REQUIRE(sphere.base_truncation() == Catch::Approx(-0.5));
        REQUIRE(sphere.apex_truncation() == Catch::Approx(0.5));
    }

    SECTION("validity enforces orientation and truncation bounds") {
        REQUIRE(sphere.is_valid());
        REQUIRE_FALSE(Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 2.0},
                             2.0, -1.0, 1.0, 0.0, 2.0 * pi)
                          .is_valid());
        REQUIRE_FALSE(Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0},
                             2.0, 1.0, -1.0, 0.0, 2.0 * pi)
                          .is_valid());
    }

    SECTION("surface area matches the spherical-zone formula") {
        REQUIRE(sphere.surface_area() == Catch::Approx(8.0 * pi));
    }

    SECTION("normal is radial") {
        const Point2D uv = sphere.to_uv(Point3D(2.0, 0.0, 0.0));
        tests::require_parallel(sphere.normal_at_uv(uv),
                                Vector3D(1.0, 0.0, 0.0));
    }

    SECTION("sphere round-trip remains a documented gap") {
        // TODO: Keep the Sphere round-trip gap visible until the long-lat
        // parametrization is finalized across meshing and plotting consumers.
        SUCCEED();
    }
}
