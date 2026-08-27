#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <functional>
#include <numbers>
#include <optional>
#include <stdexcept>

#include "pycanha-core/conduction/profile.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"

namespace {

using pycanha::conduction::MeridianProfile;
using pycanha::conduction::profile_of;
using pycanha::gmm::Cone;
using pycanha::gmm::Cube;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Disc;
using pycanha::gmm::Paraboloid;
using pycanha::gmm::Quadrilateral;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Sphere;
using pycanha::gmm::Triangle;

constexpr double pi = std::numbers::pi;

// profile_of returns an optional because a Triangle, a Quadrilateral and a
// Cube have no closed form; every other primitive must have one, so unwrap it
// once here.
[[nodiscard]] MeridianProfile require_profile(
    const pycanha::gmm::Primitive& primitive) {
    std::optional<MeridianProfile> profile = profile_of(primitive);
    if (!profile.has_value()) {
        throw std::logic_error(
            "primitive has no closed-form conduction profile");
    }
    return *profile;
}

// Composite Simpson of a smooth integrand: the reference the closed forms are
// checked against. The integrands below are the raw definition dl2 / rho, so
// agreement means the analytic potential really is that integral.
[[nodiscard]] double integrate(const std::function<double(double)>& integrand,
                               double low, double high) {
    constexpr int intervals = 20000;  // even, so the Simpson pattern closes
    const double step = (high - low) / intervals;
    double total = integrand(low) + integrand(high);
    for (int index = 1; index < intervals; ++index) {
        const double weight = index % 2 == 0 ? 2.0 : 4.0;
        total += weight * integrand(low + (index * step));
    }
    return total * step / 3.0;
}

// Every potential is defined up to an additive constant, so only differences
// are meaningful.
void check_potential(const MeridianProfile& profile,
                     const std::function<double(double)>& integrand, double low,
                     double high, double tolerance = 1e-10) {
    const double analytic = profile.potential(high) - profile.potential(low);
    const double numeric = integrate(integrand, low, high);
    REQUIRE(analytic == Catch::Approx(numeric).epsilon(0.0).margin(tolerance));
}

}  // namespace

TEST_CASE("profile: a rectangle is the degenerate rho == 1 case",
          "[conduction][profile]") {
    const Rectangle rectangle({0.0, 0.0, 0.0}, {3.0, 0.0, 0.0},
                              {0.0, 2.0, 0.0});
    const MeridianProfile profile = require_profile(rectangle);
    REQUIRE(profile.kind() == MeridianProfile::Kind::Planar);
    REQUIRE_FALSE(profile.closes_ring());

    REQUIRE(profile.rho(0.3) == Catch::Approx(1.0));
    REQUIRE(profile.dir1_coordinate(1.0) == Catch::Approx(3.0));
    REQUIRE(profile.dir1_coordinate(0.25) == Catch::Approx(0.75));
    REQUIRE(profile.potential(1.0) - profile.potential(0.0) ==
            Catch::Approx(2.0));
}

TEST_CASE("profile: a quadrilateral has no closed-form profile",
          "[conduction][profile]") {
    // A bilinear patch's faces change width along direction 2, so no
    // constant-width planar profile describes it. Treating one as its
    // "equivalent rectangle" -- the shape spanned by p2 - p1 and the
    // orthogonal part of p4 - p1 -- silently discards p3 and with it up to
    // half the area; the discrete shared-edge path handles it instead.
    const Quadrilateral quadrilateral({0.0, 0.0, 0.0}, {3.0, 0.0, 0.0},
                                      {2.5, 2.0, 0.0}, {0.0, 2.0, 0.0});
    REQUIRE_FALSE(profile_of(quadrilateral).has_value());
}

TEST_CASE("profile: disc potential is the log of the radius",
          "[conduction][profile]") {
    const double inner = 0.4;
    const double outer = 1.6;
    const Disc disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.6, 0.0, 0.0}, inner,
                    outer, 0.0, pi / 2.0);
    const MeridianProfile profile = require_profile(disc);
    REQUIRE(profile.dir1_extent() == Catch::Approx(pi / 2.0));
    REQUIRE_FALSE(profile.closes_ring());

    // The meridian is the radius itself, so dl2 / rho = dr / r.
    const auto integrand = [&](double fraction) {
        return (outer - inner) / (inner + (fraction * (outer - inner)));
    };
    check_potential(profile, integrand, 0.0, 1.0);
    check_potential(profile, integrand, 0.2, 0.7);
    REQUIRE(profile.potential(1.0) - profile.potential(0.0) ==
            Catch::Approx(std::log(outer / inner)));
}

TEST_CASE("profile: a full-revolution disc closes the ring",
          "[conduction][profile]") {
    const Disc disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 0.2, 1.0,
                    0.0, 2.0 * pi);
    const MeridianProfile profile = require_profile(disc);
    REQUIRE(profile.closes_ring());
}

TEST_CASE("profile: a disc reaching the centre is on the axis there",
          "[conduction][profile]") {
    const Disc disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 0.0, 1.0,
                    0.0, 2.0 * pi);
    const MeridianProfile profile = require_profile(disc);
    REQUIRE(profile.on_axis(0.0));
    REQUIRE_FALSE(profile.on_axis(1e-6));
    // The reference line of a face pair never sits on the axis, so the radial
    // conductance across the innermost band stays bounded.
    REQUIRE(std::isfinite(profile.potential(0.25)));
    REQUIRE(profile.potential(0.5) - profile.potential(0.25) ==
            Catch::Approx(std::log(2.0)));
}

TEST_CASE("profile: cylinder potential is the height over the radius",
          "[conduction][profile]") {
    const double radius = 1.2;
    const double height = 2.5;
    const Cylinder cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, height},
                            {radius, 0.0, 0.0}, radius, 0.0, 2.0 * pi);
    const MeridianProfile profile = require_profile(cylinder);
    REQUIRE(profile.closes_ring());
    REQUIRE(profile.rho(0.4) == Catch::Approx(radius));

    const auto integrand = [&](double /*fraction*/) { return height / radius; };
    check_potential(profile, integrand, 0.0, 1.0);
    REQUIRE(profile.potential(1.0) - profile.potential(0.0) ==
            Catch::Approx(height / radius));
}

TEST_CASE("profile: cone potential integrates along the slant",
          "[conduction][profile]") {
    const double radius1 = 0.5;
    const double radius2 = 1.5;
    const double height = 2.0;
    const Cone cone({0.0, 0.0, 0.0}, {0.0, 0.0, height}, {radius2, 0.0, 0.0},
                    radius1, radius2, 0.0, pi);
    const MeridianProfile profile = require_profile(cone);

    const double slant =
        std::hypot(height, radius2 - radius1);  // per unit of fraction
    const auto integrand = [&](double fraction) {
        return slant / (radius1 + (fraction * (radius2 - radius1)));
    };
    check_potential(profile, integrand, 0.0, 1.0);
    check_potential(profile, integrand, 0.1, 0.9);
}

TEST_CASE("profile: a cone of constant radius reduces to a cylinder",
          "[conduction][profile]") {
    const double radius = 0.8;
    const double height = 1.7;
    const Cone cone({0.0, 0.0, 0.0}, {0.0, 0.0, height}, {radius, 0.0, 0.0},
                    radius, radius, 0.0, pi);
    const Cylinder cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, height},
                            {radius, 0.0, 0.0}, radius, 0.0, pi);
    const MeridianProfile cone_profile = require_profile(cone);
    const MeridianProfile cylinder_profile = require_profile(cylinder);

    const double cone_span =
        cone_profile.potential(1.0) - cone_profile.potential(0.0);
    const double cylinder_span =
        cylinder_profile.potential(1.0) - cylinder_profile.potential(0.0);
    REQUIRE(cone_span == Catch::Approx(cylinder_span).epsilon(1e-12));

    // Approaching the cylinder from a slightly tapered cone must be smooth.
    const Cone almost({0.0, 0.0, 0.0}, {0.0, 0.0, height}, {radius, 0.0, 0.0},
                      radius, radius + 1e-4, 0.0, pi);
    const MeridianProfile almost_profile = require_profile(almost);
    REQUIRE(almost_profile.potential(1.0) - almost_profile.potential(0.0) ==
            Catch::Approx(cylinder_span).epsilon(1e-3));
}

TEST_CASE("profile: a cone with a zero end radius sits on the axis",
          "[conduction][profile]") {
    const Cone cone({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {1.5, 0.0, 0.0}, 0.0, 1.5,
                    0.0, 2.0 * pi);
    const MeridianProfile profile = require_profile(cone);
    REQUIRE(profile.on_axis(0.0));
    REQUIRE(std::isfinite(profile.potential(0.5)));
}

TEST_CASE("profile: sphere potential is the inverse Gudermannian",
          "[conduction][profile]") {
    const double radius = 1.0;
    const double base = -0.5;
    const double apex = 0.8;
    const Sphere sphere({0.0, 0.0, 0.0}, {0.0, 0.0, radius}, {radius, 0.0, 0.0},
                        radius, base, apex, 0.0, 2.0 * pi);
    const MeridianProfile profile = require_profile(sphere);
    REQUIRE(profile.closes_ring());

    const double min_latitude = std::asin(base / radius);
    const double latitude_span = std::asin(apex / radius) - min_latitude;
    // dl2 = R dphi and rho = R cos(phi), so the radius cancels.
    const auto integrand = [&](double fraction) {
        return latitude_span /
               std::cos(min_latitude + (fraction * latitude_span));
    };
    check_potential(profile, integrand, 0.0, 1.0);
    check_potential(profile, integrand, 0.25, 0.75);
}

TEST_CASE("profile: a sphere reaching a pole is on the axis there",
          "[conduction][profile]") {
    const Sphere sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 1.0,
                        -1.0, 1.0, 0.0, 2.0 * pi);
    const MeridianProfile profile = require_profile(sphere);
    REQUIRE(profile.on_axis(0.0));
    REQUIRE(profile.on_axis(1.0));
    REQUIRE(std::isfinite(profile.potential(0.5)));
}

TEST_CASE("profile: paraboloid closed form matches the quadrature",
          "[conduction][profile]") {
    const double radius = 1.5;
    const double height = 3.0;
    const Paraboloid paraboloid({0.0, 0.0, 0.0}, {0.0, 0.0, height},
                                {radius, 0.0, 0.0}, radius, 0.0, 2.0 * pi);
    const MeridianProfile profile = require_profile(paraboloid);
    REQUIRE(profile.on_axis(0.0));

    // r = radius * sqrt(f) and z = f * height, so the arc element per unit of
    // fraction is hypot(radius / (2 sqrt(f)), height).
    const auto integrand = [&](double fraction) {
        return std::hypot(radius / (2.0 * std::sqrt(fraction)), height) /
               (radius * std::sqrt(fraction));
    };
    check_potential(profile, integrand, 0.05, 1.0, 1e-8);
    check_potential(profile, integrand, 0.3, 0.9, 1e-10);
}

TEST_CASE("profile: meridian length matches the arc quadrature",
          "[conduction][profile]") {
    // The near-axis conductance needs the band's arc length rather than its
    // potential, so the closed forms are checked against the same quadrature.
    const double radius = 1.4;
    const double height = 2.2;

    const Disc disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 0.0,
                    radius, 0.0, 2.0 * pi);
    REQUIRE(require_profile(disc).meridian_length(0.0, 0.4) ==
            Catch::Approx(0.4 * radius));

    const Cone cone({0.0, 0.0, 0.0}, {0.0, 0.0, height}, {radius, 0.0, 0.0},
                    0.0, radius, 0.0, 2.0 * pi);
    REQUIRE(require_profile(cone).meridian_length(0.0, 1.0) ==
            Catch::Approx(std::hypot(height, radius)));

    const Sphere sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 1.0,
                        -1.0, 1.0, 0.0, 2.0 * pi);
    // Pole to pole is half a great circle.
    REQUIRE(require_profile(sphere).meridian_length(0.0, 1.0) ==
            Catch::Approx(pi));

    const Paraboloid paraboloid({0.0, 0.0, 0.0}, {0.0, 0.0, height},
                                {radius, 0.0, 0.0}, radius, 0.0, 2.0 * pi);
    // r = R*sqrt(f), z = f*H, so the arc element per unit fraction is
    // hypot(R / (2 sqrt(f)), H).
    const auto arc_element = [&](double fraction) {
        return std::hypot(radius / (2.0 * std::sqrt(fraction)), height);
    };
    REQUIRE(require_profile(paraboloid).meridian_length(0.02, 1.0) ==
            Catch::Approx(integrate(arc_element, 0.02, 1.0)).epsilon(1e-6));
}

TEST_CASE("profile: triangle and cube have no closed form",
          "[conduction][profile]") {
    const Triangle triangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    const Cube cube({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    REQUIRE_FALSE(profile_of(triangle).has_value());
    REQUIRE_FALSE(profile_of(cube).has_value());
}
