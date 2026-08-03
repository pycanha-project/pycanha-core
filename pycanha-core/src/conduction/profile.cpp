#include "pycanha-core/conduction/profile.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>
#include <variant>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"

namespace pycanha::conduction {

namespace {

constexpr double infinite_potential = std::numeric_limits<double>::infinity();

[[nodiscard]] bool spans_full_revolution(double start_angle,
                                         double end_angle) noexcept {
    constexpr double full_turn = 2.0 * std::numbers::pi;
    return std::abs((end_angle - start_angle) - full_turn) <= ANGLE_TOL;
}

[[nodiscard]] double lerp(double start, double end, double fraction) noexcept {
    return start + ((end - start) * fraction);
}

// Inverse Gudermannian: the meridian potential of a sphere, integral of
// dphi / cos(phi). Unbounded at either pole, which on_axis() screens first.
[[nodiscard]] double inverse_gudermannian(double latitude) noexcept {
    constexpr double quarter_pi = std::numbers::pi / 4.0;
    return std::log(std::tan((latitude / 2.0) + quarter_pi));
}

}  // namespace

MeridianProfile MeridianProfile::make_planar(double dir1_length,
                                             double dir2_length) {
    MeridianProfile profile;
    profile._kind = Kind::Planar;
    profile._dir1_extent = dir1_length;
    profile._dir2_start = 0.0;
    profile._dir2_end = dir2_length;
    return profile;
}

MeridianProfile MeridianProfile::make_disc(double angle_span, bool closes_ring,
                                           double inner_radius,
                                           double outer_radius) {
    MeridianProfile profile;
    profile._kind = Kind::Disc;
    profile._dir1_extent = angle_span;
    profile._closes_ring = closes_ring;
    profile._dir2_start = inner_radius;
    profile._dir2_end = outer_radius;
    return profile;
}

MeridianProfile MeridianProfile::make_cylinder(double angle_span,
                                               bool closes_ring, double radius,
                                               double height) {
    MeridianProfile profile;
    profile._kind = Kind::Cylinder;
    profile._dir1_extent = angle_span;
    profile._closes_ring = closes_ring;
    profile._radius = radius;
    profile._dir2_start = 0.0;
    profile._dir2_end = height;
    profile._height = height;
    return profile;
}

MeridianProfile MeridianProfile::make_cone(double angle_span, bool closes_ring,
                                           double radius1, double radius2,
                                           double height) {
    MeridianProfile profile;
    profile._kind = Kind::Cone;
    profile._dir1_extent = angle_span;
    profile._closes_ring = closes_ring;
    profile._radius = radius1;
    profile._radius_end = radius2;
    profile._height = height;
    profile._dir2_start = 0.0;
    profile._dir2_end = height;
    // The mesher walks the cone by axial height, so one unit of the native
    // parameter covers sqrt(1 + (dr/dh)^2) units of meridian arc.
    const double slope =
        height > LENGTH_TOL ? (radius2 - radius1) / height : 0.0;
    profile._slant = std::sqrt(1.0 + (slope * slope));
    return profile;
}

MeridianProfile MeridianProfile::make_sphere(double angle_span,
                                             bool closes_ring, double radius,
                                             double min_latitude,
                                             double max_latitude) {
    MeridianProfile profile;
    profile._kind = Kind::Sphere;
    profile._dir1_extent = angle_span;
    profile._closes_ring = closes_ring;
    profile._radius = radius;
    profile._dir2_start = min_latitude;
    profile._dir2_end = max_latitude;
    return profile;
}

MeridianProfile MeridianProfile::make_paraboloid(double angle_span,
                                                 bool closes_ring,
                                                 double radius, double height) {
    MeridianProfile profile;
    profile._kind = Kind::Paraboloid;
    profile._dir1_extent = angle_span;
    profile._closes_ring = closes_ring;
    profile._radius = radius;
    profile._height = height;
    profile._dir2_start = 0.0;
    profile._dir2_end = height;
    return profile;
}

double MeridianProfile::dir1_coordinate(double fraction) const noexcept {
    return fraction * _dir1_extent;
}

double MeridianProfile::rho(double fraction) const noexcept {
    switch (_kind) {
        case Kind::Planar:
            return 1.0;
        case Kind::Disc:
            return lerp(_dir2_start, _dir2_end, fraction);
        case Kind::Cylinder:
            return _radius;
        case Kind::Cone:
            return lerp(_radius, _radius_end, fraction);
        case Kind::Sphere:
            return _radius * std::cos(lerp(_dir2_start, _dir2_end, fraction));
        case Kind::Paraboloid:
            return _radius * std::sqrt(std::max(fraction, 0.0));
    }
    return 1.0;
}

bool MeridianProfile::on_axis(double fraction) const noexcept {
    return rho(fraction) <= LENGTH_TOL;
}

double MeridianProfile::potential(double fraction) const noexcept {
    if (on_axis(fraction)) {
        // Approaching the axis the meridian keeps its length while the
        // circumference vanishes, so the potential runs off logarithmically.
        return -infinite_potential;
    }

    switch (_kind) {
        case Kind::Planar:
            return lerp(_dir2_start, _dir2_end, fraction);
        case Kind::Disc:
            return std::log(lerp(_dir2_start, _dir2_end, fraction));
        case Kind::Cylinder:
            return (fraction * _height) / _radius;
        case Kind::Cone: {
            const double delta_radius = _radius_end - _radius;
            const double scale =
                std::max(std::abs(_radius), std::abs(_radius_end));
            if (std::abs(delta_radius) <= LENGTH_TOL * std::max(scale, 1.0)) {
                // A cone of constant radius is a cylinder; the general form is
                // 0/0 there.
                return (_slant * fraction * _height) / _radius;
            }
            return (_slant * _height / delta_radius) *
                   std::log(lerp(_radius, _radius_end, fraction));
        }
        case Kind::Sphere:
            return inverse_gudermannian(lerp(_dir2_start, _dir2_end, fraction));
        case Kind::Paraboloid: {
            // With h = height * u^2 the meridian potential integrates in
            // closed form: integral of sqrt(u^2 + a^2) / (a * u) du, where
            // a = radius / (2 * height) is the paraboloid's apex slope.
            const double apex_slope = _radius / (2.0 * _height);
            const double param = std::sqrt(fraction);
            const double hypot = std::hypot(param, apex_slope);
            return (hypot -
                    (apex_slope * std::log((apex_slope + hypot) / param))) /
                   apex_slope;
        }
    }
    return 0.0;
}

double MeridianProfile::meridian_length(double low,
                                        double high) const noexcept {
    const double span = high - low;
    switch (_kind) {
        case Kind::Planar:
        case Kind::Disc:
            return std::abs(span * (_dir2_end - _dir2_start));
        case Kind::Cylinder:
            return std::abs(span * _height);
        case Kind::Cone:
            return std::abs(_slant * span * _height);
        case Kind::Sphere:
            return std::abs(_radius * span * (_dir2_end - _dir2_start));
        case Kind::Paraboloid: {
            if (_height <= LENGTH_TOL) {
                // A paraboloid of no height is a flat disc of radius R, whose
                // meridian is the radius itself.
                return std::abs(_radius * (std::sqrt(std::max(high, 0.0)) -
                                           std::sqrt(std::max(low, 0.0))));
            }
            // With h = height * u^2 the arc element is 2*height*hypot(u, a) du,
            // a = radius / (2 * height), which integrates in closed form.
            const double apex_slope = _radius / (2.0 * _height);
            const auto arc = [&](double fraction) {
                const double param = std::sqrt(std::max(fraction, 0.0));
                const double hypot = std::hypot(param, apex_slope);
                return _height *
                       ((param * hypot) + (apex_slope * apex_slope *
                                           std::asinh(param / apex_slope)));
            };
            return std::abs(arc(high) - arc(low));
        }
    }
    return 0.0;
}

std::optional<MeridianProfile> profile_of(const gmm::Primitive& primitive) {
    return std::visit(
        [](const auto& concrete) -> std::optional<MeridianProfile> {
            using T = std::decay_t<decltype(concrete)>;

            if constexpr (std::is_same_v<T, gmm::Rectangle>) {
                return MeridianProfile::make_planar(
                    (concrete.p2() - concrete.p1()).norm(),
                    concrete.to_uv(concrete.p3()).y());
            } else if constexpr (std::is_same_v<T, gmm::Quadrilateral>) {
                // Meshed as the equivalent rectangle spanned by p2 - p1 and
                // the orthogonal part of p4 - p1; p3 never enters the mesh.
                return MeridianProfile::make_planar(
                    (concrete.p2() - concrete.p1()).norm(),
                    concrete.to_uv(concrete.p4()).y());
            } else if constexpr (std::is_same_v<T, gmm::Disc>) {
                return MeridianProfile::make_disc(
                    concrete.end_angle() - concrete.start_angle(),
                    spans_full_revolution(concrete.start_angle(),
                                          concrete.end_angle()),
                    concrete.inner_radius(), concrete.outer_radius());
            } else if constexpr (std::is_same_v<T, gmm::Cylinder>) {
                return MeridianProfile::make_cylinder(
                    concrete.end_angle() - concrete.start_angle(),
                    spans_full_revolution(concrete.start_angle(),
                                          concrete.end_angle()),
                    concrete.radius(), (concrete.p2() - concrete.p1()).norm());
            } else if constexpr (std::is_same_v<T, gmm::Cone>) {
                return MeridianProfile::make_cone(
                    concrete.end_angle() - concrete.start_angle(),
                    spans_full_revolution(concrete.start_angle(),
                                          concrete.end_angle()),
                    concrete.radius1(), concrete.radius2(),
                    (concrete.p2() - concrete.p1()).norm());
            } else if constexpr (std::is_same_v<T, gmm::Sphere>) {
                return MeridianProfile::make_sphere(
                    concrete.end_angle() - concrete.start_angle(),
                    spans_full_revolution(concrete.start_angle(),
                                          concrete.end_angle()),
                    concrete.radius(),
                    std::asin(concrete.base_truncation() / concrete.radius()),
                    std::asin(concrete.apex_truncation() / concrete.radius()));
            } else if constexpr (std::is_same_v<T, gmm::Paraboloid>) {
                return MeridianProfile::make_paraboloid(
                    concrete.end_angle() - concrete.start_angle(),
                    spans_full_revolution(concrete.start_angle(),
                                          concrete.end_angle()),
                    concrete.radius(), (concrete.p2() - concrete.p1()).norm());
            } else {
                // Triangle (fan parametrisation, discrete fallback) and Cube
                // (cutter-only, never meshed).
                return std::nullopt;
            }
        },
        primitive);
}

}  // namespace pycanha::conduction
