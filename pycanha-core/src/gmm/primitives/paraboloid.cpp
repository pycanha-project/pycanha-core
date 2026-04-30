#include "pycanha-core/gmm/primitives/paraboloid.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

#include "detail.hpp"
#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {
namespace {

[[nodiscard]] double paraboloid_height(const Paraboloid& paraboloid) noexcept {
    return (paraboloid.p2() - paraboloid.p1()).norm();
}

[[nodiscard]] double paraboloid_radius_at_height(const Paraboloid& paraboloid,
                                                 double height) noexcept {
    const double total_height = paraboloid_height(paraboloid);
    if (total_height <= LENGTH_TOL) {
        return 0.0;
    }
    return paraboloid.radius() *
           std::sqrt(std::max(height, 0.0) / total_height);
}

}  // namespace

Paraboloid::Paraboloid(Point3D p1, Point3D p2, Point3D p3, double radius,
                       double start_angle, double end_angle) noexcept
    : _p1(std::move(p1)),
      _p2(std::move(p2)),
      _p3(std::move(p3)),
      _radius(radius),
      _start_angle(start_angle),
      _end_angle(end_angle) {}

const Point3D& Paraboloid::p1() const noexcept { return _p1; }

const Point3D& Paraboloid::p2() const noexcept { return _p2; }

const Point3D& Paraboloid::p3() const noexcept { return _p3; }

double Paraboloid::radius() const noexcept { return _radius; }

double Paraboloid::start_angle() const noexcept { return _start_angle; }

double Paraboloid::end_angle() const noexcept { return _end_angle; }

void Paraboloid::set_p1(Point3D p1) noexcept { _p1 = std::move(p1); }

void Paraboloid::set_p2(Point3D p2) noexcept { _p2 = std::move(p2); }

void Paraboloid::set_p3(Point3D p3) noexcept { _p3 = std::move(p3); }

void Paraboloid::set_radius(double radius) noexcept { _radius = radius; }

void Paraboloid::set_start_angle(double start_angle) noexcept {
    _start_angle = start_angle;
}

void Paraboloid::set_end_angle(double end_angle) noexcept {
    _end_angle = end_angle;
}

bool Paraboloid::is_valid() const noexcept {
    const Vector3D axis = _p2 - _p1;
    const Vector3D reference = _p3 - _p1;
    return detail::has_nonzero_length(axis) &&
           detail::has_nonzero_length(reference) &&
           detail::are_orthogonal(axis, reference) && _radius > LENGTH_TOL &&
           detail::angle_span_is_valid(_start_angle, _end_angle);
}

Point2D Paraboloid::to_uv(const Point3D& point) const {
    const Vector3D axis = detail::axis_direction(_p1, _p2);
    const Vector3D radial_reference = detail::radial_reference(_p1, _p3, axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const Vector3D delta = point - _p1;
    const double height = delta.dot(axis);
    const Vector3D radial_vector = delta - height * axis;
    const double radial_distance = radial_vector.norm();
    const Vector3D radial = radial_distance > LENGTH_TOL
                                ? radial_vector / radial_distance
                                : radial_reference;
    const double theta =
        detail::angle_about_axis(radial, radial_reference, tangent);
    return {theta * radial_distance, height};
}

Point3D Paraboloid::to_cartesian(const Point2D& uv) const {
    const Vector3D axis = detail::axis_direction(_p1, _p2);
    const Vector3D radial_reference = detail::radial_reference(_p1, _p3, axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const double height = uv.y();
    const double radial_distance = paraboloid_radius_at_height(*this, height);
    const double theta =
        radial_distance > LENGTH_TOL ? uv.x() / radial_distance : 0.0;
    return _p1 + height * axis +
           radial_distance *
               (std::cos(theta) * radial_reference + std::sin(theta) * tangent);
}

Vector3D Paraboloid::normal_at_uv(const Point2D& uv) const noexcept {
    const Vector3D axis = detail::axis_direction(_p1, _p2);
    const Vector3D radial_reference = detail::radial_reference(_p1, _p3, axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const double height = uv.y();
    const double radial_distance = paraboloid_radius_at_height(*this, height);
    if (radial_distance <= LENGTH_TOL) {
        return (-axis).normalized();
    }
    const double theta = uv.x() / radial_distance;
    const Vector3D radial =
        (std::cos(theta) * radial_reference + std::sin(theta) * tangent)
            .normalized();
    const double coefficient = (_radius * _radius) / paraboloid_height(*this);
    return (2.0 * radial_distance * radial - coefficient * axis).normalized();
}

double Paraboloid::surface_area() const noexcept {
    const double height = paraboloid_height(*this);
    const double radius_squared = _radius * _radius;
    const double full_surface_area =
        std::numbers::pi * _radius *
        (std::pow(radius_squared + 4.0 * height * height, 1.5) -
         std::pow(_radius, 3.0)) /
        (6.0 * height * height);
    return full_surface_area *
           ((_end_angle - _start_angle) / (2.0 * std::numbers::pi));
}

}  // namespace pycanha::gmm
