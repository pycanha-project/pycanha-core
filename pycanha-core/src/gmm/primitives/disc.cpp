#include "pycanha-core/gmm/primitives/disc.hpp"

#include <cmath>
#include <utility>

#include "detail.hpp"
#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

Disc::Disc(Point3D p1, Point3D p2, Point3D p3, double inner_radius,
           double outer_radius, double start_angle, double end_angle) noexcept
    : _p1(std::move(p1)),
      _p2(std::move(p2)),
      _p3(std::move(p3)),
      _inner_radius(inner_radius),
      _outer_radius(outer_radius),
      _start_angle(start_angle),
      _end_angle(end_angle) {}

const Point3D& Disc::p1() const noexcept { return _p1; }

const Point3D& Disc::p2() const noexcept { return _p2; }

const Point3D& Disc::p3() const noexcept { return _p3; }

double Disc::inner_radius() const noexcept { return _inner_radius; }

double Disc::outer_radius() const noexcept { return _outer_radius; }

double Disc::start_angle() const noexcept { return _start_angle; }

double Disc::end_angle() const noexcept { return _end_angle; }

void Disc::set_p1(Point3D p1) noexcept { _p1 = std::move(p1); }

void Disc::set_p2(Point3D p2) noexcept { _p2 = std::move(p2); }

void Disc::set_p3(Point3D p3) noexcept { _p3 = std::move(p3); }

void Disc::set_inner_radius(double inner_radius) noexcept {
    _inner_radius = inner_radius;
}

void Disc::set_outer_radius(double outer_radius) noexcept {
    _outer_radius = outer_radius;
}

void Disc::set_start_angle(double start_angle) noexcept {
    _start_angle = start_angle;
}

void Disc::set_end_angle(double end_angle) noexcept { _end_angle = end_angle; }

bool Disc::is_valid() const noexcept {
    const Vector3D axis = _p2 - _p1;
    const Vector3D reference = _p3 - _p1;
    return detail::has_nonzero_length(axis) &&
           detail::has_nonzero_length(reference) &&
           detail::are_orthogonal(axis, reference) && _inner_radius >= 0.0 &&
           _outer_radius > _inner_radius + LENGTH_TOL &&
           detail::angle_span_is_valid(_start_angle, _end_angle);
}

Point2D Disc::to_uv(const Point3D& point) const {
    const Vector3D axis = detail::axis_direction(_p1, _p2);
    const Vector3D radial_reference = detail::radial_reference(_p1, _p3, axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const Vector3D delta = point - _p1;
    const double radius = delta.norm();
    const Vector3D radial =
        radius > LENGTH_TOL ? delta.normalized() : radial_reference;
    const double theta =
        detail::angle_about_axis(radial, radial_reference, tangent);
    return {theta * radius, radius};
}

Point3D Disc::to_cartesian(const Point2D& uv) const {
    const Vector3D axis = detail::axis_direction(_p1, _p2);
    const Vector3D radial_reference = detail::radial_reference(_p1, _p3, axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const double radius = uv.y();
    const double theta = radius > LENGTH_TOL ? uv.x() / radius : 0.0;
    return _p1 + radius * (std::cos(theta) * radial_reference +
                           std::sin(theta) * tangent);
}

Vector3D Disc::normal_at_uv(const Point2D& /*uv*/) const noexcept {
    return (_p2 - _p1).normalized();
}

double Disc::surface_area() const noexcept {
    return 0.5 * (_end_angle - _start_angle) *
           ((_outer_radius * _outer_radius) - (_inner_radius * _inner_radius));
}

}  // namespace pycanha::gmm
