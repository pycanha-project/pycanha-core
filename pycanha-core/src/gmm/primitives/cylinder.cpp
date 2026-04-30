#include "pycanha-core/gmm/primitives/cylinder.hpp"

#include <cmath>
#include <utility>

#include "detail.hpp"
#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

Cylinder::Cylinder(Point3D p1, Point3D p2, Point3D p3, double radius,
                   double start_angle, double end_angle) noexcept
    : _p1(std::move(p1)),
      _p2(std::move(p2)),
      _p3(std::move(p3)),
      _radius(radius),
      _start_angle(start_angle),
      _end_angle(end_angle) {}

const Point3D& Cylinder::p1() const noexcept { return _p1; }

const Point3D& Cylinder::p2() const noexcept { return _p2; }

const Point3D& Cylinder::p3() const noexcept { return _p3; }

double Cylinder::radius() const noexcept { return _radius; }

double Cylinder::start_angle() const noexcept { return _start_angle; }

double Cylinder::end_angle() const noexcept { return _end_angle; }

void Cylinder::set_p1(Point3D p1) noexcept { _p1 = std::move(p1); }

void Cylinder::set_p2(Point3D p2) noexcept { _p2 = std::move(p2); }

void Cylinder::set_p3(Point3D p3) noexcept { _p3 = std::move(p3); }

void Cylinder::set_radius(double radius) noexcept { _radius = radius; }

void Cylinder::set_start_angle(double start_angle) noexcept {
    _start_angle = start_angle;
}

void Cylinder::set_end_angle(double end_angle) noexcept {
    _end_angle = end_angle;
}

bool Cylinder::is_valid() const noexcept {
    const Vector3D axis = _p2 - _p1;
    const Vector3D reference = _p3 - _p1;
    return detail::has_nonzero_length(axis) &&
           detail::has_nonzero_length(reference) &&
           detail::are_orthogonal(axis, reference) && _radius > LENGTH_TOL &&
           detail::angle_span_is_valid(_start_angle, _end_angle);
}

Point2D Cylinder::to_uv(const Point3D& point) const {
    const Vector3D axis = detail::axis_direction(_p1, _p2);
    const Vector3D radial_reference = detail::radial_reference(_p1, _p3, axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const Vector3D delta = point - _p1;
    const double height = delta.dot(axis);
    const Vector3D radial = (delta - height * axis).normalized();
    const double theta =
        detail::angle_about_axis(radial, radial_reference, tangent);
    return {theta * _radius, height};
}

Point3D Cylinder::to_cartesian(const Point2D& uv) const {
    const Vector3D axis = detail::axis_direction(_p1, _p2);
    const Vector3D radial_reference = detail::radial_reference(_p1, _p3, axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const double theta = uv.x() / _radius;
    return _p1 + uv.y() * axis +
           _radius *
               (std::cos(theta) * radial_reference + std::sin(theta) * tangent);
}

Vector3D Cylinder::normal_at_uv(const Point2D& uv) const noexcept {
    const Vector3D axis = detail::axis_direction(_p1, _p2);
    const Vector3D radial_reference = detail::radial_reference(_p1, _p3, axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const double theta = uv.x() / _radius;
    return (std::cos(theta) * radial_reference + std::sin(theta) * tangent)
        .normalized();
}

double Cylinder::surface_area() const noexcept {
    return (_end_angle - _start_angle) * _radius * (_p2 - _p1).norm();
}

}  // namespace pycanha::gmm
