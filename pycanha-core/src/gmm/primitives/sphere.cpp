#include "pycanha-core/gmm/primitives/sphere.hpp"

#include <cmath>
#include <utility>

#include "detail.hpp"
#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {
namespace {

[[nodiscard]] Point3D sphere_point_from_local(const Sphere& sphere,
                                              double longitude,
                                              double latitude) {
    const auto frame =
        detail::make_sphere_frame(sphere.p1(), sphere.p2(), sphere.p3());
    return detail::from_local_spherical(sphere.p1(), frame, sphere.radius(),
                                        longitude, latitude);
}

}  // namespace

Sphere::Sphere(Point3D p1, Point3D p2, Point3D p3, double radius,
               double base_truncation, double apex_truncation,
               double start_angle, double end_angle) noexcept
    : _p1(std::move(p1)),
      _p2(std::move(p2)),
      _p3(std::move(p3)),
      _radius(radius),
      _base_truncation(base_truncation),
      _apex_truncation(apex_truncation),
      _start_angle(start_angle),
      _end_angle(end_angle) {}

const Point3D& Sphere::p1() const noexcept { return _p1; }

const Point3D& Sphere::p2() const noexcept { return _p2; }

const Point3D& Sphere::p3() const noexcept { return _p3; }

double Sphere::radius() const noexcept { return _radius; }

double Sphere::base_truncation() const noexcept { return _base_truncation; }

double Sphere::apex_truncation() const noexcept { return _apex_truncation; }

double Sphere::start_angle() const noexcept { return _start_angle; }

double Sphere::end_angle() const noexcept { return _end_angle; }

void Sphere::set_p1(Point3D p1) noexcept { _p1 = std::move(p1); }

void Sphere::set_p2(Point3D p2) noexcept { _p2 = std::move(p2); }

void Sphere::set_p3(Point3D p3) noexcept { _p3 = std::move(p3); }

void Sphere::set_radius(double radius) noexcept { _radius = radius; }

void Sphere::set_base_truncation(double base_truncation) noexcept {
    _base_truncation = base_truncation;
}

void Sphere::set_apex_truncation(double apex_truncation) noexcept {
    _apex_truncation = apex_truncation;
}

void Sphere::set_start_angle(double start_angle) noexcept {
    _start_angle = start_angle;
}

void Sphere::set_end_angle(double end_angle) noexcept {
    _end_angle = end_angle;
}

bool Sphere::is_valid() const noexcept {
    const Vector3D axis = _p2 - _p1;
    const Vector3D reference = _p3 - _p1;
    return detail::has_nonzero_length(axis) &&
           detail::has_nonzero_length(reference) &&
           !detail::are_orthogonal(axis.cross(reference), Vector3D::Zero()) &&
           std::abs(axis.normalized().dot(reference.normalized())) <
               1.0 - ANGLE_TOL &&
           _radius > LENGTH_TOL && _base_truncation >= -_radius &&
           _apex_truncation <= _radius &&
           _base_truncation < _apex_truncation - LENGTH_TOL &&
           detail::angle_span_is_valid(_start_angle, _end_angle);
}

Point2D Sphere::to_uv(const Point3D& point) const {
    const auto frame = detail::make_sphere_frame(_p1, _p2, _p3);
    const Vector3D delta = point - _p1;
    const double local_x = delta.dot(frame.ref);
    const double local_y = delta.dot(frame.tangent);
    const double local_z = delta.dot(frame.axis);
    const double longitude = std::atan2(local_y, local_x);
    const double latitude = std::asin(detail::clamp_unit(local_z / _radius));
    return {_radius * longitude * std::cos(latitude), _radius * latitude};
}

Point3D Sphere::to_cartesian(const Point2D& uv) const {
    const double latitude = uv.y() / _radius;
    const double cos_latitude = std::cos(latitude);
    const double longitude = std::abs(cos_latitude) > LENGTH_TOL
                                 ? uv.x() / (_radius * cos_latitude)
                                 : 0.0;
    return sphere_point_from_local(*this, longitude, latitude);
}

Vector3D Sphere::normal_at_uv(const Point2D& uv) const noexcept {
    return (to_cartesian(uv) - _p1).normalized();
}

double Sphere::surface_area() const noexcept {
    return (_end_angle - _start_angle) * _radius *
           (_apex_truncation - _base_truncation);
}

}  // namespace pycanha::gmm
