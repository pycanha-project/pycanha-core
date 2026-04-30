#include "pycanha-core/gmm/primitives/cone.hpp"

#include <cmath>
#include <utility>

#include "detail.hpp"
#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {
namespace {

[[nodiscard]] double cone_height(const Cone& cone) noexcept {
    return (cone.p2() - cone.p1()).norm();
}

[[nodiscard]] double cone_radius_at_height(const Cone& cone,
                                           double height) noexcept {
    const double total_height = cone_height(cone);
    if (total_height <= LENGTH_TOL) {
        return cone.radius1();
    }
    return cone.radius1() +
           (cone.radius2() - cone.radius1()) * (height / total_height);
}

}  // namespace

Cone::Cone(Point3D p1, Point3D p2, Point3D p3, double radius1, double radius2,
           double start_angle, double end_angle) noexcept
    : _p1(std::move(p1)),
      _p2(std::move(p2)),
      _p3(std::move(p3)),
      _radius1(radius1),
      _radius2(radius2),
      _start_angle(start_angle),
      _end_angle(end_angle) {}

const Point3D& Cone::p1() const noexcept { return _p1; }

const Point3D& Cone::p2() const noexcept { return _p2; }

const Point3D& Cone::p3() const noexcept { return _p3; }

double Cone::radius1() const noexcept { return _radius1; }

double Cone::radius2() const noexcept { return _radius2; }

double Cone::start_angle() const noexcept { return _start_angle; }

double Cone::end_angle() const noexcept { return _end_angle; }

void Cone::set_p1(Point3D p1) noexcept { _p1 = std::move(p1); }

void Cone::set_p2(Point3D p2) noexcept { _p2 = std::move(p2); }

void Cone::set_p3(Point3D p3) noexcept { _p3 = std::move(p3); }

void Cone::set_radius1(double radius1) noexcept { _radius1 = radius1; }

void Cone::set_radius2(double radius2) noexcept { _radius2 = radius2; }

void Cone::set_start_angle(double start_angle) noexcept {
    _start_angle = start_angle;
}

void Cone::set_end_angle(double end_angle) noexcept { _end_angle = end_angle; }

bool Cone::is_valid() const noexcept {
    const Vector3D axis = _p2 - _p1;
    const Vector3D reference = _p3 - _p1;
    return detail::has_nonzero_length(axis) &&
           detail::has_nonzero_length(reference) &&
           detail::are_orthogonal(axis, reference) && _radius1 >= 0.0 &&
           _radius2 >= 0.0 &&
           (_radius1 > LENGTH_TOL || _radius2 > LENGTH_TOL) &&
           detail::angle_span_is_valid(_start_angle, _end_angle);
}

Point2D Cone::to_uv(const Point3D& point) const {
    const Vector3D axis = detail::axis_direction(_p1, _p2);
    const Vector3D radial_reference = detail::radial_reference(_p1, _p3, axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const Vector3D delta = point - _p1;
    const double height = delta.dot(axis);
    const Vector3D radial_vector = delta - height * axis;
    const double radius = radial_vector.norm();
    const Vector3D radial =
        radius > LENGTH_TOL ? radial_vector / radius : radial_reference;
    const double theta =
        detail::angle_about_axis(radial, radial_reference, tangent);
    return {theta * radius, height};
}

Point3D Cone::to_cartesian(const Point2D& uv) const {
    const Vector3D axis = detail::axis_direction(_p1, _p2);
    const Vector3D radial_reference = detail::radial_reference(_p1, _p3, axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const double height = uv.y();
    const double radius = cone_radius_at_height(*this, height);
    const double theta = radius > LENGTH_TOL ? uv.x() / radius : 0.0;
    return _p1 + height * axis +
           radius *
               (std::cos(theta) * radial_reference + std::sin(theta) * tangent);
}

Vector3D Cone::normal_at_uv(const Point2D& uv) const noexcept {
    const Vector3D axis = detail::axis_direction(_p1, _p2);
    const Vector3D radial_reference = detail::radial_reference(_p1, _p3, axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const double height = uv.y();
    const double radius = cone_radius_at_height(*this, height);
    const double theta = radius > LENGTH_TOL ? uv.x() / radius : 0.0;
    const Vector3D radial =
        (std::cos(theta) * radial_reference + std::sin(theta) * tangent)
            .normalized();
    const double slope = (cone_height(*this) > LENGTH_TOL)
                             ? (_radius2 - _radius1) / cone_height(*this)
                             : 0.0;
    return (radial - slope * axis).normalized();
}

double Cone::surface_area() const noexcept {
    const double height = cone_height(*this);
    const double radius_delta = _radius2 - _radius1;
    const double slant =
        std::sqrt(height * height + radius_delta * radius_delta);
    return 0.5 * (_end_angle - _start_angle) * (_radius1 + _radius2) * slant;
}

}  // namespace pycanha::gmm
