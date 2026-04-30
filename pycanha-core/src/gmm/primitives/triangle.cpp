#include "pycanha-core/gmm/primitives/triangle.hpp"

#include <utility>

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {
namespace {

[[nodiscard]] Vector3D triangle_edge_1(const Triangle& triangle) noexcept {
    return triangle.p2() - triangle.p1();
}

[[nodiscard]] Vector3D triangle_edge_2(const Triangle& triangle) noexcept {
    return triangle.p3() - triangle.p1();
}

[[nodiscard]] Vector3D triangle_normal_unnormalized(
    const Triangle& triangle) noexcept {
    return triangle_edge_1(triangle).cross(triangle_edge_2(triangle));
}

}  // namespace

Triangle::Triangle(Point3D p1, Point3D p2, Point3D p3) noexcept
    : _p1(std::move(p1)), _p2(std::move(p2)), _p3(std::move(p3)) {}

const Point3D& Triangle::p1() const noexcept { return _p1; }

const Point3D& Triangle::p2() const noexcept { return _p2; }

const Point3D& Triangle::p3() const noexcept { return _p3; }

void Triangle::set_p1(Point3D p1) noexcept { _p1 = std::move(p1); }

void Triangle::set_p2(Point3D p2) noexcept { _p2 = std::move(p2); }

void Triangle::set_p3(Point3D p3) noexcept { _p3 = std::move(p3); }

bool Triangle::is_valid() const noexcept {
    const Vector3D edge_1 = triangle_edge_1(*this);
    const Vector3D edge_2 = triangle_edge_2(*this);
    const Vector3D edge_3 = _p3 - _p2;

    return edge_1.norm() > LENGTH_TOL && edge_2.norm() > LENGTH_TOL &&
           edge_3.norm() > LENGTH_TOL &&
           triangle_normal_unnormalized(*this).norm() > LENGTH_TOL;
}

Point2D Triangle::to_uv(const Point3D& point) const {
    const Vector3D edge_1 = triangle_edge_1(*this);
    const Vector3D edge_2 = triangle_edge_2(*this);
    const Vector3D u_axis = edge_1.normalized();
    const Vector3D v_axis = (edge_2 - edge_2.dot(u_axis) * u_axis).normalized();
    const Vector3D delta = point - _p1;

    return {delta.dot(u_axis), delta.dot(v_axis)};
}

Point3D Triangle::to_cartesian(const Point2D& uv) const {
    const Vector3D edge_1 = triangle_edge_1(*this);
    const Vector3D edge_2 = triangle_edge_2(*this);
    const Vector3D u_axis = edge_1.normalized();
    const Vector3D v_axis = (edge_2 - edge_2.dot(u_axis) * u_axis).normalized();

    return _p1 + uv.x() * u_axis + uv.y() * v_axis;
}

Vector3D Triangle::normal_at_uv(const Point2D& /*uv*/) const noexcept {
    const Vector3D normal = triangle_normal_unnormalized(*this);
    return normal.normalized();
}

double Triangle::surface_area() const noexcept {
    return 0.5 * triangle_normal_unnormalized(*this).norm();
}

}  // namespace pycanha::gmm
