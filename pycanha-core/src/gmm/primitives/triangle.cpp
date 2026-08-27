#include "pycanha-core/gmm/primitives/triangle.hpp"

#include <cmath>
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

// uv is normalised to [0, 1]^2 for every planar primitive. For a triangle
// that is the strip parametrisation the mesher subdivides in: u runs from the
// p1 corner out to the opposite edge, and v slides along that edge from p2 to
// p3, so the whole domain maps onto the triangle and u = 0 collapses to p1.
Point2D Triangle::to_uv(const Point3D& point) const {
    const Vector3D edge_1 = triangle_edge_1(*this);
    const Vector3D edge_2 = triangle_edge_2(*this);
    const Vector3D delta = point - _p1;

    // Components of delta in the (edge_1, edge_2) basis, which is not
    // orthogonal in general.
    const double d11 = edge_1.squaredNorm();
    const double d12 = edge_1.dot(edge_2);
    const double d22 = edge_2.squaredNorm();
    const double denominator = (d11 * d22) - (d12 * d12);
    if (std::abs(denominator) <= LENGTH_TOL * LENGTH_TOL) {
        return {0.0, 0.0};
    }
    const double delta_1 = delta.dot(edge_1);
    const double delta_2 = delta.dot(edge_2);
    const double along_1 = ((delta_1 * d22) - (delta_2 * d12)) / denominator;
    const double along_2 = ((delta_2 * d11) - (delta_1 * d12)) / denominator;

    // delta = u(1 - v) * edge_1 + u * v * edge_2.
    const double u = along_1 + along_2;
    return {u, std::abs(u) > LENGTH_TOL ? along_2 / u : 0.0};
}

Point3D Triangle::to_cartesian(const Point2D& uv) const {
    const Vector3D edge_1 = triangle_edge_1(*this);
    const Vector3D edge_2 = triangle_edge_2(*this);
    return _p1 + (uv.x() * (((1.0 - uv.y()) * edge_1) + (uv.y() * edge_2)));
}

Vector3D Triangle::normal_at_uv(const Point2D& /*uv*/) const noexcept {
    const Vector3D normal = triangle_normal_unnormalized(*this);
    return normal.normalized();
}

double Triangle::surface_area() const noexcept {
    return 0.5 * triangle_normal_unnormalized(*this).norm();
}

}  // namespace pycanha::gmm
