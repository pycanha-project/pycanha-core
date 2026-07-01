#include "pycanha-core/gmm/primitives/quadrilateral.hpp"

#include <array>
#include <cmath>
#include <tuple>
#include <utility>

#include "detail.hpp"
#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

Quadrilateral::Quadrilateral(Point3D p1, Point3D p2, Point3D p3,
                             Point3D p4) noexcept
    : _p1(std::move(p1)),
      _p2(std::move(p2)),
      _p3(std::move(p3)),
      _p4(std::move(p4)) {}

const Point3D& Quadrilateral::p1() const noexcept { return _p1; }

const Point3D& Quadrilateral::p2() const noexcept { return _p2; }

const Point3D& Quadrilateral::p3() const noexcept { return _p3; }

const Point3D& Quadrilateral::p4() const noexcept { return _p4; }

void Quadrilateral::set_p1(Point3D p1) noexcept { _p1 = std::move(p1); }

void Quadrilateral::set_p2(Point3D p2) noexcept { _p2 = std::move(p2); }

void Quadrilateral::set_p3(Point3D p3) noexcept { _p3 = std::move(p3); }

void Quadrilateral::set_p4(Point3D p4) noexcept { _p4 = std::move(p4); }

bool Quadrilateral::is_valid() const noexcept {
    Vector3D edge_1 = _p2 - _p1;
    Vector3D edge_2 = _p4 - _p1;

    if (!detail::has_nonzero_length(edge_1) ||
        !detail::has_nonzero_length(edge_2)) {
        return false;
    }

    if (std::abs(edge_1.normalized().dot(edge_2.normalized())) >=
        1.0 - ANGLE_TOL) {
        return false;
    }

    Vector3D normal = edge_1.cross(edge_2);
    if (!detail::has_nonzero_length(normal)) {
        return false;
    }
    normal.normalize();

    if (std::abs((_p3 - _p1).dot(normal)) > LENGTH_TOL) {
        return false;
    }

    const std::array<std::tuple<Point3D, Point3D, Point3D>, 3> corners = {
        std::make_tuple(_p2, _p3, _p1),
        std::make_tuple(_p3, _p4, _p2),
        std::make_tuple(_p4, _p1, _p3),
    };

    for (const auto& [current, previous, next] : corners) {
        edge_1 = previous - current;
        edge_2 = next - current;
        if (!detail::has_nonzero_length(edge_1) ||
            !detail::has_nonzero_length(edge_2)) {
            return false;
        }

        const double cosine =
            edge_1.dot(edge_2) / (edge_1.norm() * edge_2.norm());
        if (cosine <= -1.0 + ANGLE_TOL || cosine >= 1.0 - ANGLE_TOL) {
            return false;
        }
    }

    return true;
}

Point2D Quadrilateral::to_uv(const Point3D& point) const {
    const auto basis = detail::make_plane_basis(_p2 - _p1, _p4 - _p1);
    const Vector3D delta = point - _p1;
    return {delta.dot(basis.u), delta.dot(basis.v)};
}

Point3D Quadrilateral::to_cartesian(const Point2D& uv) const {
    const auto basis = detail::make_plane_basis(_p2 - _p1, _p4 - _p1);
    return _p1 + uv.x() * basis.u + uv.y() * basis.v;
}

Vector3D Quadrilateral::normal_at_uv(const Point2D& /*uv*/) const noexcept {
    return ((_p2 - _p1).cross(_p4 - _p1)).normalized();
}

double Quadrilateral::surface_area() const noexcept {
    return ((_p2 - _p1).cross(_p4 - _p1)).norm();
}

}  // namespace pycanha::gmm
