#include "pycanha-core/gmm/primitives/rectangle.hpp"

#include <utility>

#include "detail.hpp"
#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

Rectangle::Rectangle(Point3D p1, Point3D p2, Point3D p3) noexcept
    : _p1(std::move(p1)), _p2(std::move(p2)), _p3(std::move(p3)) {}

const Point3D& Rectangle::p1() const noexcept { return _p1; }

const Point3D& Rectangle::p2() const noexcept { return _p2; }

const Point3D& Rectangle::p3() const noexcept { return _p3; }

void Rectangle::set_p1(Point3D p1) noexcept { _p1 = std::move(p1); }

void Rectangle::set_p2(Point3D p2) noexcept { _p2 = std::move(p2); }

void Rectangle::set_p3(Point3D p3) noexcept { _p3 = std::move(p3); }

bool Rectangle::is_valid() const noexcept {
    const Vector3D edge_1 = _p2 - _p1;
    const Vector3D edge_2 = _p3 - _p1;
    return detail::has_nonzero_length(edge_1) &&
           detail::has_nonzero_length(edge_2) &&
           detail::are_orthogonal(edge_1, edge_2);
}

Point2D Rectangle::to_uv(const Point3D& point) const {
    const auto basis = detail::make_plane_basis(_p2 - _p1, _p3 - _p1);
    const Vector3D delta = point - _p1;
    return {delta.dot(basis.u), delta.dot(basis.v)};
}

Point3D Rectangle::to_cartesian(const Point2D& uv) const {
    const auto basis = detail::make_plane_basis(_p2 - _p1, _p3 - _p1);
    return _p1 + uv.x() * basis.u + uv.y() * basis.v;
}

Vector3D Rectangle::normal_at_uv(const Point2D& /*uv*/) const noexcept {
    return ((_p2 - _p1).cross(_p3 - _p1)).normalized();
}

double Rectangle::surface_area() const noexcept {
    return ((_p2 - _p1).cross(_p3 - _p1)).norm();
}

}  // namespace pycanha::gmm
