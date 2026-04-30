#include "pycanha-core/gmm/primitives/cube.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "detail.hpp"
#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {
namespace {

[[nodiscard]] Point3D cube_local_point(const Cube& cube,
                                       const Point3D& point) noexcept {
    return point - cube.center();
}

[[nodiscard]] int cube_face_from_uv(const Point2D& uv) noexcept {
    const int face = static_cast<int>(std::floor(uv.x()));
    return std::clamp(face, 0, 5);
}

[[nodiscard]] double cube_local_u(const Point2D& uv) noexcept {
    return uv.x() - std::floor(uv.x());
}

}  // namespace

Cube::Cube(Point3D center, Vector3D extent) noexcept
    : _center(std::move(center)), _extent(std::move(extent)) {}

const Point3D& Cube::center() const noexcept { return _center; }

const Vector3D& Cube::extent() const noexcept { return _extent; }

void Cube::set_center(Point3D center) noexcept { _center = std::move(center); }

void Cube::set_extent(Vector3D extent) noexcept { _extent = std::move(extent); }

bool Cube::is_valid() const noexcept {
    return _extent.x() > LENGTH_TOL && _extent.y() > LENGTH_TOL &&
           _extent.z() > LENGTH_TOL;
}

Point2D Cube::to_uv(const Point3D& point) const {
    const Point3D local = cube_local_point(*this, point);
    const double hx = detail::cube_half_extent(_extent.x());
    const double hy = detail::cube_half_extent(_extent.y());
    const double hz = detail::cube_half_extent(_extent.z());

    const double dx = std::abs(std::abs(local.x()) - hx);
    const double dy = std::abs(std::abs(local.y()) - hy);
    const double dz = std::abs(std::abs(local.z()) - hz);

    if (dx <= dy && dx <= dz) {
        if (local.x() >= 0.0) {
            return {0.0 + detail::unit_from_interval(local.y(), -hy, hy),
                    detail::unit_from_interval(local.z(), -hz, hz)};
        }
        return {1.0 + detail::unit_from_interval(local.y(), hy, -hy),
                detail::unit_from_interval(local.z(), -hz, hz)};
    }

    if (dy <= dx && dy <= dz) {
        if (local.y() >= 0.0) {
            return {2.0 + detail::unit_from_interval(local.x(), hx, -hx),
                    detail::unit_from_interval(local.z(), -hz, hz)};
        }
        return {3.0 + detail::unit_from_interval(local.x(), -hx, hx),
                detail::unit_from_interval(local.z(), -hz, hz)};
    }

    if (local.z() >= 0.0) {
        return {4.0 + detail::unit_from_interval(local.x(), -hx, hx),
                detail::unit_from_interval(local.y(), -hy, hy)};
    }
    return {5.0 + detail::unit_from_interval(local.x(), -hx, hx),
            detail::unit_from_interval(local.y(), hy, -hy)};
}

Point3D Cube::to_cartesian(const Point2D& uv) const {
    const int face = cube_face_from_uv(uv);
    const double s = cube_local_u(uv);
    const double t = uv.y();
    const double hx = detail::cube_half_extent(_extent.x());
    const double hy = detail::cube_half_extent(_extent.y());
    const double hz = detail::cube_half_extent(_extent.z());

    switch (face) {
        case 0:
            return _center + Vector3D(hx,
                                      detail::interval_from_unit(s, -hy, hy),
                                      detail::interval_from_unit(t, -hz, hz));
        case 1:
            return _center + Vector3D(-hx,
                                      detail::interval_from_unit(s, hy, -hy),
                                      detail::interval_from_unit(t, -hz, hz));
        case 2:
            return _center + Vector3D(detail::interval_from_unit(s, hx, -hx),
                                      hy,
                                      detail::interval_from_unit(t, -hz, hz));
        case 3:
            return _center + Vector3D(detail::interval_from_unit(s, -hx, hx),
                                      -hy,
                                      detail::interval_from_unit(t, -hz, hz));
        case 4:
            return _center + Vector3D(detail::interval_from_unit(s, -hx, hx),
                                      detail::interval_from_unit(t, -hy, hy),
                                      hz);
        default:
            return _center + Vector3D(detail::interval_from_unit(s, -hx, hx),
                                      detail::interval_from_unit(t, hy, -hy),
                                      -hz);
    }
}

// This stays a member to preserve the common primitive API.
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
Vector3D Cube::normal_at_uv(const Point2D& uv) const noexcept {
    switch (cube_face_from_uv(uv)) {
        case 0:
            return Vector3D::UnitX();
        case 1:
            return -Vector3D::UnitX();
        case 2:
            return Vector3D::UnitY();
        case 3:
            return -Vector3D::UnitY();
        case 4:
            return Vector3D::UnitZ();
        default:
            return -Vector3D::UnitZ();
    }
}

double Cube::surface_area() const noexcept {
    return 2.0 * (_extent.x() * _extent.y() + _extent.x() * _extent.z() +
                  _extent.y() * _extent.z());
}

}  // namespace pycanha::gmm
