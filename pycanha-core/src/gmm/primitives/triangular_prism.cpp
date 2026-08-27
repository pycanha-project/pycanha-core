#include "pycanha-core/gmm/primitives/triangular_prism.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "detail.hpp"
#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {
namespace {

// The prism's surface is five faces: walls 0, 1 and 2 on the three base edges,
// then the base (3) and the top (4). uv follows the Cube convention -- the
// integer part of u selects the face and its fraction is the first surface
// parameter, v is the second -- so the whole solid has one parametrisation
// even though it is cutter-only and never meshed.
constexpr std::size_t num_walls = 3U;
constexpr std::size_t base_face = 3U;
constexpr std::size_t top_face = 4U;
constexpr std::size_t num_faces = 5U;

[[nodiscard]] std::size_t face_from_uv(const Point2D& uv) noexcept {
    const auto face = static_cast<std::size_t>(std::clamp(
        std::floor(uv.x()), 0.0, static_cast<double>(num_faces - 1)));
    return face;
}

[[nodiscard]] double local_u(const Point2D& uv) noexcept {
    return uv.x() - std::floor(uv.x());
}

// Components of `delta` in the (first, second) basis, which is not orthogonal
// in general. Zero when the two directions are parallel.
[[nodiscard]] std::pair<double, double> components_in_basis(
    const Vector3D& delta, const Vector3D& first, const Vector3D& second) {
    const double d11 = first.squaredNorm();
    const double d12 = first.dot(second);
    const double d22 = second.squaredNorm();
    const double denominator = (d11 * d22) - (d12 * d12);
    if (std::abs(denominator) <= LENGTH_TOL * LENGTH_TOL) {
        return {0.0, 0.0};
    }
    const double along_first = delta.dot(first);
    const double along_second = delta.dot(second);
    return {((along_first * d22) - (along_second * d12)) / denominator,
            ((along_second * d11) - (along_first * d12)) / denominator};
}

}  // namespace

TriangularPrism::TriangularPrism(Point3D p1, Point3D p2, Point3D p3,
                                 Point3D p4) noexcept
    : _p1(std::move(p1)),
      _p2(std::move(p2)),
      _p3(std::move(p3)),
      _p4(std::move(p4)) {}

const Point3D& TriangularPrism::p1() const noexcept { return _p1; }

const Point3D& TriangularPrism::p2() const noexcept { return _p2; }

const Point3D& TriangularPrism::p3() const noexcept { return _p3; }

const Point3D& TriangularPrism::p4() const noexcept { return _p4; }

void TriangularPrism::set_p1(Point3D p1) noexcept { _p1 = std::move(p1); }

void TriangularPrism::set_p2(Point3D p2) noexcept { _p2 = std::move(p2); }

void TriangularPrism::set_p3(Point3D p3) noexcept { _p3 = std::move(p3); }

void TriangularPrism::set_p4(Point3D p4) noexcept { _p4 = std::move(p4); }

Vector3D TriangularPrism::height() const noexcept { return _p4 - _p1; }

std::array<Point3D, 3> TriangularPrism::base() const noexcept {
    return {_p1, _p2, _p3};
}

bool TriangularPrism::is_valid() const noexcept {
    const Vector3D base_normal = (_p2 - _p1).cross(_p3 - _p1);
    if (!detail::has_nonzero_length(base_normal)) {
        return false;
    }
    const Vector3D extrusion = height();
    if (!detail::has_nonzero_length(extrusion)) {
        return false;
    }
    // An extrusion inside the base plane leaves the prism flat: no volume, so
    // nothing to subtract.
    return std::abs(base_normal.normalized().dot(extrusion.normalized())) >
           ANGLE_TOL;
}

Point3D TriangularPrism::to_cartesian(const Point2D& uv) const {
    const std::size_t face = face_from_uv(uv);
    const double first = local_u(uv);
    const double second = uv.y();
    const std::array<Point3D, 3> corners = base();
    const Vector3D extrusion = height();

    if (face < num_walls) {
        const Point3D& start = corners.at(face);
        const Point3D& end = corners.at((face + 1U) % num_walls);
        return start + (first * (end - start)) + (second * extrusion);
    }

    // The two triangular ends use the same strip parametrisation a Triangle
    // does: `first` walks out from p1 and `second` blends the two edges.
    const Vector3D edge_1 = _p2 - _p1;
    const Vector3D edge_2 = _p3 - _p1;
    const Point3D origin = face == top_face ? Point3D{_p1 + extrusion} : _p1;
    return origin + (first * (((1.0 - second) * edge_1) + (second * edge_2)));
}

Vector3D TriangularPrism::normal_at_uv(const Point2D& uv) const noexcept {
    const std::size_t face = face_from_uv(uv);
    const std::array<Point3D, 3> corners = base();
    const Vector3D extrusion = height();
    // The base vertices are ordered so (p2 - p1) x (p3 - p1) points ALONG the
    // extrusion, which makes edge x height the OUTWARD normal of each wall and
    // the base normal the outward normal of the top.
    const Vector3D base_normal = (_p2 - _p1).cross(_p3 - _p1).normalized();

    if (face < num_walls) {
        const Point3D& start = corners.at(face);
        const Point3D& end = corners.at((face + 1U) % num_walls);
        return (end - start).cross(extrusion).normalized();
    }
    return face == top_face ? base_normal : Vector3D{-base_normal};
}

Point2D TriangularPrism::to_uv(const Point3D& point) const {
    const std::array<Point3D, 3> corners = base();
    const Vector3D extrusion = height();
    const Vector3D base_normal = (_p2 - _p1).cross(_p3 - _p1).normalized();

    // Pick the face whose plane the point is nearest, the way a Cube does.
    std::size_t nearest_face = 0U;
    double nearest_distance = std::numeric_limits<double>::infinity();
    const auto consider = [&](std::size_t face, const Point3D& origin,
                              const Vector3D& normal) {
        const double distance = std::abs((point - origin).dot(normal));
        if (distance < nearest_distance) {
            nearest_distance = distance;
            nearest_face = face;
        }
    };

    for (std::size_t wall = 0; wall < num_walls; ++wall) {
        const Point3D& start = corners.at(wall);
        const Point3D& end = corners.at((wall + 1U) % num_walls);
        consider(wall, start, (end - start).cross(extrusion).normalized());
    }
    consider(base_face, _p1, base_normal);
    consider(top_face, Point3D{_p1 + extrusion}, base_normal);

    if (nearest_face < num_walls) {
        const Point3D& start = corners.at(nearest_face);
        const Point3D& end = corners.at((nearest_face + 1U) % num_walls);
        const auto [along_edge, along_height] =
            components_in_basis(point - start, end - start, extrusion);
        return {static_cast<double>(nearest_face) + along_edge, along_height};
    }

    const Point3D origin =
        nearest_face == top_face ? Point3D{_p1 + extrusion} : _p1;
    const auto [along_1, along_2] =
        components_in_basis(point - origin, _p2 - _p1, _p3 - _p1);
    const double out = along_1 + along_2;
    return {static_cast<double>(nearest_face) + out,
            std::abs(out) > LENGTH_TOL ? along_2 / out : 0.0};
}

double TriangularPrism::surface_area() const noexcept {
    const Vector3D edge_a = _p2 - _p1;
    const Vector3D edge_b = _p3 - _p1;
    const Vector3D extrusion = height();
    const double base_area = 0.5 * edge_a.cross(edge_b).norm();
    const double walls = edge_a.cross(extrusion).norm() +
                         edge_b.cross(extrusion).norm() +
                         (_p3 - _p2).cross(extrusion).norm();
    return (2.0 * base_area) + walls;
}

}  // namespace pycanha::gmm
