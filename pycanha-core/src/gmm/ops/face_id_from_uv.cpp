#include "pycanha-core/gmm/ops/face_id_from_uv.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"

namespace pycanha::gmm::ops {
namespace {

[[nodiscard]] std::size_t find_cell(std::span<const double> cuts,
                                    double normalized_value) {
    const double clamped = std::clamp(normalized_value, 0.0, 1.0);
    const auto upper = std::upper_bound(cuts.begin(), cuts.end(), clamped);
    if (upper == cuts.begin()) {
        return 0U;
    }
    if (upper == cuts.end()) {
        return cuts.size() - 2U;
    }
    return static_cast<std::size_t>(std::distance(cuts.begin(), upper) - 1);
}

[[nodiscard]] double normalize_linear(double value, double max_value) {
    return max_value > LENGTH_TOL ? value / max_value : 0.0;
}

[[nodiscard]] double clamp_periodic_angle(double angle, double start,
                                          double end) noexcept {
    constexpr double full_turn = 2.0 * std::numbers::pi;
    double best = start;
    double best_delta = std::numeric_limits<double>::infinity();

    for (const double shifted : {angle - full_turn, angle, angle + full_turn}) {
        const double candidate = std::clamp(shifted, start, end);
        const double delta = std::abs(shifted - candidate);
        if (delta < best_delta) {
            best = candidate;
            best_delta = delta;
        }
    }

    return best;
}

[[nodiscard]] std::pair<double, double> normalized_uv(const Triangle& triangle,
                                                      const Point2D& uv) {
    const Vector3D edge_1 = triangle.p2() - triangle.p1();
    const Vector3D edge_2 = triangle.p3() - triangle.p1();
    const Vector3D u_axis = edge_1.normalized();
    const Vector3D v_axis = (edge_2 - edge_2.dot(u_axis) * u_axis).normalized();
    const double edge_1_length = edge_1.norm();
    const double edge_2_along_u = edge_2.dot(u_axis);
    const double edge_2_along_v = edge_2.dot(v_axis);

    const double along_edge_2 =
        edge_2_along_v > LENGTH_TOL ? uv.y() / edge_2_along_v : 0.0;
    const double along_edge_1 =
        edge_1_length > LENGTH_TOL
            ? (uv.x() - along_edge_2 * edge_2_along_u) / edge_1_length
            : 0.0;

    const double dir1 = std::clamp(along_edge_1 + along_edge_2, 0.0, 1.0);
    const double dir2 =
        dir1 > LENGTH_TOL ? std::clamp(along_edge_2 / dir1, 0.0, 1.0) : 0.0;

    return {dir1, dir2};
}

[[nodiscard]] std::pair<double, double> normalized_uv(
    const Rectangle& rectangle, const Point2D& uv) {
    return {normalize_linear(uv.x(), (rectangle.p2() - rectangle.p1()).norm()),
            normalize_linear(uv.y(), rectangle.to_uv(rectangle.p3()).y())};
}

[[nodiscard]] std::pair<double, double> normalized_uv(
    const Quadrilateral& quadrilateral, const Point2D& uv) {
    return {
        normalize_linear(uv.x(),
                         (quadrilateral.p2() - quadrilateral.p1()).norm()),
        normalize_linear(uv.y(), quadrilateral.to_uv(quadrilateral.p4()).y())};
}

[[nodiscard]] std::pair<double, double> normalized_uv(const Disc& disc,
                                                      const Point2D& uv) {
    const double radius = uv.y();
    const double theta =
        radius > LENGTH_TOL ? uv.x() / radius : disc.start_angle();
    return {(clamp_periodic_angle(theta, disc.start_angle(), disc.end_angle()) -
             disc.start_angle()) /
                (disc.end_angle() - disc.start_angle()),
            (radius - disc.inner_radius()) /
                (disc.outer_radius() - disc.inner_radius())};
}

[[nodiscard]] std::pair<double, double> normalized_uv(const Cylinder& cylinder,
                                                      const Point2D& uv) {
    const double theta = uv.x() / cylinder.radius();
    return {(clamp_periodic_angle(theta, cylinder.start_angle(),
                                  cylinder.end_angle()) -
             cylinder.start_angle()) /
                (cylinder.end_angle() - cylinder.start_angle()),
            normalize_linear(uv.y(), (cylinder.p2() - cylinder.p1()).norm())};
}

[[nodiscard]] std::pair<double, double> normalized_uv(const Cone& cone,
                                                      const Point2D& uv) {
    const double total_height = (cone.p2() - cone.p1()).norm();
    const double height_fraction = normalize_linear(uv.y(), total_height);
    const double radius =
        cone.radius1() + (cone.radius2() - cone.radius1()) * height_fraction;
    const double theta =
        radius > LENGTH_TOL ? uv.x() / radius : cone.start_angle();
    return {(clamp_periodic_angle(theta, cone.start_angle(), cone.end_angle()) -
             cone.start_angle()) /
                (cone.end_angle() - cone.start_angle()),
            height_fraction};
}

[[nodiscard]] std::pair<double, double> normalized_uv(const Sphere& sphere,
                                                      const Point2D& uv) {
    const double latitude = uv.y() / sphere.radius();
    const double longitude =
        std::abs(std::cos(latitude)) > LENGTH_TOL
            ? uv.x() / (sphere.radius() * std::cos(latitude))
            : sphere.start_angle();
    const double min_latitude =
        std::asin(sphere.base_truncation() / sphere.radius());
    const double max_latitude =
        std::asin(sphere.apex_truncation() / sphere.radius());
    return {(clamp_periodic_angle(longitude, sphere.start_angle(),
                                  sphere.end_angle()) -
             sphere.start_angle()) /
                (sphere.end_angle() - sphere.start_angle()),
            (latitude - min_latitude) / (max_latitude - min_latitude)};
}

[[nodiscard]] std::pair<double, double> normalized_uv(
    const Paraboloid& paraboloid, const Point2D& uv) {
    const double total_height = (paraboloid.p2() - paraboloid.p1()).norm();
    const double height_fraction = normalize_linear(uv.y(), total_height);
    const double radius = paraboloid.radius() * std::sqrt(height_fraction);
    const double theta =
        radius > LENGTH_TOL ? uv.x() / radius : paraboloid.start_angle();
    return {(clamp_periodic_angle(theta, paraboloid.start_angle(),
                                  paraboloid.end_angle()) -
             paraboloid.start_angle()) /
                (paraboloid.end_angle() - paraboloid.start_angle()),
            height_fraction};
}

[[nodiscard]] std::pair<double, double> normalized_uv(
    [[maybe_unused]] const Cube& cube, [[maybe_unused]] const Point2D& uv) {
    throw std::logic_error(
        "Cube face_id_from_uv requires a face topology definition");
}

}  // namespace

FaceId face_id_from_uv(const Primitive& primitive,
                       const ThermalMesh& thermal_mesh, const Point2D& uv,
                       Side side) {
    const auto [dir1, dir2] = std::visit(
        [&uv](const auto& concrete_primitive) {
            return normalized_uv(concrete_primitive, uv);
        },
        primitive);

    return thermal_mesh.face_id(find_cell(thermal_mesh.dir1_cuts(), dir1),
                                find_cell(thermal_mesh.dir2_cuts(), dir2),
                                side);
}

}  // namespace pycanha::gmm::ops
