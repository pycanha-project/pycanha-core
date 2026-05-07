#include "pycanha-core/gmm/ops/distance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>
#include <variant>

#include "../primitives/detail.hpp"
#include "pycanha-core/globals.hpp"
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

constexpr double full_turn = 2.0 * std::numbers::pi;

[[nodiscard]] double clamp_scalar(double value, double low,
                                  double high) noexcept {
    return std::clamp(value, low, high);
}

[[nodiscard]] double clamp_periodic_angle(double angle, double start,
                                          double end) noexcept {
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

[[nodiscard]] double distance_to_triangle(const Point3D& point,
                                          const Triangle& triangle) {
    const Vector3D ab = triangle.p2() - triangle.p1();
    const Vector3D ac = triangle.p3() - triangle.p1();
    const Vector3D ap = point - triangle.p1();

    const double d1 = ab.dot(ap);
    const double d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return ap.norm();
    }

    const Vector3D bp = point - triangle.p2();
    const double d3 = ab.dot(bp);
    const double d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return bp.norm();
    }

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        return (point - (triangle.p1() + v * ab)).norm();
    }

    const Vector3D cp = point - triangle.p3();
    const double d5 = ab.dot(cp);
    const double d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return cp.norm();
    }

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        return (point - (triangle.p1() + w * ac)).norm();
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const Vector3D bc = triangle.p3() - triangle.p2();
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (point - (triangle.p2() + w * bc)).norm();
    }

    return std::abs(ap.dot(ab.cross(ac).normalized()));
}

template <typename Surface>
[[nodiscard]] double distance_to_planar_patch(const Point3D& point,
                                              const Surface& surface,
                                              double max_u, double max_v) {
    const Point2D uv = surface.to_uv(point);
    const Point2D clamped_uv{clamp_scalar(uv.x(), 0.0, max_u),
                             clamp_scalar(uv.y(), 0.0, max_v)};
    return (point - surface.to_cartesian(clamped_uv)).norm();
}

[[nodiscard]] double distance_impl(const Rectangle& rectangle,
                                   const Point3D& point) {
    return distance_to_planar_patch(point, rectangle,
                                    (rectangle.p2() - rectangle.p1()).norm(),
                                    rectangle.to_uv(rectangle.p3()).y());
}

[[nodiscard]] double distance_impl(const Quadrilateral& quadrilateral,
                                   const Point3D& point) {
    return distance_to_planar_patch(
        point, quadrilateral, (quadrilateral.p2() - quadrilateral.p1()).norm(),
        quadrilateral.to_uv(quadrilateral.p4()).y());
}

[[nodiscard]] double distance_impl(const Disc& disc, const Point3D& point) {
    const Vector3D axis = detail::axis_direction(disc.p1(), disc.p2());
    const Vector3D radial_reference =
        detail::radial_reference(disc.p1(), disc.p3(), axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const Vector3D delta = point - disc.p1();
    const double axial = delta.dot(axis);
    const Vector3D planar = delta - axial * axis;
    const double radius = planar.norm();
    const Vector3D radial =
        radius > LENGTH_TOL ? planar / radius : radial_reference;
    const double theta =
        detail::angle_about_axis(radial, radial_reference, tangent);
    const double clamped_theta =
        clamp_periodic_angle(theta, disc.start_angle(), disc.end_angle());
    const double clamped_radius =
        clamp_scalar(radius, disc.inner_radius(), disc.outer_radius());
    const Vector3D clamped_direction =
        std::cos(clamped_theta) * radial_reference +
        std::sin(clamped_theta) * tangent;
    const Point3D closest = disc.p1() + clamped_radius * clamped_direction;
    return (point - closest).norm();
}

[[nodiscard]] double distance_impl(const Cylinder& cylinder,
                                   const Point3D& point) {
    const Vector3D axis = detail::axis_direction(cylinder.p1(), cylinder.p2());
    const Vector3D radial_reference =
        detail::radial_reference(cylinder.p1(), cylinder.p3(), axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const Vector3D delta = point - cylinder.p1();
    const double height = clamp_scalar(delta.dot(axis), 0.0,
                                       (cylinder.p2() - cylinder.p1()).norm());
    const Vector3D radial_vector = delta - delta.dot(axis) * axis;
    const double radial_distance = radial_vector.norm();
    const Vector3D radial = radial_distance > LENGTH_TOL
                                ? radial_vector / radial_distance
                                : radial_reference;
    const double theta =
        detail::angle_about_axis(radial, radial_reference, tangent);
    const double clamped_theta = clamp_periodic_angle(
        theta, cylinder.start_angle(), cylinder.end_angle());
    const Vector3D clamped_direction =
        std::cos(clamped_theta) * radial_reference +
        std::sin(clamped_theta) * tangent;
    const Point3D closest =
        cylinder.p1() + height * axis + cylinder.radius() * clamped_direction;
    return (point - closest).norm();
}

[[nodiscard]] double distance_impl(const Cone& cone, const Point3D& point) {
    const Vector3D axis = detail::axis_direction(cone.p1(), cone.p2());
    const Vector3D radial_reference =
        detail::radial_reference(cone.p1(), cone.p3(), axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const Vector3D delta = point - cone.p1();
    const double total_height = (cone.p2() - cone.p1()).norm();
    const double raw_height = delta.dot(axis);
    const Vector3D radial_vector = delta - raw_height * axis;
    const double radial_distance = radial_vector.norm();
    const Vector3D radial = radial_distance > LENGTH_TOL
                                ? radial_vector / radial_distance
                                : radial_reference;
    const double theta =
        detail::angle_about_axis(radial, radial_reference, tangent);
    const double clamped_theta =
        clamp_periodic_angle(theta, cone.start_angle(), cone.end_angle());

    const Eigen::Vector2d segment_start(0.0, cone.radius1());
    const Eigen::Vector2d segment_end(total_height, cone.radius2());
    const Eigen::Vector2d segment = segment_end - segment_start;
    const Eigen::Vector2d point_2d(raw_height, radial_distance);
    const double segment_length_sq = segment.squaredNorm();
    const double projection =
        segment_length_sq > LENGTH_TOL
            ? clamp_scalar(
                  (point_2d - segment_start).dot(segment) / segment_length_sq,
                  0.0, 1.0)
            : 0.0;
    const Eigen::Vector2d closest_2d = segment_start + projection * segment;
    const Vector3D clamped_direction =
        std::cos(clamped_theta) * radial_reference +
        std::sin(clamped_theta) * tangent;
    const Point3D closest =
        cone.p1() + closest_2d.x() * axis + closest_2d.y() * clamped_direction;
    return (point - closest).norm();
}

[[nodiscard]] double distance_impl(const Sphere& sphere, const Point3D& point) {
    const auto frame =
        detail::make_sphere_frame(sphere.p1(), sphere.p2(), sphere.p3());
    const Vector3D delta = point - sphere.p1();
    const double delta_norm = delta.norm();
    const Vector3D direction =
        delta_norm > LENGTH_TOL ? delta / delta_norm : frame.ref;
    const double local_x = direction.dot(frame.ref);
    const double local_y = direction.dot(frame.tangent);
    const double local_z = direction.dot(frame.axis);
    const double longitude = std::atan2(local_y, local_x);
    const double latitude = std::asin(detail::clamp_unit(local_z));
    const double min_latitude =
        std::asin(sphere.base_truncation() / sphere.radius());
    const double max_latitude =
        std::asin(sphere.apex_truncation() / sphere.radius());
    const double clamped_longitude = clamp_periodic_angle(
        longitude, sphere.start_angle(), sphere.end_angle());
    const double clamped_latitude =
        clamp_scalar(latitude, min_latitude, max_latitude);
    const Point2D closest_uv{
        sphere.radius() * clamped_longitude * std::cos(clamped_latitude),
        sphere.radius() * clamped_latitude,
    };
    const Point3D closest = sphere.to_cartesian(closest_uv);
    return (point - closest).norm();
}

[[nodiscard]] double paraboloid_radius_at_height(const Paraboloid& paraboloid,
                                                 double height) {
    const double total_height = (paraboloid.p2() - paraboloid.p1()).norm();
    if (total_height <= LENGTH_TOL) {
        return 0.0;
    }
    return paraboloid.radius() *
           std::sqrt(std::max(height, 0.0) / total_height);
}

[[nodiscard]] double distance_impl(const Paraboloid& paraboloid,
                                   const Point3D& point) {
    const Vector3D axis =
        detail::axis_direction(paraboloid.p1(), paraboloid.p2());
    const Vector3D radial_reference =
        detail::radial_reference(paraboloid.p1(), paraboloid.p3(), axis);
    const Vector3D tangent =
        detail::tangential_direction(axis, radial_reference);
    const Vector3D delta = point - paraboloid.p1();
    const double total_height = (paraboloid.p2() - paraboloid.p1()).norm();
    const double raw_height = delta.dot(axis);
    const Vector3D radial_vector = delta - raw_height * axis;
    const double radial_distance = radial_vector.norm();
    const Vector3D radial = radial_distance > LENGTH_TOL
                                ? radial_vector / radial_distance
                                : radial_reference;
    const double theta =
        detail::angle_about_axis(radial, radial_reference, tangent);
    const double clamped_theta = clamp_periodic_angle(
        theta, paraboloid.start_angle(), paraboloid.end_angle());

    double low = 0.0;
    double high = total_height;
    for (int iteration = 0; iteration < 48; ++iteration) {
        const double left = low + (high - low) / 3.0;
        const double right = high - (high - low) / 3.0;
        const double radius_left =
            paraboloid_radius_at_height(paraboloid, left);
        const double radius_right =
            paraboloid_radius_at_height(paraboloid, right);
        const double dist_left = std::pow(left - raw_height, 2.0) +
                                 std::pow(radius_left - radial_distance, 2.0);
        const double dist_right = std::pow(right - raw_height, 2.0) +
                                  std::pow(radius_right - radial_distance, 2.0);
        if (dist_left <= dist_right) {
            high = right;
        } else {
            low = left;
        }
    }

    const double best_height = 0.5 * (low + high);
    const double best_radius =
        paraboloid_radius_at_height(paraboloid, best_height);
    const Vector3D clamped_direction =
        std::cos(clamped_theta) * radial_reference +
        std::sin(clamped_theta) * tangent;
    const Point3D closest =
        paraboloid.p1() + best_height * axis + best_radius * clamped_direction;
    return (point - closest).norm();
}

[[nodiscard]] double distance_impl(const Cube& cube, const Point3D& point) {
    const Vector3D half_extent = cube.extent() * 0.5;
    const Vector3D local =
        (cube.orientation().conjugate() * (point - cube.center())).cwiseAbs();
    const Vector3D outside = (local - half_extent).cwiseMax(0.0);

    if (outside.maxCoeff() > 0.0) {
        return outside.norm();
    }

    return (half_extent - local).minCoeff();
}

}  // namespace

double distance(const Primitive& primitive, const Point3D& point) {
    return std::visit(
        [&point](const auto& concrete_primitive) {
            if constexpr (std::is_same_v<
                              std::decay_t<decltype(concrete_primitive)>,
                              Triangle>) {
                return distance_to_triangle(point, concrete_primitive);
            } else {
                return distance_impl(concrete_primitive, point);
            }
        },
        primitive);
}

}  // namespace pycanha::gmm::ops
