#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm::detail {

struct PlaneBasis {
    Vector3D u;
    Vector3D v;
    Vector3D n;
};

struct SphereFrame {
    Vector3D ref;
    Vector3D tangent;
    Vector3D axis;
};

[[nodiscard]] inline bool has_nonzero_length(const Vector3D& vector) noexcept {
    return vector.norm() > LENGTH_TOL;
}

[[nodiscard]] inline bool are_orthogonal(const Vector3D& lhs,
                                         const Vector3D& rhs) noexcept {
    return has_nonzero_length(lhs) && has_nonzero_length(rhs) &&
           std::abs(lhs.normalized().dot(rhs.normalized())) <= ANGLE_TOL;
}

[[nodiscard]] inline Vector3D project_onto_plane(const Vector3D& vector,
                                                 const Vector3D& normal) {
    return vector - vector.dot(normal) * normal;
}

[[nodiscard]] inline PlaneBasis make_plane_basis(const Vector3D& primary,
                                                 const Vector3D& secondary) {
    const Vector3D u = primary.normalized();
    const Vector3D secondary_projected =
        project_onto_plane(secondary, u).normalized();
    const Vector3D n = u.cross(secondary_projected).normalized();
    return {.u = u, .v = secondary_projected, .n = n};
}

[[nodiscard]] inline bool angle_span_is_valid(double start_angle,
                                              double end_angle) noexcept {
    constexpr double full_turn = std::numbers::pi * 2.0;
    return start_angle >= -full_turn && start_angle <= full_turn &&
           end_angle >= -full_turn && end_angle <= full_turn &&
           (end_angle - start_angle) >= ANGLE_TOL &&
           (end_angle - start_angle) < full_turn + ANGLE_TOL;
}

[[nodiscard]] inline double wrap_angle_positive(double angle) noexcept {
    constexpr double full_turn = std::numbers::pi * 2.0;
    while (angle < 0.0) {
        angle += full_turn;
    }
    while (angle >= full_turn) {
        angle -= full_turn;
    }
    return angle;
}

[[nodiscard]] inline double clamp_unit(double value) noexcept {
    return std::clamp(value, -1.0, 1.0);
}

[[nodiscard]] inline Vector3D axis_direction(const Point3D& p1,
                                             const Point3D& p2) {
    return (p2 - p1).normalized();
}

[[nodiscard]] inline Vector3D radial_reference(const Point3D& origin,
                                               const Point3D& reference_point,
                                               const Vector3D& axis) {
    return project_onto_plane(reference_point - origin, axis).normalized();
}

[[nodiscard]] inline Vector3D tangential_direction(const Vector3D& axis,
                                                   const Vector3D& radial) {
    return axis.cross(radial).normalized();
}

[[nodiscard]] inline double angle_about_axis(const Vector3D& radial,
                                             const Vector3D& reference,
                                             const Vector3D& tangent) noexcept {
    return wrap_angle_positive(
        std::atan2(radial.dot(tangent), radial.dot(reference)));
}

[[nodiscard]] inline SphereFrame make_sphere_frame(const Point3D& center,
                                                   const Point3D& axis_point,
                                                   const Point3D& ref_point) {
    const Vector3D axis = axis_direction(center, axis_point);
    const Vector3D ref = radial_reference(center, ref_point, axis);
    return {
        .ref = ref, .tangent = tangential_direction(axis, ref), .axis = axis};
}

[[nodiscard]] inline Point3D from_local_spherical(const Point3D& center,
                                                  const SphereFrame& frame,
                                                  double radius,
                                                  double longitude,
                                                  double latitude) {
    const double cos_latitude = std::cos(latitude);
    return center +
           radius * (cos_latitude * std::cos(longitude) * frame.ref +
                     cos_latitude * std::sin(longitude) * frame.tangent +
                     std::sin(latitude) * frame.axis);
}

[[nodiscard]] inline double cube_half_extent(double extent_component) {
    return extent_component * 0.5;
}

[[nodiscard]] inline double unit_from_interval(double value, double min_value,
                                               double max_value) {
    return (value - min_value) / (max_value - min_value);
}

[[nodiscard]] inline double interval_from_unit(double unit_value,
                                               double min_value,
                                               double max_value) {
    return min_value + unit_value * (max_value - min_value);
}

}  // namespace pycanha::gmm::detail
