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

// Inverts the planar bilinear patch
//   P(u,v) = p1 + u*e + v*f + u*v*g,  e = p2-p1, f = p4-p1, g = p1-p2+p3-p4
// for a point assumed to lie in the patch's plane. Substituting and crossing
// with (f + u*g) removes v and leaves a quadratic in u:
//   (e x g) u^2 + [(e x f) - (q x g)] u - (q x f) = 0,   q = P - p1
// where x is the 2D cross product in the patch plane. g vanishes for a
// parallelogram, which degrades the quadratic to the linear solution.
// Once u is known, v follows from a single projection.
[[nodiscard]] inline Point2D invert_bilinear(const Vector3D& q,
                                             const Vector3D& e,
                                             const Vector3D& f,
                                             const Vector3D& g,
                                             const Vector3D& normal) {
    const auto cross2d = [&normal](const Vector3D& lhs, const Vector3D& rhs) {
        return lhs.cross(rhs).dot(normal);
    };

    const double quad_a = cross2d(e, g);
    const double quad_b = cross2d(e, f) - cross2d(q, g);
    const double quad_c = -cross2d(q, f);

    double u = 0.0;
    if (std::abs(quad_a) <= LENGTH_TOL * LENGTH_TOL) {
        u = std::abs(quad_b) > LENGTH_TOL * LENGTH_TOL ? -quad_c / quad_b : 0.0;
    } else {
        const double discriminant =
            std::max((quad_b * quad_b) - (4.0 * quad_a * quad_c), 0.0);
        const double root = std::sqrt(discriminant);
        // Both roots are real for a convex patch; the one inside [0,1] is the
        // point's own parameter, the other lies on the patch's other branch.
        const double u_minus = (-quad_b - root) / (2.0 * quad_a);
        const double u_plus = (-quad_b + root) / (2.0 * quad_a);
        const auto outside = [](double value) {
            return std::max({0.0 - value, value - 1.0, 0.0});
        };
        u = outside(u_minus) <= outside(u_plus) ? u_minus : u_plus;
    }

    const Vector3D v_direction = f + (u * g);
    const double v_scale = v_direction.squaredNorm();
    const double v = v_scale > LENGTH_TOL * LENGTH_TOL
                         ? (q - (u * e)).dot(v_direction) / v_scale
                         : 0.0;
    return {u, v};
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
    return min_value + ((unit_value * (max_value - min_value)));
}

}  // namespace pycanha::gmm::detail
