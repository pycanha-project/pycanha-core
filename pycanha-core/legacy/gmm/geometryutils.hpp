#pragma once

#include <Eigen/Dense>
#include <initializer_list>

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

/**
 * The type is either a Eigen::Vector2d or Eigen::Vector3d.
 */
template <typename T>
concept EigenVector2dOr3d =
    std::is_same_v<T, Eigen::Vector2d> || std::is_same_v<T, Eigen::Vector3d>;

/**
 * @brief Calculates the minimum distance from a point to a line segment in 3D
 * space.
 * @param point The point.
 * @param a The first end of the line segment.
 * @param b The second end of the line segment.
 * @return The minimum distance from the point to the line segment.
 */
inline double dist_point_to_line_segment(const Point3D& point, const Point3D& a,
                                         const Point3D& b) {
    const Eigen::Vector3d ab = b - a;
    const double t = (point - a).dot(ab) / ab.squaredNorm();
    if (t < 0.0) {
        return (point - a).norm();
    }
    if (t > 1.0) {
        return (point - b).norm();
    }
    return (point - (a + t * ab)).norm();
}

/**
 * @brief Check if a vector has a non-zero length.
 *
 * @tparam T Type of the vector. Must be an Eigen::Vector2d or Eigen::Vector3d.
 * @param v The vector to check.
 * @return True if the vector has a length greater than a LENGTH_TOL, false
 * otherwise.
 */
template <EigenVector2dOr3d T>
inline bool is_vector_nonzero_length(const T& v) {
    return v.norm() > LENGTH_TOL;
}

/**
 * @brief Check if a list of vectors have a non-zero length.
 *
 * @tparam T Type of the vectors. Must be an Eigen::Vector2d or Eigen::Vector3d.
 * @param vectors A list of pointers to vectors to check.
 * @return True if all vectors have a length greater than LENGTH_TOL, false
 * otherwise.
 */
template <EigenVector2dOr3d T>
inline bool are_vectors_nonzero_length(
    const std::initializer_list<const T*> vectors) {
    return std::ranges::all_of(
        vectors, [](const T* v) { return is_vector_nonzero_length(*v); });
}

/**
 * @brief Check if two vectors are orthogonal.
 * @tparam T Type of the vectors. Must be an Eigen::Vector2d or Eigen::Vector3d.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @return True if orthogonal within ANGLE_TOL.
 *
 * @note ANGLE_TOL is a constant defined in the same namespace as this function.
 */
template <EigenVector2dOr3d T>
inline bool are_vectors_orthogonal(const T& v1, const T& v2) {
    return std::abs(v1.normalized().dot(v2.normalized())) < ANGLE_TOL;
}

/**
 * @brief Check if two vectors are parallel (form 0 or 180 degrees). 3D version.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @return True if parallel within ANGLE_TOL.
 *
 * @note ANGLE_TOL is a constant defined in the same namespace as this function.
 * @note Assumed vectors are non-zero length.
 */

inline bool are_vectors_parallel(const Vector3D& v1, const Vector3D& v2) {
    // Compute the dot product of the normalized vectors v1 and v2
    const double cross = v1.normalized().cross(v2.normalized()).norm();

    // Check if the angle between v1 and v2 is too close to 0 or 180 degrees
    return cross <= ANGLE_TOL;
}

/**
 * @brief Check if two vectors are parallel (form 0 or 180 degrees). 2D version.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @return True if parallel within ANGLE_TOL.
 *
 * @note ANGLE_TOL is a constant defined in the same namespace as this function.
 * @note Assumed vectors are non-zero length.
 */

inline bool are_vectors_parallel(const Vector2D& v1, const Vector2D& v2) {
    // Compute the dot product of the normalized vectors v1 and v2
    const double cross = std::abs(v1.x() * v2.y() - v1.y() * v2.x());

    // Check if the angle between v1 and v2 is too close to 0 or 180 degrees
    return cross <= ANGLE_TOL;
}

}  // namespace pycanha::gmm
