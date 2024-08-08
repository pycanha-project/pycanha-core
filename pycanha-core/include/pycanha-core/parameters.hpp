#pragma once
#include <Eigen/Dense>
#include <limits>
#include <vector>

namespace pycanha {

// Tolerances (SI Units)

constexpr double LENGTH_TOL = 1e-9;  // NOLINT(readability-identifier-naming)
constexpr double ANGLE_TOL = 1e-9;   // NOLINT(readability-identifier-naming)
constexpr double TOL = 1e-11;        // NOLINT(readability-identifier-naming)
constexpr double ZERO_THR_ATTR = std::numeric_limits<double>::epsilon() *
                                 1e3;  // NOLINT(readability-identifier-naming)

constexpr double ALMOST_EQUAL_COUPLING_EPSILON = 1.0e-5;

using Point2D = Eigen::Vector2d;
using Point3D = Eigen::Vector3d;

using Vector2D = Eigen::Vector2d;
using Vector3D = Eigen::Vector3d;

using Index = Eigen::Index;
using VectorIndex = std::vector<double>::size_type;

using SizeType = size_t;

using MeshIndex = uint32_t;
using Index64 = uint64_t;
using SizeType32 = uint32_t;
using SizeType64 = uint64_t;

using IntAddress = uint64_t;  // Unsigned 64 bit integer to pass memory
                              // addresses

}  // namespace pycanha
