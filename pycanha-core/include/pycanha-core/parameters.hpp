#pragma once
#include <Eigen/Dense>
#include <limits>
#include <vector>

namespace pycanha {

// Tolerances (SI Units)

// Constexpr should be usually lowercase (checked by clang-tidy), but I want
// these to be uppercase NOLINTBEGIN(readability-identifier-naming)
constexpr double LENGTH_TOL = 1e-9;
constexpr double ANGLE_TOL = 1e-9;
constexpr double TOL = 1e-11;
constexpr double ZERO_THR_ATTR = std::numeric_limits<double>::epsilon() * 1e3;
constexpr double ALMOST_EQUAL_COUPLING_EPSILON = 1.0e-5;
// NOLINTEND(readability-identifier-naming)

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
