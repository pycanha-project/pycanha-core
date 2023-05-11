#pragma once
#include <Eigen/Dense>

namespace pycanha {

// Tolerances (SI Units)
constexpr double LENGTH_TOL = 1e-9;
constexpr double ANGLE_TOL = 1e-9;

using Point3D = Eigen::Vector3d;
using Vector3D = Eigen::Vector3d;

}  // namespace pycanha
