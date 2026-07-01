#pragma once

#include <Eigen/Geometry>

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

class CoordinateTransformation {
  public:
    CoordinateTransformation() noexcept;
    CoordinateTransformation(Vector3D translation,
                             Eigen::Matrix3d rotation) noexcept;

    [[nodiscard]] static CoordinateTransformation from_euler(
        Vector3D translation, Vector3D euler_xyz) noexcept;
    [[nodiscard]] static CoordinateTransformation from_translation(
        Vector3D translation) noexcept;
    [[nodiscard]] static CoordinateTransformation from_rotation(
        const Eigen::Quaterniond& rotation) noexcept;
    [[nodiscard]] static CoordinateTransformation from_rotation(
        const Eigen::AngleAxisd& rotation) noexcept;

    [[nodiscard]] const Vector3D& translation() const noexcept;
    [[nodiscard]] const Eigen::Matrix3d& rotation() const noexcept;
    void set_translation(Vector3D translation) noexcept;
    void set_rotation(Eigen::Matrix3d rotation) noexcept;

    [[nodiscard]] bool is_identity() const noexcept;
    [[nodiscard]] Point3D apply(const Point3D& point) const noexcept;
    [[nodiscard]] Vector3D apply_normal(const Vector3D& normal) const noexcept;
    [[nodiscard]] CoordinateTransformation compose(
        const CoordinateTransformation& outer) const noexcept;
    [[nodiscard]] CoordinateTransformation inverse() const noexcept;

    [[nodiscard]] const Eigen::Matrix3d& linear() const noexcept;

  private:
    Vector3D _translation = Vector3D::Zero();
    Eigen::Matrix3d _rotation = Eigen::Matrix3d::Identity();
};

}  // namespace pycanha::gmm
