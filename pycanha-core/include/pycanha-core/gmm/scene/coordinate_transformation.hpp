#pragma once

#include <Eigen/Geometry>
#include <utility>

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

class CoordinateTransformation {
  public:
    CoordinateTransformation() noexcept = default;

    explicit CoordinateTransformation(Eigen::Affine3d affine) noexcept
        : _affine(std::move(affine)) {}

    explicit CoordinateTransformation(
        Vector3D translation,
        Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity()) noexcept
        : _affine(Eigen::Affine3d(Eigen::Translation3d(std::move(translation)) *
                                  rotation)) {}

    [[nodiscard]] static CoordinateTransformation from_translation(
        Vector3D translation) noexcept {
        return CoordinateTransformation(
            Eigen::Affine3d(Eigen::Translation3d(std::move(translation))));
    }

    [[nodiscard]] static CoordinateTransformation from_rotation(
        const Eigen::Quaterniond& rotation) noexcept {
        return CoordinateTransformation(Eigen::Affine3d(rotation));
    }

    [[nodiscard]] static CoordinateTransformation from_rotation(
        const Eigen::AngleAxisd& rotation) noexcept {
        return from_rotation(Eigen::Quaterniond(rotation));
    }

    [[nodiscard]] Point3D apply(const Point3D& point) const noexcept {
        return _affine * point;
    }

    [[nodiscard]] Vector3D apply_normal(const Vector3D& normal) const noexcept {
        return (_affine.linear() * normal).normalized();
    }

    [[nodiscard]] CoordinateTransformation compose(
        const CoordinateTransformation& after) const noexcept {
        return CoordinateTransformation(after._affine * _affine);
    }

    [[nodiscard]] CoordinateTransformation inverse() const noexcept {
        return CoordinateTransformation(_affine.inverse());
    }

    [[nodiscard]] bool is_identity(
        double tolerance = LENGTH_TOL) const noexcept {
        return _affine.matrix().isApprox(Eigen::Matrix4d::Identity(),
                                         tolerance);
    }

    [[nodiscard]] const Eigen::Affine3d& affine() const noexcept {
        return _affine;
    }

    [[nodiscard]] Eigen::Matrix3d linear() const noexcept {
        return _affine.linear();
    }

  private:
    Eigen::Affine3d _affine = Eigen::Affine3d::Identity();
};

}  // namespace pycanha::gmm
