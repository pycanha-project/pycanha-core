#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

#include <Eigen/Geometry>
#include <utility>

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

CoordinateTransformation::CoordinateTransformation() noexcept = default;

CoordinateTransformation::CoordinateTransformation(
    Vector3D translation, Eigen::Matrix3d rotation) noexcept
    : _translation(std::move(translation)), _rotation(std::move(rotation)) {}

CoordinateTransformation CoordinateTransformation::from_euler(
    Vector3D translation, Vector3D euler_xyz) noexcept {
    const Eigen::Matrix3d rotation_matrix =
        (Eigen::AngleAxisd(euler_xyz.x(), Vector3D::UnitX()) *
         Eigen::AngleAxisd(euler_xyz.y(), Vector3D::UnitY()) *
         Eigen::AngleAxisd(euler_xyz.z(), Vector3D::UnitZ()))
            .toRotationMatrix();
    return {std::move(translation), rotation_matrix};
}

CoordinateTransformation CoordinateTransformation::from_translation(
    Vector3D translation) noexcept {
    return {std::move(translation), Eigen::Matrix3d::Identity()};
}

CoordinateTransformation CoordinateTransformation::from_rotation(
    const Eigen::Quaterniond& rotation) noexcept {
    return {Vector3D::Zero(), rotation.toRotationMatrix()};
}

CoordinateTransformation CoordinateTransformation::from_rotation(
    const Eigen::AngleAxisd& rotation) noexcept {
    return from_rotation(Eigen::Quaterniond(rotation));
}

const Vector3D& CoordinateTransformation::translation() const noexcept {
    return _translation;
}

const Eigen::Matrix3d& CoordinateTransformation::rotation() const noexcept {
    return _rotation;
}

void CoordinateTransformation::set_translation(Vector3D translation) noexcept {
    _translation = std::move(translation);
}

void CoordinateTransformation::set_rotation(Eigen::Matrix3d rotation) noexcept {
    _rotation = std::move(rotation);
}

bool CoordinateTransformation::is_identity() const noexcept {
    return _translation.isZero(LENGTH_TOL) &&
           _rotation.isApprox(Eigen::Matrix3d::Identity(), LENGTH_TOL);
}

Point3D CoordinateTransformation::apply(const Point3D& point) const noexcept {
    return _rotation * point + _translation;
}

Vector3D CoordinateTransformation::apply_normal(
    const Vector3D& normal) const noexcept {
    return (_rotation * normal).normalized();
}

CoordinateTransformation CoordinateTransformation::compose(
    const CoordinateTransformation& outer) const noexcept {
    return {outer._rotation * _translation + outer._translation,
            outer._rotation * _rotation};
}

CoordinateTransformation CoordinateTransformation::inverse() const noexcept {
    const Eigen::Matrix3d inverse_rotation = _rotation.inverse();
    return {-(inverse_rotation * _translation), inverse_rotation};
}

const Eigen::Matrix3d& CoordinateTransformation::linear() const noexcept {
    return _rotation;
}

}  // namespace pycanha::gmm
