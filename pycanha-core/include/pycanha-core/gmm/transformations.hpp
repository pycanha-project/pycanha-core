#pragma once
#include <Eigen/Dense>
#include <utility>

#include "../parameters.hpp"
#include "./id.hpp"
#include "./trimesh.hpp"

namespace pycanha::gmm {

enum class TransformOrder {
    TRANSLATION_THEN_ROTATION,
    ROTATION_THEN_TRANSLATION
};

// This is a very inefficient class!!
// TODO: Make this class more efficient

/**
 * @class CoordinateTransformation
 * @brief A class that represents a coordinate transformation.
 *
 * This class uses a translation vector, a rotation vector and the order of
 * operations (translation then rotation, or rotation then translation) to
 * transform a point in 3D space.
 */
class CoordinateTransformation : public UniqueID {
  public:
    /**
     * @brief Default constructor for CoordinateTransformation class.
     *
     * Initializes translation and rotation vectors to zero, and
     * sets the transformation order to TRANSLATION_THEN_ROTATION.
     */
    CoordinateTransformation()
        : _translation(Vector3D::Zero()),
          _rotation_matrix(Eigen::Matrix3d::Identity()),
          _order(TransformOrder::TRANSLATION_THEN_ROTATION) {}

    /**
     * @brief Constructor for CoordinateTransformation class.
     *
     * @param translation The translation vector with (X, Y, Z) coordinates.
     * @param rotation The rotation vector with rotation around (X, Y, Z)
     * @param order The order of the transformations.
     */
    explicit CoordinateTransformation(Vector3D translation, Vector3D rotation,
                                      TransformOrder order)
        : _translation(std::move(translation)), _order(order) {
        _rotation_matrix = create_rotation_matrix(std::move(rotation));
    }

    /**
     * @brief Constructor for CoordinateTransformation class.
     *
     * @param translation The translation vector with (X, Y, Z) coordinates.
     * @param rotation_matrix The rotation matrix.
     * @param order The order of the transformations.
     */
    explicit CoordinateTransformation(Vector3D translation,
                                      Eigen::Matrix3d rotation_matrix,
                                      TransformOrder order)
        : _translation(std::move(translation)),
          _rotation_matrix(std::move(rotation_matrix)),
          _order(order) {}

    /**
     * @brief Transforms a point using the stored translation and rotation
     * vectors.
     *
     * @param point The point to be transformed.
     * @return The transformed point.
     */
    [[nodiscard]] Vector3D transform_point(const Vector3D& point) const {
        switch (_order) {
            case TransformOrder::TRANSLATION_THEN_ROTATION:
                return _rotation_matrix * (point + _translation);
            case TransformOrder::ROTATION_THEN_TRANSLATION:
                return _rotation_matrix * point + _translation;
        }
        throw std::invalid_argument("Invalid transform order");
    }

    /**
     * @brief Create rotation matrix.
     * @return The rotation matrix.
     */
    [[nodiscard]] static Eigen::Matrix3d create_rotation_matrix(
        Vector3D rotation_angles) {
        Eigen::Matrix3d rotation_matrix;
        // This is XYZ rotation
        rotation_matrix =
            Eigen::AngleAxisd(rotation_angles.z(), Vector3D::UnitZ()) *
            Eigen::AngleAxisd(rotation_angles.y(), Vector3D::UnitY()) *
            Eigen::AngleAxisd(rotation_angles.x(), Vector3D::UnitX());
        return rotation_matrix;
    }

    /**
     * @brief Transform trimesh points inplace.
     * @param trimesh The trimesh to be transformed.
     */
    void transform_point_list_inplace(VerticesList* points) const {
        switch (_order) {
            case TransformOrder::TRANSLATION_THEN_ROTATION:
                // Here we first add the translation to all points and then
                // apply the rotation.
                points->rowwise() += _translation.transpose();
                *points = (_rotation_matrix * points->transpose()).transpose();
                break;
            case TransformOrder::ROTATION_THEN_TRANSLATION:
                // Here we first rotate all points and then add the
                // translation.
                *points = (_rotation_matrix * points->transpose()).transpose();
                points->rowwise() += _translation.transpose();
                break;
            default:
                throw std::invalid_argument("Invalid transform order");
        }
    }
    /**
     * @brief Transform point list.
     * @param points The points to be transformed.
     * @return The transformed points.
     */
    [[nodiscard]] VerticesList transform_point_list(
        const VerticesList& points) const {
        // Copy the point to a new list
        VerticesList transformed_points(points);
        transform_point_list_inplace(&transformed_points);
        return transformed_points;
    }

    /**
     * @brief Transform trimesh inplace.
     * @param trimesh The trimesh to be transformed.
     */
    void transform_trimesh_inplace(TriMesh& trimesh) const {
        transform_point_list_inplace(&trimesh.get_vertices());
    }

    /**
     * @brief Transform trimesh.
     * @param trimesh The trimesh to be transformed.
     */
    [[nodiscard]] TriMesh transform_trimesh(const TriMesh& trimesh) const {
        TriMesh transformed_trimesh(trimesh);
        transform_trimesh_inplace(transformed_trimesh);
        return transformed_trimesh;
    }

    /**
     * @brief Creates a new transformation equivalent to applying other and then
     * this.
     * @param other The other transformation.
     * @return The chained transformation.
     */
    [[nodiscard]] CoordinateTransformation chain(
        const CoordinateTransformation& first_transf) const {
        // First, we change the transformations to Rotation then Translation
        Vector3D translation_1;
        const Eigen::Matrix3d& rotation_1 = first_transf._rotation_matrix;
        if (first_transf._order == TransformOrder::TRANSLATION_THEN_ROTATION) {
            translation_1 = rotation_1 * first_transf._translation;
        } else {
            translation_1 = first_transf._translation;
        }

        Vector3D translation_2;
        const Eigen::Matrix3d& rotation_2 = _rotation_matrix;
        if (_order == TransformOrder::TRANSLATION_THEN_ROTATION) {
            translation_2 = rotation_2 * _translation;
        } else {
            translation_2 = _translation;
        }

        // Prot1 = (R1 * P) + T1
        // Prot2 = (R2 * Prot1) + T2
        // Prot2 = R2 * (R1 * P + T1) + T2
        // Prot2 = R2*R1*P + R2*T1 + T2
        // Prot2 = R'*P + T'
        // R' = R2 * R1
        // T' = R2 * T1 + T2
        return CoordinateTransformation(
            Vector3D(rotation_2 * translation_1 + translation_2),
            Eigen::Matrix3d(rotation_2 * rotation_1),
            TransformOrder::ROTATION_THEN_TRANSLATION);
    }

    /// Getters
    /**
     * @brief Gets the translation vector.
     *
     * @return The translation vector.
     */
    [[nodiscard]] const Vector3D& get_translation() const {
        return _translation;
    }

    /**
     * @brief Gets the rotation matrix.
     *
     * @return The rotation matrix.
     */
    [[nodiscard]] const Eigen::Matrix3d& get_rotation_matrix() const {
        return _rotation_matrix;
    }

    /**
     * @brief Gets the order of the transformations.
     *
     * @return The order of the transformations.
     */
    [[nodiscard]] TransformOrder get_order() const { return _order; }

    /// Setters
    /**
     * @brief Sets the translation vector.
     *
     * @param translation The new translation vector.
     */
    void set_translation(Vector3D translation) {
        _translation = std::move(translation);
    }

    /**
     * @brief Sets the rotation matrix.
     *
     * @param rotation_matrix The new rotation matrix.
     */
    void set_rotation_matrix(Eigen::Matrix3d rotation_matrix) {
        _rotation_matrix = std::move(rotation_matrix);
    }

    /**
     * @brief Sets the rotation matrix.
     *
     * @param rotation The new rotation matrix.
     */
    void set_rotation_angles(Vector3D rotation) {
        _rotation_matrix = create_rotation_matrix(std::move(rotation));
    }

    /**
     * @brief Sets the order of the transformations.
     *
     * @param order The new order of the transformations.
     */
    void set_order(TransformOrder order) { _order = order; }

  private:
    Vector3D _translation;             ///< The translation vector.
    Eigen::Matrix3d _rotation_matrix;  ///< The rotation matrix.
    TransformOrder _order;             ///< The order of the transformations.
};

}  // namespace pycanha::gmm
