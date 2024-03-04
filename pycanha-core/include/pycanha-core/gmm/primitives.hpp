#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numbers>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "../parameters.hpp"
#include "./callbacks.hpp"
#include "./geometryutils.hpp"
#include "./id.hpp"
#include "./thermalmesh.hpp"
#include "./transformations.hpp"
#include "./trimesh.hpp"

using namespace pycanha;  // NOLINT
// using pycanha::gmm::trimesher::print_point3d;
// using pycanha::gmm::trimesher::print_point2d;

namespace pycanha::gmm {

/**
 * @class Primitive
 * @brief A geometry primitive.
 *
 */
class Primitive : public UniqueID, public GeometryUpdateCallback {
  public:
    /**
     * @brief Calculates the minimum distance from a point to the surface of the
     * primitive. The distance is calculated to the real surface.
     * @param point The point to calculate the distance to.
     * @return The minimum distance from the point to this primitive.
     */
    [[nodiscard]] virtual double distance(const Point3D& point) const = 0;

    /**
     * @brief Calculates the minimum distance and the jacobian from a point to
     * the extended surface. The extended surface depend on the type of
     * primitive. These are the extended surfaces:
     * - Triangle: The extended surface is the plane that contains the triangle.
     * - Rectangle: The extended surface is the plane that contains the
     * rectangle.
     * - Quadrilateral: The extended surface is the plane that contains the
     * quadrilateral.
     * - Disc: The extended surface is the cylinder that contains the disc.
     * - Cylinder: The extended surface is an infinite cylinder (without caps)
     * - Sphere: The extended surface is the whole sphere.
     *
     * The Jacobian is also calculated and defined as:
     * Jac = [dD/dx, dD/dy, dD/dz]
     *
     * @param point The point to calculate the distance to.
     * @return Array of 4 elements. First element is the disntance, the other 3
     * are the jacobian.
     */
    [[nodiscard]] virtual std::array<double, 4>
    distance_jacobian_cutted_surface(const Point3D& point) const = 0;

    /**
     * @brief Calculates the minimum distance and the jacobian from a point to
     * the primitive when used as a cutting tool. When primitives are used as
     * cutting tools, the surface might not be the same. For example, a cylinder
     * when used as cutting tool, it contains the caps. The cutting surfaces are
     * defined as:
     * - Triangle: Not defined (yet?). Raise exception.
     * - Rectangle: Not defined (yet?). Raise exception.
     * - Quadrilateral: Not defined (yet?). Raise exception.
     * - Disc: Not defined (yet?). Raise exception.
     * - Cylinder: The surface of the cylinder with the caps. TODO: Decide if
     * allow for sector of cylinder.
     * - Sphere: The whole sphere. TODO: Decide if allow for sector of sphere.
     *
     * The Jacobian is also calculated and defined as:
     * Jac = [dD/dx, dD/dy, dD/dz]
     *
     * @param point The point to calculate the distance to.
     * @return Array of 4 elements. First element is the disntance, the other 3
     * are the jacobian.
     */
    [[nodiscard]] virtual std::array<double, 4>
    distance_jacobian_cutting_surface(const Point3D& point) const = 0;

    /**
     * @brief Validate the primitive.
     * @return If the primitive is valid
     */
    [[nodiscard]] virtual bool is_valid() const = 0;

    /**
     * @brief 3D coordinates of a point ON the primitive surface from 2D
     * coordinates. Note: The method is not valid for the following primitives:
     * - Sphere
     * @param p2d The 2D coordinates of the point ON the surface of the
     * primitive. rectangle.
     * @return 3D point ON the surface of the primitive.
     */
    [[nodiscard]] virtual Point3D from_2d_to_3d(const Point2D& p2d) const = 0;

    /**
     * @brief 2D coordinates of a point ON the primitive surface from 3D
     * coordinates. Note: It is assumed that the 3D point is ON the surface of
     * the primitive. The behaviour is undefined if the point is not on the
     * surface. Note: The method is not valid for the following primitives:
     * - Sphere
     * @param p3d The 3D coordinates of the point ON the surface of the
     * primitive. rectangle.
     * @return 2D coordinates.
     */
    [[nodiscard]] virtual Point2D from_3d_to_2d(const Point3D& p3d) const = 0;

    /**
     * @brief Creates the triangular mesh from the primitive.
     * @param thermal_mesh The thermal mesh to use.
     * @param tolerance The tolerance to use.
     * @return The mesh.
     */
    [[nodiscard]] virtual TriMesh create_mesh(const ThermalMesh& thermal_mesh,
                                              double tolerance) const = 0;

    /**
     * @brief Creates a new transformed primitive.
     * @param transformation The transformation to apply.
     * @return The transformed primitive.
     */
    [[nodiscard]] virtual std::shared_ptr<Primitive> transform(
        const CoordinateTransformation& transformation) const = 0;

    /**
     * @brief Returns the ID of the face in the thermal mesh that contains the
     * given UV coordinates. Return -1 if UV point is outside the primitive.
     * @param thermal_mesh The thermal mesh to use.
     * @param uv The UV coordinates to search for.
     * @return The ID of the face containing the UV coordinates.
     *
     * TODO(TEST): Untested return of -1.
     */
    [[nodiscard]] virtual MeshIndex get_faceid_from_uv(
        const ThermalMesh& thermal_mesh, const Point2D& uv) const = 0;

    /**
     * @brief Returns the ID of the face in the thermal mesh that contains the
     * given point. Point is assumed to be on the surface of the primitive.
     * @param thermal_mesh The thermal mesh to use.
     * @param point The point to search for.
     * @return The ID of the face containing the point.
     */
    [[nodiscard]] MeshIndex get_faceid_from_point(
        const ThermalMesh& thermal_mesh, const Point3D& point) const {
        return get_faceid_from_uv(thermal_mesh, from_3d_to_2d(point));
    }

    /**
     * @brief Iterate over all the edges in the mesh and assign each one to the
     * correspondent face. Mesh points are assumed to be on UV coords.
     * @param tolerance The mesh.
     * @param thermal_mesh The thermal mesh to use.
     *
     * Note: The face edges won't be sorted.
     */
    void reconstruct_face_edges_2d(const TriMeshPtr& trimesh,
                                   const ThermalMesh& thermal_mesh) const {
        // Number of pair of faces in the mesh.
        const MeshIndex n_faces = thermal_mesh.get_number_of_pair_faces();
        FaceEdges face_edges(n_faces);

        constexpr double eps = LENGTH_TOL * 10.0;
        constexpr VectorIndex n_offsets = 4;

        const std::array<Vector2D, n_offsets> offsets = {
            Vector2D(-eps, -eps), Vector2D(eps, -eps), Vector2D(eps, eps),
            Vector2D(-eps, eps)};

        // Iterate over all the edges in the mesh.
        for (MeshIndex edge_id = 0; edge_id < trimesh->get_edges().size();
             ++edge_id) {
            const auto& edge = trimesh->get_edges()[edge_id];
            const Point2D p1(trimesh->get_vertices()(edge[0], 0),
                             trimesh->get_vertices()(edge[0], 1));
            const Point2D p2(trimesh->get_vertices()(edge.tail<1>()(0), 0),
                             trimesh->get_vertices()(edge.tail<1>()(0), 1));

            std::array<int64_t, n_offsets> p1_faces = {-1, -1, -1, -1};
            std::array<int64_t, n_offsets> p2_faces = {-1, -1, -1, -1};

            for (VectorIndex i = 0; i < n_offsets; ++i) {
                // get_faceid_from_uv will throw if the point is outside the
                // primitive. We catch the exception and set the face to -1.
                // TODO: This might be slow...
                // NOLINTBEGIN(bugprone-empty-catch)
                // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
                try {
                    p1_faces[i] =
                        get_faceid_from_uv(thermal_mesh, p1 + offsets[i]);
                } catch (const std::exception& /*e*/) {
                    // p1_faces[i] = -1
                }

                try {
                    p2_faces[i] =
                        get_faceid_from_uv(thermal_mesh, p2 + offsets[i]);
                } catch (const std::exception& /*e*/) {
                    // p2_faces[i] = -1
                }
                // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
                // NOLINTEND(bugprone-empty-catch)
            }

            // Sort the arrays
            std::sort(p1_faces.begin(), p1_faces.end());
            std::sort(p2_faces.begin(), p2_faces.end());
            std::array<int64_t, n_offsets> common_faces = {-1, -1, -1, -1};

            // NOLINTBEGIN(readability-qualified-auto)
            auto last_common_face_it = std::set_intersection(
                p1_faces.begin(), p1_faces.end(), p2_faces.begin(),
                p2_faces.end(), common_faces.begin());
            // Output the intersection
            for (auto it = common_faces.begin(); it != last_common_face_it;
                 ++it) {
                if (*it != -1) {
                    auto index = static_cast<VectorIndex>(*it) / 2;

                    // Get the current size of the matrix at face_edges[index]
                    const Index current_size = face_edges[index].size();

                    // Resize the matrix to accommodate one more element
                    face_edges[index].conservativeResize(current_size + 1,
                                                         Eigen::NoChange);

                    // Append edge_id to the matrix
                    face_edges[index](current_size, 0) = edge_id;
                }
            }
            // NOLINTEND(readability-qualified-auto)
        }

        trimesh->set_faces_edges(face_edges);
    }

    // Constructor
    Primitive() = default;

    // Virtual Destructor
    virtual ~Primitive() = default;

    // Delete copy constructor and copy assignment operator
    Primitive(const Primitive&) = delete;
    Primitive& operator=(const Primitive&) = delete;

    // Explicitly define or delete move constructor and move assignment operator
    Primitive(Primitive&&) = delete;
    Primitive& operator=(Primitive&&) = delete;
};

/**
 * @class Triangle
 * @brief A class representing a triangle in 3D space.
 *
 *            p3
 *            /\  ^
 *           /  \  \ v2
 *          /    \
 *        p1------p2
 *          --> v1
 */
class Triangle : public Primitive {
  public:
    /**
     * @brief Constructs a triangle with the three vertices.
     * @param p1 First vertex of the triangle.
     * @param p2 Second vertex of the triangle.
     * @param p3 Third vertex of the triangle.
     */
    Triangle(Point3D p1, Point3D p2, Point3D p3)
        : _p1(std::move(p1)), _p2(std::move(p2)), _p3(std::move(p3)) {}

    /**
     * @brief Checks if the triangle is valid.
     * @return `true` if the triangle is valid, `false` otherwise.
     */
    [[nodiscard]] bool is_valid() const override;

    /**
     * @brief Returns the first vertex of the triangle.
     * @return The first vertex of the triangle.
     */
    [[nodiscard]] const Point3D& get_p1() const { return _p1; }

    /**
     * @brief Returns the second vertex of the triangle.
     * @return The second vertex of the triangle.
     */
    [[nodiscard]] const Point3D& get_p2() const { return _p2; }

    /**
     * @brief Returns the third vertex of the triangle.
     * @return The third vertex of the triangle.
     */
    [[nodiscard]] const Point3D& get_p3() const { return _p3; }

    /**
     * @brief Returns the first direction v1 = p2 - p1.
     * @return The first direction of the triangle.
     */
    [[nodiscard]] Vector3D v1() const { return _p2 - _p1; }

    /**
     * @brief Returns the second direction v2 = p3 - p2.
     * @return The second direction of the triangle.
     */
    [[nodiscard]] Vector3D v2() const { return _p3 - _p2; }

    /**
     * @brief Updates the first vertex of the triangle.
     * @param p1 The new first vertex of the triangle.
     */
    void set_p1(Point3D p1) {
        _p1 = std::move(p1);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the second vertex of the triangle.
     * @param p2 The new second vertex of the triangle.
     */
    void set_p2(Point3D p2) {
        _p2 = std::move(p2);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the third vertex of the triangle.
     * @param p3 The new third vertex of the triangle.
     */
    void set_p3(Point3D p3) {
        _p3 = std::move(p3);
        callback_with_id(get_id());
    }

    [[nodiscard]] double distance(const Point3D& point) const override;

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutted_surface(
        const Point3D& point) const override;

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutting_surface(
        const Point3D& /*p3d*/) const override {
        throw std::logic_error("Not implemented");
    }

    [[nodiscard]] Point3D from_2d_to_3d(const Point2D& p2d) const override;

    [[nodiscard]] Point2D from_3d_to_2d(const Point3D& p3d) const override;

    [[nodiscard]] TriMesh create_mesh(const ThermalMesh& thermal_mesh,
                                      double tolerance) const override;

    [[nodiscard]] MeshIndex get_faceid_from_uv(
        const ThermalMesh& thermal_mesh,
        const Point2D& point_uv) const override;

    [[nodiscard]] std::shared_ptr<Primitive> transform(
        const CoordinateTransformation& transformation) const override {
        return std::make_shared<Triangle>(transformation.transform_point(_p1),
                                          transformation.transform_point(_p2),
                                          transformation.transform_point(_p3));
    };

  private:
    Point3D _p1;  ///< First vertex of the triangle.
    Point3D _p2;  ///< Second vertex of the triangle.
    Point3D _p3;  ///< Third vertex of the triangle.
};

/**
 * @class Rectangle
 * @brief A class representing a rectangle in 3D space.
 *
 *          p3 ------------
 *        ^    |          |
 *     v2 |    |          |
 *          p1 ------------ p2
 *                ---> v1
 */
class Rectangle : public Primitive {
  public:
    /**
     * @brief Constructs a rectangle with the three vertices.
     * @param p1 First vertex of the rectangle.
     * @param p2 Second vertex of the rectangle.
     * @param p3 Third vertex of the rectangle.
     */
    Rectangle(Point3D p1, Point3D p2, Point3D p3)
        : _p1(std::move(p1)), _p2(std::move(p2)), _p3(std::move(p3)) {}

    /**
     * @brief Returns the first vertex of the rectangle.
     * @return The first vertex of the rectangle.
     */
    [[nodiscard]] const Point3D& get_p1() const { return _p1; }

    /**
     * @brief Returns the second vertex of the rectangle.
     * @return The second vertex of the rectangle.
     */
    [[nodiscard]] const Point3D& get_p2() const { return _p2; }

    /**
     * @brief Returns the third vertex of the rectangle.
     * @return The third vertex of the rectangle.
     */
    [[nodiscard]] const Point3D& get_p3() const { return _p3; }

    /**
     * @brief Returns the first direction v1 = p2 - p1.
     * @return The first direction of the rectangle.
     */
    [[nodiscard]] Vector3D v1() const { return _p2 - _p1; }

    /**
     * @brief Returns the second direction v2 = p3 - p1.
     * @return The second direction of the rectangle.
     */
    [[nodiscard]] Vector3D v2() const { return _p3 - _p1; }

    /**
     * @brief Updates the first vertex of the rectangle.
     * @param p1 The new first vertex of the rectangle.
     */
    void set_p1(Point3D p1) {
        _p1 = std::move(p1);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the second vertex of the rectangle.
     * @param p2 The new second vertex of the rectangle.
     */
    void set_p2(Point3D p2) {
        _p2 = std::move(p2);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the third vertex of the rectangle.
     * @param p3 The new third vertex of the rectangle.
     */
    void set_p3(Point3D p3) {
        _p3 = std::move(p3);
        callback_with_id(get_id());
    }

    [[nodiscard]] bool is_valid() const override;

    [[nodiscard]] double distance(const Point3D& point) const override;

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutted_surface(
        const Point3D& point) const override;

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutting_surface(
        const Point3D& /*p3d*/) const override {
        throw std::logic_error("Not implemented");
    }

    [[nodiscard]] Point2D from_3d_to_2d(const Point3D& p3d) const override;

    [[nodiscard]] Point3D from_2d_to_3d(const Point2D& p2d) const override;

    [[nodiscard]] TriMesh create_mesh(const ThermalMesh& thermal_mesh,
                                      double tolerance) const override;

    [[nodiscard]] MeshIndex get_faceid_from_uv(
        const ThermalMesh& thermal_mesh,
        const Point2D& point_uv) const override;

    [[nodiscard]] std::shared_ptr<Primitive> transform(
        const CoordinateTransformation& transformation) const override {
        return std::make_shared<Rectangle>(transformation.transform_point(_p1),
                                           transformation.transform_point(_p2),
                                           transformation.transform_point(_p3));
    };

  private:
    Point3D _p1;  ///< First vertex of the rectangle.
    Point3D _p2;  ///< Second vertex of the rectangle.
    Point3D _p3;  ///< Third vertex of the rectangle.
};

/**
 * @class Quadrilateral
 * @brief A class representing a quadrilateral in 3D space.
 *
 *          p4 --------- p3
 *        ^    |        \
 *     v2 |    |         \
 *          p1 ----------- p2
 *                ---> v1
 */
class Quadrilateral : public Primitive {
  public:
    /**
     * @brief Constructs a quadrilateral with the four vertices.
     * @param p1 First vertex of the quadrilateral.
     * @param p2 Second vertex of the quadrilateral.
     * @param p3 Third vertex of the quadrilateral.
     * @param p4 Fourth vertex of the quadrilateral.
     */
    Quadrilateral(Point3D p1, Point3D p2, Point3D p3, Point3D p4)
        : _p1(std::move(p1)),
          _p2(std::move(p2)),
          _p3(std::move(p3)),
          _p4(std::move(p4)) {}

    /**
     * @brief Checks if the quadrilateral is valid.
     * @return `true` if the quadrilateral is valid, `false` otherwise.
     */
    [[nodiscard]] bool is_valid() const override;

    /**
     * @brief Returns the first vertex of the quadrilateral.
     * @return The first vertex of the quadrilateral.
     */
    [[nodiscard]] const Point3D& get_p1() const { return _p1; }

    /**
     * @brief Returns the second vertex of the quadrilateral.
     * @return The second vertex of the quadrilateral.
     */
    [[nodiscard]] const Point3D& get_p2() const { return _p2; }

    /**
     * @brief Returns the third vertex of the quadrilateral.
     * @return The third vertex of the quadrilateral.
     */
    [[nodiscard]] const Point3D& get_p3() const { return _p3; }

    /**
     * @brief Returns the fourth vertex of the quadrilateral.
     * @return The fourth vertex of the quadrilateral.
     */
    [[nodiscard]] const Point3D& get_p4() const { return _p4; }

    /**
     * @brief Returns the first direction v1 = p2 - p1.
     * @return The first direction of the quadrilateral.
     */
    [[nodiscard]] Vector3D v1() const { return _p2 - _p1; }

    /**
     * @brief Returns the second direction v2 = p4 - p1.
     * @return The second direction of the quadrilateral.
     */
    [[nodiscard]] Vector3D v2() const { return _p4 - _p1; }

    /**
     * @brief Updates the first vertex of the quadrilateral.
     * @param p1 The new first vertex of the quadrilateral.
     */
    void set_p1(Point3D p1) {
        _p1 = std::move(p1);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the second vertex of the quadrilateral.
     * @param p2 The new second vertex of the quadrilateral.
     */
    void set_p2(Point3D p2) {
        _p2 = std::move(p2);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the third vertex of the quadrilateral.
     * @param p3 The new third vertex of the quadrilateral.
     */
    void set_p3(Point3D p3) {
        _p3 = std::move(p3);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the fourth vertex of the quadrilateral.
     * @param p4 The new fourth vertex of the quadrilateral.
     */
    void set_p4(Point3D p4) {
        _p4 = std::move(p4);
        callback_with_id(get_id());
    }

    [[nodiscard]] double distance(const Point3D& p3d) const override;

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutted_surface(
        const Point3D& point) const override;

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutting_surface(
        const Point3D& /*p3d*/) const override {
        throw std::logic_error("Not implemented");
    }

    [[nodiscard]] Point2D from_3d_to_2d(const Point3D& p3d) const override;

    [[nodiscard]] Point3D from_2d_to_3d(const Point2D& p2d) const override;

    // TODO(IMPLEMENTATION)
    [[nodiscard]] TriMesh create_mesh(const ThermalMesh& /*thermal_mesh*/,
                                      double /*tolerance*/) const override {
        throw std::logic_error("Not implemented");
    };

    // TODO(IMPLEMENTATION)
    [[nodiscard]] MeshIndex get_faceid_from_uv(
        const ThermalMesh& /*thermal_mesh*/,
        const Point2D& /*uv*/) const override {
        throw std::logic_error("Not implemented");
    }

    [[nodiscard]] std::shared_ptr<Primitive> transform(
        const CoordinateTransformation& transformation) const override {
        return std::make_shared<Quadrilateral>(
            transformation.transform_point(_p1),
            transformation.transform_point(_p2),
            transformation.transform_point(_p3),
            transformation.transform_point(_p4));
    };

  private:
    Point3D _p1;  ///< First vertex of the quadrilateral.
    Point3D _p2;  ///< Second vertex of the quadrilateral.
    Point3D _p3;  ///< Third vertex of the quadrilateral.
    Point3D _p4;  ///< Fourth vertex of the quadrilateral.
};

/**
 * @class Disc
 * @brief A class representing a disc in 3D space.
 *
 * The disc is defined by three points (p1, p2, p3) and has an inner and
 * outer radius. It spans from a start angle to an end angle.
 */
class Disc : public Primitive {
  public:
    /**
     * @brief Constructs a disc with the three vertices, inner radius,
     * outer radius, start angle and end angle.
     * @param p1 Center of the disc.
     * @param p2 Second vertex of the revolution axis together with p1 the disc.
     * @param p3 Second vertex of the starting axis together with p1 of the
     * disc.
     * @param inner_radius Inner radius of the disc.
     * @param outer_radius Outer radius of the disc.
     * @param start_angle  Start angle of the disc.
     * @param end_angle    End angle of the disc.
     */
    Disc(Point3D p1, Point3D p2, Point3D p3, double inner_radius,
         double outer_radius, double start_angle, double end_angle)
        : _p1(std::move(p1)),
          _p2(std::move(p2)),
          _p3(std::move(p3)),
          _inner_radius(inner_radius),
          _outer_radius(outer_radius),
          _start_angle(start_angle),
          _end_angle(end_angle) {}

    /**
     * @brief Returns the first point of the disc.
     * @return The first point of the disc.
     */
    [[nodiscard]] const Point3D& get_p1() const { return _p1; }

    /**
     * @brief Returns the second point of the disc.
     * @return The second point of the disc.
     */
    [[nodiscard]] const Point3D& get_p2() const { return _p2; }

    /**
     * @brief Returns the third point of the disc.
     * @return The third point of the disc.
     */
    [[nodiscard]] const Point3D& get_p3() const { return _p3; }

    /**
     * @brief Returns the inner radius of the disc.
     * @return The inner radius of the disc.
     */
    [[nodiscard]] double get_inner_radius() const { return _inner_radius; }

    /**
     * @brief Returns the outer radius of the disc.
     * @return The outer radius of the disc.
     */
    [[nodiscard]] double get_outer_radius() const { return _outer_radius; }

    /**
     * @brief Returns the start angle of the disc.
     * @return The start angle of the disc.
     */
    [[nodiscard]] double get_start_angle() const { return _start_angle; }

    /**
     * @brief Returns the end angle of the disc.
     * @return The end angle of the disc.
     */
    [[nodiscard]] double get_end_angle() const { return _end_angle; }

    /**
     * @brief Returns the first direction v1 = p2 - p1.
     * @return The first direction of the triangle.
     */
    [[nodiscard]] Vector3D v1() const { return _p3 - _p1; }

    /**
     * @brief Returns the second direction v2 = p3 - p2.
     * @return The second direction of the triangle.
     */
    [[nodiscard]] Vector3D v2() const { return v1().cross(_p3 - _p2); }

    /**
     * @brief Updates the first point of the disc.
     * @param p1 The new first point of the disc.
     */
    void set_p1(Point3D p1) {
        _p1 = std::move(p1);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the second point of the disc.
     * @param p2 The new second point of the disc.
     */
    void set_p2(Point3D p2) {
        _p2 = std::move(p2);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the third point of the disc.
     * @param p3 The new third point of the disc.
     */
    void set_p3(Point3D p3) {
        _p3 = std::move(p3);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the inner radius of the disc.
     * @param inner_radius The new inner radius of the disc.
     */
    void set_inner_radius(double inner_radius) {
        _inner_radius = inner_radius;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the outer radius of the disc.
     * @param outer_radius The new outer radius of the disc.
     */
    void set_outer_radius(double outer_radius) {
        _outer_radius = outer_radius;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the start angle of the disc.
     * @param start_angle The new start angle of the disc.
     */
    void set_start_angle(double start_angle) {
        _start_angle = start_angle;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the end angle of the disc.
     * @param end_angle The new end angle of the disc.
     */
    void set_end_angle(double end_angle) {
        _end_angle = end_angle;
        callback_with_id(get_id());
    }

    [[nodiscard]] bool is_valid() const override;

    [[nodiscard]] double distance(const Point3D& /*point*/) const override {
        throw std::logic_error("Not implemented");
    };

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutted_surface(
        const Point3D& /*point*/) const override {
        throw std::logic_error("Not implemented");
    };

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutting_surface(
        const Point3D& /*p3d*/) const override {
        throw std::logic_error("Not implemented");
    }

    [[nodiscard]] Point3D from_2d_to_3d(const Point2D& p2d) const override;

    [[nodiscard]] Point2D from_3d_to_2d(const Point3D& p3d) const override;

    [[nodiscard]] TriMesh create_mesh(const ThermalMesh& thermal_mesh,
                                      double tolerance) const override;

    [[nodiscard]] MeshIndex get_faceid_from_uv(
        const ThermalMesh& thermal_mesh,
        const Point2D& point_uv) const override;

    [[nodiscard]] std::shared_ptr<Primitive> transform(
        const CoordinateTransformation& transformation) const override {
        return std::make_shared<Disc>(transformation.transform_point(_p1),
                                      transformation.transform_point(_p2),
                                      transformation.transform_point(_p3),
                                      _inner_radius, _outer_radius,
                                      _start_angle, _end_angle);
    };

  private:
    Point3D _p1;           ///< First vertex of the disc.
    Point3D _p2;           ///< Second vertex of the disc.
    Point3D _p3;           ///< Third vertex of the disc.
    double _inner_radius;  ///< Inner radius of the disc.
    double _outer_radius;  ///< Outer radius of the disc.
    double _start_angle;   ///< Start angle of the disc.
    double _end_angle;     ///< End angle of the disc.
};

/**
 * @class Cylinder
 * @brief A class representing a cylinder in 3D space.
 *
 * The cylinder is defined by three points (p1, p2, p3), a radius, and
 * spans from a start angle to an end angle.
 */
class Cylinder : public Primitive {
  public:
    /**
     * @brief Constructs a cylinder with the three vertices, radius,
     * start angle and end angle.
     * @param p1 First point of the cylinder.
     * @param p2 Second point of the cylinder.
     * @param p3 Third point of the cylinder.
     * @param radius Radius of the cylinder.
     * @param start_angle Start angle of the cylinder.
     * @param end_angle End angle of the cylinder.
     */
    Cylinder(Point3D p1, Point3D p2, Point3D p3, double radius,
             double start_angle, double end_angle)
        : _p1(std::move(p1)),
          _p2(std::move(p2)),
          _p3(std::move(p3)),
          _radius(radius),
          _start_angle(start_angle),
          _end_angle(end_angle) {}

    /**
     * @brief Returns the first point of the cylinder.
     * @return The first point of the cylinder.
     */
    [[nodiscard]] const Point3D& get_p1() const { return _p1; }

    /**
     * @brief Returns the second point of the cylinder.
     * @return The second point of the cylinder.
     */
    [[nodiscard]] const Point3D& get_p2() const { return _p2; }

    /**
     * @brief Returns the third point of the cylinder.
     * @return The third point of the cylinder.
     */
    [[nodiscard]] const Point3D& get_p3() const { return _p3; }

    /**
     * @brief Returns the radius of the cylinder.
     * @return The radius of the cylinder.
     */
    [[nodiscard]] double get_radius() const { return _radius; }

    /**
     * @brief Returns the start angle of the cylinder.
     * @return The start angle of the cylinder.
     */
    [[nodiscard]] double get_start_angle() const { return _start_angle; }

    /**
     * @brief Returns the end angle of the cylinder.
     * @return The end angle of the cylinder.
     */
    [[nodiscard]] double get_end_angle() const { return _end_angle; }
    /**
     * @brief Updates the first point of the cylinder.
     * @param p1 The new first point of the cylinder.
     */
    void set_p1(Point3D p1) {
        _p1 = std::move(p1);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the second point of the cylinder.
     * @param p2 The new second point of the cylinder.
     */
    void set_p2(Point3D p2) {
        _p2 = std::move(p2);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the third point of the cylinder.
     * @param p3 The new third point of the cylinder.
     */
    void set_p3(Point3D p3) {
        _p3 = std::move(p3);
        callback_with_id(get_id());
    }
    /**
     * @brief Updates the radius of the cylinder.
     * @param radius The new radius of the cylinder.
     */
    void set_radius(double radius) {
        _radius = radius;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the start angle of the cylinder.
     * @param start_angle The new start angle of the cylinder.
     */
    void set_start_angle(double start_angle) {
        _start_angle = start_angle;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the end angle of the cylinder.
     * @param end_angle The new end angle of the cylinder.
     */
    void set_end_angle(double end_angle) {
        _end_angle = end_angle;
        callback_with_id(get_id());
    }

    [[nodiscard]] bool is_valid() const override;

    [[nodiscard]] double distance(const Point3D& point) const override;

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutted_surface(
        const Point3D& point) const override;

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutting_surface(
        const Point3D& /*p3d*/) const override {
        throw std::logic_error("Not implemented");
    }

    [[nodiscard]] Point3D from_2d_to_3d(const Point2D& p2d) const override;

    [[nodiscard]] Point2D from_3d_to_2d(const Point3D& p3d) const override;

    [[nodiscard]] TriMesh create_mesh(const ThermalMesh& thermal_mesh,
                                      double tolerance) const override;

    [[nodiscard]] MeshIndex get_faceid_from_uv(
        const ThermalMesh& thermal_mesh,
        const Point2D& point_uv) const override;

    [[nodiscard]] std::shared_ptr<Primitive> transform(
        const CoordinateTransformation& transformation) const override {
        return std::make_shared<Cylinder>(transformation.transform_point(_p1),
                                          transformation.transform_point(_p2),
                                          transformation.transform_point(_p3),
                                          _radius, _start_angle, _end_angle);
    };

  private:
    Point3D _p1;          ///< First point of the cylinder.
    Point3D _p2;          ///< Second point of the cylinder.
    Point3D _p3;          ///< Third point of the cylinder.
    double _radius;       ///< Radius of the cylinder.
    double _start_angle;  ///< Start angle of the cylinder.
    double _end_angle;    ///< End angle of the cylinder.
};

/**
 * @class Cone
 * @brief A class representing a cone in 3D space.
 */
class Cone : public Primitive {
  public:
    /**
     * @brief Constructs a cone with the three vertices, radius,
     * start angle and end angle.
     * @param p1 First point of the cone, defines the base of the cone.
     * @param p2 Second point of the cone, defines the top of the cone.
     * Together with p1 defines the revolution axis of the cone and its heigth.
     * @param p3 Third point of the cone. Together with p1 and p2 defines the
     * starting plane of the cone from which start_angle and end_angle are
     * defined.
     * @param radius1 Radius of the cone at p1.
     * @param radius2 Radius of the cone at p2.
     * @param start_angle Start angle of the cone from the starting plane.
     * @param end_angle End angle of the cone from the starting plane.
     */
    Cone(Point3D p1, Point3D p2, Point3D p3, double radius1, double radius2,
         double start_angle, double end_angle)
        : _p1(std::move(p1)),
          _p2(std::move(p2)),
          _p3(std::move(p3)),
          _radius1(radius1),
          _radius2(radius2),
          _start_angle(start_angle),
          _end_angle(end_angle) {}

    /**
     * @brief Returns the first point of the cone.
     * @return The first point of the cone.
     */
    [[nodiscard]] const Point3D& get_p1() const { return _p1; }

    /**
     * @brief Returns the second point of the cone.
     * @return The second point of the cone.
     */
    [[nodiscard]] const Point3D& get_p2() const { return _p2; }

    /**
     * @brief Returns the third point of the cone.
     * @return The third point of the cone.
     */
    [[nodiscard]] const Point3D& get_p3() const { return _p3; }

    /**
     * @brief Returns the radius1 of the cone.
     * @return The radius1 of the cone.
     */
    [[nodiscard]] double get_radius1() const { return _radius1; }

    /**
     * @brief Returns the radius2 of the cone.
     * @return The radius2 of the cone.
     */
    [[nodiscard]] double get_radius2() const { return _radius2; }

    /**
     * @brief Returns the start angle of the cone.
     * @return The start angle of the cone.
     */
    [[nodiscard]] double get_start_angle() const { return _start_angle; }

    /**
     * @brief Returns the end angle of the cone.
     * @return The end angle of the cone.
     */
    [[nodiscard]] double get_end_angle() const { return _end_angle; }
    /**
     * @brief Updates the first point of the cone.
     * @param p1 The new first point of the cone.
     */
    void set_p1(Point3D p1) {
        _p1 = std::move(p1);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the second point of the cone.
     * @param p2 The new second point of the cone.
     */
    void set_p2(Point3D p2) {
        _p2 = std::move(p2);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the third point of the cone.
     * @param p3 The new third point of the cone.
     */
    void set_p3(Point3D p3) {
        _p3 = std::move(p3);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the radius1 of the cone.
     * @param radius The new radius1 of the cone.
     */
    void set_radius1(double radius1) {
        _radius1 = radius1;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the radius2 of the cone.
     * @param radius The new radius2 of the cone.
     */
    void set_radius2(double radius2) {
        _radius2 = radius2;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the start angle of the cone.
     * @param start_angle The new start angle of the cone.
     */
    void set_start_angle(double start_angle) {
        _start_angle = start_angle;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the end angle of the cone.
     * @param end_angle The new end angle of the cone.
     */
    void set_end_angle(double end_angle) {
        _end_angle = end_angle;
        callback_with_id(get_id());
    }

    [[nodiscard]] bool is_valid() const override {
        throw std::logic_error("Not implemented");
    };

    [[nodiscard]] double distance(const Point3D& /*point*/) const override {
        throw std::logic_error("Not implemented");
    };

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutted_surface(
        const Point3D& /*point*/) const override {
        throw std::logic_error("Not implemented");
    };

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutting_surface(
        const Point3D& /*p3d*/) const override {
        throw std::logic_error("Not implemented");
    }

    [[nodiscard]] Point3D from_2d_to_3d(const Point2D& p2d) const override;

    [[nodiscard]] Point2D from_3d_to_2d(const Point3D& p3d) const override;

    [[nodiscard]] TriMesh create_mesh(const ThermalMesh& thermal_mesh,
                                      double tolerance) const override;

    [[nodiscard]] MeshIndex get_faceid_from_uv(
        const ThermalMesh& thermal_mesh,
        const Point2D& point_uv) const override;

    [[nodiscard]] std::shared_ptr<Primitive> transform(
        const CoordinateTransformation& transformation) const override {
        return std::make_shared<Cone>(transformation.transform_point(_p1),
                                      transformation.transform_point(_p2),
                                      transformation.transform_point(_p3),
                                      _radius1, _radius2, _start_angle,
                                      _end_angle);
    };

  private:
    Point3D _p1;          ///< First point of the cone.
    Point3D _p2;          ///< Second point of the cone.
    Point3D _p3;          ///< Third point of the cone.
    double _radius1;      ///< Radius1 of the cone.
    double _radius2;      ///< Radius2 of the cone.
    double _start_angle;  ///< Start angle of the cone.
    double _end_angle;    ///< End angle of the cone.
};

/**
 * @class Sphere
 * @brief A class representing a sphere in 3D space.
 */
class Sphere : public Primitive {
  public:
    /**
     * @brief Constructs a sphere with given properties.
     * @param p1 First point of the sphere.
     * @param p2 Second point of the sphere.
     * @param p3 Third point of the sphere.
     * @param radius Radius of the sphere.
     * @param base_truncation Base truncation of the sphere.
     * @param apex_truncation Apex truncation of the sphere.
     * @param start_angle Start angle of the sphere.
     * @param end_angle End angle of the sphere.
     */
    Sphere(Point3D p1, Point3D p2, Point3D p3, double radius,
           double base_truncation, double apex_truncation, double start_angle,
           double end_angle)
        : _p1(std::move(p1)),
          _p2(std::move(p2)),
          _p3(std::move(p3)),
          _radius(radius),
          _base_truncation(base_truncation),
          _apex_truncation(apex_truncation),
          _start_angle(start_angle),
          _end_angle(end_angle) {}

    /**
     * @brief Returns the first point of the sphere.
     * @return The first point of the sphere.
     */
    [[nodiscard]] const Point3D& get_p1() const { return _p1; }

    /**
     * @brief Returns the second point of the sphere.
     * @return The second point of the sphere.
     */
    [[nodiscard]] const Point3D& get_p2() const { return _p2; }

    /**
     * @brief Returns the third point of the sphere.
     * @return The third point of the sphere.
     */
    [[nodiscard]] const Point3D& get_p3() const { return _p3; }

    /**
     * @brief Returns the radius of the sphere.
     * @return The radius of the sphere.
     */
    [[nodiscard]] double get_radius() const { return _radius; }

    /**
     * @brief Returns the base truncation of the sphere.
     * @return The base truncation of the sphere.
     */
    [[nodiscard]] double get_base_truncation() const {
        return _base_truncation;
    }

    /**
     * @brief Returns the apex truncation of the sphere.
     * @return The apex truncation of the sphere.
     */
    [[nodiscard]] double get_apex_truncation() const {
        return _apex_truncation;
    }

    /**
     * @brief Returns the start angle of the sphere.
     * @return The start angle of the sphere.
     */
    [[nodiscard]] double get_start_angle() const { return _start_angle; }

    /**
     * @brief Returns the end angle of the sphere.
     * @return The end angle of the sphere.
     */
    [[nodiscard]] double get_end_angle() const { return _end_angle; }

    /**
     * @brief Updates the first point of the sphere.
     * @param p1 The new first point of the sphere.
     */
    void set_p1(Point3D p1) {
        _p1 = std::move(p1);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the second point of the sphere.
     * @param p2 The new second point of the sphere.
     */
    void set_p2(Point3D p2) {
        _p2 = std::move(p2);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the third point of the sphere.
     * @param p3 The new third point of the sphere.
     */
    void set_p3(Point3D p3) {
        _p3 = std::move(p3);
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the radius of the sphere.
     * @param radius The new radius of the sphere.
     */
    void set_radius(double radius) {
        _radius = radius;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the base truncation of the sphere.
     * @param base_truncation The new base truncation of the sphere.
     */
    void set_base_truncation(double base_truncation) {
        _base_truncation = base_truncation;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the apex truncation of the sphere.
     * @param apex_truncation The new apex truncation of the sphere.
     */
    void set_apex_truncation(double apex_truncation) {
        _apex_truncation = apex_truncation;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the start angle of the sphere.
     * @param start_angle The new start angle of the sphere.
     */
    void set_start_angle(double start_angle) {
        _start_angle = start_angle;
        callback_with_id(get_id());
    }

    /**
     * @brief Updates the end angle of the sphere.
     * @param end_angle The new end angle of the sphere.
     */
    void set_end_angle(double end_angle) {
        _end_angle = end_angle;
        callback_with_id(get_id());
    }

    /**
     * @brief Checks if the sphere is valid.
     * @return `true` if the sphere is valid, `false` otherwise.
     */
    [[nodiscard]] bool is_valid() const override {
        throw std::logic_error("Not implemented");
    };

    [[nodiscard]] double distance(const Point3D& /*point*/) const override {
        throw std::logic_error("Not implemented");
    };

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutted_surface(
        const Point3D& /*point*/) const override {
        throw std::logic_error("Not implemented");
    };

    [[nodiscard]] std::array<double, 4> distance_jacobian_cutting_surface(
        const Point3D& /*p3d*/) const override {
        throw std::logic_error("Not implemented");
    }

    [[nodiscard]] Point3D from_2d_to_3d(const Point2D& p2d) const override {
        return from_2d_to_3d_sinusoidal(p2d, 0);
    }

    [[nodiscard]] Point2D from_3d_to_2d(const Point3D& p3d) const override {
        return from_3d_to_2d_sinusoidal(from_cartesian_to_spherical(p3d), 0);
    }

    [[nodiscard]] Point3D from_2d_to_3d_mollweide(const Point2D& p2d) const;

    [[nodiscard]] Point2D from_3d_to_2d_mollweide(const Point3D& p3d) const;

    [[nodiscard]] Point3D from_2d_to_3d_albers(const Point2D& p2d, double lat1,
                                               double lat2) const;

    [[nodiscard]] Point2D from_3d_to_2d_albers(const Point3D& p3d, double lat1,
                                               double lat2) const;

    [[nodiscard]] Point3D from_2d_to_3d_sinusoidal(const Point2D& p2d,
                                                   double lon0) const;

    [[nodiscard]] Point2D from_3d_to_2d_sinusoidal(
        const Eigen::Vector2d& spherical_coordinates, double lon0) const;

    [[nodiscard]] TriMesh create_mesh(const ThermalMesh& thermal_mesh,
                                      double tolerance) const override {
        return create_mesh2(thermal_mesh, tolerance);
    };

    [[nodiscard]] TriMesh create_mesh1(const ThermalMesh& thermal_mesh,
                                       double tolerance) const;

    [[nodiscard]] TriMesh create_mesh2(const ThermalMesh& thermal_mesh,
                                       double tolerance) const;

    [[nodiscard]] MeshIndex get_faceid_from_uv(
        const ThermalMesh& thermal_mesh,
        const Point2D& point_uv) const override;

    // Function to change from cartesian coordinates to longitude and latitude
    [[nodiscard]] Eigen::Vector2d from_cartesian_to_spherical(
        const Point3D& p3d) const {
        using std::numbers::pi;

        Eigen::Vector3d p = p3d - _p1;
        Eigen::Vector2d spherical_coordinates;

        if (p[2] == _radius) {
            spherical_coordinates[0] = _start_angle - pi;
            spherical_coordinates[1] = pi / 2;
            return spherical_coordinates;
        } else if (p[2] == -_radius) {
            spherical_coordinates[0] = _start_angle - pi;
            spherical_coordinates[1] = -pi / 2;
            return spherical_coordinates;
        }

        // Longitude
        double lon = -pi + std::atan2(p[1], p[0]);
        while (lon < -pi || lon >= pi) {
            if (lon < -pi) {
                lon += 2 * pi;
            } else if (lon >= pi) {
                lon -= 2 * pi;
            }
        }

        spherical_coordinates[0] = lon;
        // Latitude
        spherical_coordinates[1] = std::asin(p[2] / p.norm());
        return spherical_coordinates;
    }

    // Function to change from longitude and latitude to cartesian coordinates
    [[nodiscard]] Point3D from_spherical_to_cartesian(
        const Eigen::Vector2d& spherical_coordinates) const {
        Point3D p;
        p[0] = _p1[0] + _radius * std::cos(spherical_coordinates[1]) *
                            std::cos(spherical_coordinates[0]);
        p[1] = _p1[1] + _radius * std::sin(spherical_coordinates[1]) *
                            std::cos(spherical_coordinates[0]);
        p[2] = _p1[2] + _radius * std::sin(spherical_coordinates[1]);
        return p;
    }

    [[nodiscard]] std::shared_ptr<Primitive> transform(
        const CoordinateTransformation& transformation) const override {
        return std::make_shared<Sphere>(
            transformation.transform_point(_p1),
            transformation.transform_point(_p2),
            transformation.transform_point(_p3), _radius, _base_truncation,
            _apex_truncation, _start_angle, _end_angle);
    };

  private:
    Point3D _p1;              ///< First point of the sphere.
    Point3D _p2;              ///< Second point of the sphere.
    Point3D _p3;              ///< Third point of the sphere.
    double _radius;           ///< Radius of the sphere.
    double _base_truncation;  ///< Base truncation of the sphere.
    double _apex_truncation;  ///< Apex truncation of the sphere.
    double _start_angle;      ///< Start angle of the sphere.
    double _end_angle;        ///< End angle of the sphere.
};

// Distance methods

inline double Triangle::distance(const Point3D& point) const {
    const Eigen::Vector3d v0 = _p1 - point;
    const Eigen::Vector3d v1 = _p2 - _p1;
    const Eigen::Vector3d v2 = _p3 - _p1;
    const Eigen::Vector3d n = v1.cross(v2);  // Normal to the triangle's plane.

    const double d = v0.dot(n);  // Signed distance from the point to the plane.
    const double dist_to_plane = std::abs(d / n.norm());

    // Compute the barycentric coordinates of the foot of the
    // perpendicular.
    const double d1 = -v0.dot(v1);
    const double d2 = -v0.dot(v2);
    const double d3 = v1.dot(v1);
    const double d4 = v1.dot(v2);
    const double d5 = v2.dot(v2);
    const double denom = d3 * d5 - d4 * d4;

    const double b1 = (d1 * d5 - d2 * d4) / denom;
    const double b2 = (d3 * d2 - d1 * d4) / denom;
    const double b3 = 1.0 - b1 - b2;

    // If the foot of the perpendicular is inside the triangle, this is
    // the shortest distance.
    if (b1 >= 0 && b1 <= 1 && b2 >= 0 && b2 <= 1 && b3 >= 0 && b3 <= 1) {
        return dist_to_plane;
    }

    // Otherwise, the shortest distance is to an edge or vertex of the
    // triangle.
    const double dist_to_edges = std::min(
        {(point - _p1).norm(), (point - _p2).norm(), (point - _p3).norm(),
         dist_point_to_line_segment(point, _p1, _p2),
         dist_point_to_line_segment(point, _p2, _p3),
         dist_point_to_line_segment(point, _p3, _p1)});

    return dist_to_edges;
}

inline double Rectangle::distance(const Point3D& point) const {
    // Calculate rectangle's normal.
    const Vector3D normal = v1().cross(v2()).normalized();

    // Project the point onto the plane of the rectangle.
    const double dist_to_plane = (point - _p1).dot(normal);
    const Point3D projected_point = point - dist_to_plane * normal;

    // Check if the projected point is inside the rectangle.
    if ((projected_point - _p1).dot(v1()) >= 0 &&
        (projected_point - _p2).dot(v1()) <= 0 &&
        (projected_point - _p1).dot(v2()) >= 0 &&
        (projected_point - _p3).dot(v2()) <= 0) {
        // The projected point is inside the rectangle.
        return std::abs(dist_to_plane);
    }

    // The projected point is outside the rectangle.
    // Calculate the distance to each edge and take the minimum.
    constexpr VectorIndex num_vertices = 4;

    std::array<Vector3D, num_vertices> edges = {v1(), v2(), v2(), v1()};
    std::array<Point3D, num_vertices> vertices = {_p1, _p1, _p2, _p3};
    double min_dist = std::numeric_limits<double>::max();

    for (VectorIndex i = 0; i < num_vertices; ++i) {
        // https://github.com/llvm/llvm-project/issues/53997
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
        const Vector3D& edge = edges[i];
        const Point3D& vertex = vertices[i];
        // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)

        double t = (projected_point - vertex).dot(edge) / edge.dot(edge);
        t = std::max(0.0, std::min(1.0, t));

        const Point3D closest_point_on_edge = vertex + t * edge;
        const double dist_to_edge =
            (projected_point - closest_point_on_edge).norm();

        min_dist = std::min(min_dist, dist_to_edge);
    }

    return std::sqrt(min_dist * min_dist + dist_to_plane * dist_to_plane);
}

inline double Quadrilateral::distance(const Point3D& p3d) const {
    // We compute the distance of the point to the two triangles the
    // quadrilateral is composed of and return the minimum of the two.

    // Triangles
    const Triangle t1(_p1, _p2, _p3);
    const Triangle t2(_p1, _p3, _p4);

    // Distances
    const double d1 = t1.distance(p3d);
    const double d2 = t2.distance(p3d);

    return std::min(d1, d2);
}

inline double Cylinder::distance(const Point3D& point) const {
    Point3D direction = _p2 - _p1;
    const double length = direction.norm();
    direction.normalize();

    double t = (point - _p1).dot(direction);
    const double dist_to_surf =
        (-(direction * t) + (point - _p1)).norm() - _radius;

    if (t < 0.0) {
        // nothing

    } else if (t > length) {
        t -= length;
    } else {
        t = 0.0;
    }

    return sqrt(dist_to_surf * dist_to_surf + t * t);
}

inline std::array<double, 4> Triangle::distance_jacobian_cutted_surface(
    const Point3D& point) const {
    // Compute the vector from the point to the first vertex of the
    // triangle.
    const Vector3D v = point - _p1;

    // Compute the normal to the plane of the triangle.
    Vector3D n = this->v1().cross(this->v2()).normalized();

    const double d = v.dot(n);  // Signed distance

    if (d < 0) {
        return {-d, -n[0], -n[1], -n[2]};
    } else {
        return {d, n[0], n[1], n[2]};
    }
}

inline std::array<double, 4> Rectangle::distance_jacobian_cutted_surface(
    const Point3D& point) const {
    // Compute the vector from the point to the first vertex of the
    // rect.
    const Vector3D v = point - _p1;

    // Compute the normal to the plane of the rect.
    Vector3D n = this->v1().cross(this->v2()).normalized();

    const double d = v.dot(n);  // Signed distance

    if (d < 0) {
        return {-d, -n[0], -n[1], -n[2]};
    } else {
        return {d, n[0], n[1], n[2]};
    }
}

inline std::array<double, 4> Quadrilateral::distance_jacobian_cutted_surface(
    const Point3D& point) const {
    // Compute the vector from the point to the first vertex of the
    // quad.
    const Vector3D v = point - _p1;

    // Compute the normal to the plane of the quad.
    Vector3D n = this->v1().cross(this->v2()).normalized();

    const double d = v.dot(n);  // Signed distance

    if (d < 0) {
        return {-d, -n[0], -n[1], -n[2]};
    } else {
        return {d, n[0], n[1], n[2]};
    }
}

inline std::array<double, 4> Cylinder::distance_jacobian_cutted_surface(
    const Point3D& point) const {
    // Compute the axis vector
    const Vector3D axis_dir = (_p2 - _p1).normalized();

    // Compute the vector from the point to the first point of the
    // cylinder.
    const Vector3D v = point - _p1;

    // Compute the vector perpendicular to the axis to the point
    Vector3D v_n_axis = (v.dot(axis_dir) * axis_dir) - v;

    // Compute the distance
    const double d = v_n_axis.norm() - _radius;

    if (d < 0) {
        return {-d, v_n_axis[0], v_n_axis[1], v_n_axis[2]};
    } else {
        return {d, -v_n_axis[0], -v_n_axis[1], -v_n_axis[2]};
    }
}

// Validate methods

inline bool Triangle::is_valid() const {
    const Vector3D v1 = _p2 - _p1;
    const Vector3D v2 = _p3 - _p2;
    const Vector3D v3 = _p1 - _p3;

    // Check that edges have length greater than LENGTH_TOL.
    // Check that the points are not colinear.
    // Return false otherwise.
    return are_vectors_nonzero_length({&v1, &v2, &v3}) &&
           !are_vectors_parallel(v1, v2);
}

inline bool Rectangle::is_valid() const {
    // Compute and store the vectors v1 and v2
    const Vector3D v1(this->v1());
    const Vector3D v2(this->v2());

    // Check if the vectors v1 and v2 are orthogonal.
    // Check if the vectors v1 and v2 are non-zero.
    // Return false otherwise
    return are_vectors_orthogonal(v1, v2) &&
           are_vectors_nonzero_length({&v1, &v2});
}

inline bool Quadrilateral::is_valid() const {
    // Compute and store the vectors v1 and v2
    Vector3D v1(this->v1());
    Vector3D v2(this->v2());

    // Check if v1 or v2 is too small
    if (!are_vectors_nonzero_length({&v1, &v2})) {
        return false;
    }

    // Check if the angle between v1 and v2 is too close to 0 or 180
    // degrees
    if (are_vectors_parallel(v1, v2)) {
        return false;
    }

    // Compute the normal vector n via cross product
    Vector3D n = v1.cross(v2);
    const double nlen = n.norm();

    // Check if the length of n is too small
    if (nlen <= LENGTH_TOL) {
        return false;
    }

    // Normalize n
    n = n / nlen;

    // Check if _p3 is in the plane spanned by _p1, _p2, and _p4
    const Vector3D m(_p3 - _p1);
    if (std::abs(m.dot(n)) > LENGTH_TOL) {
        return false;
    }

    // Verify correct interior angles at points _p2, _p3, and _p4
    const std::array<std::tuple<Point3D, Point3D, Point3D>, 3> points = {
        std::make_tuple(_p2, _p3, _p1),  // Corresponds to case 2
        std::make_tuple(_p3, _p4, _p2),  // Corresponds to case 3
        std::make_tuple(_p4, _p1, _p3)   // Corresponds to case 4
    };

    // cppcheck-suppress unassignedVariable
    for (const auto& [curr, prev, next] : points) {
        v1 = prev - curr;
        v2 = next - curr;

        if (v1.norm() <= LENGTH_TOL || v2.norm() <= LENGTH_TOL) {
            return false;
        }

        const double dot = v1.dot(v2) / (v1.norm() * v2.norm());

        if (dot <= (-1.0 + ANGLE_TOL) || dot >= (1.0 - ANGLE_TOL)) {
            return false;
        }
    }

    return true;
}

// TODO: TEST THIS Function
inline bool Disc::is_valid() const {
    using std::numbers::pi;

    const Eigen::Vector3d v21 = _p2 - _p1;
    const Eigen::Vector3d v31 = _p3 - _p1;
    const double start_angle_degrees = _start_angle * 180.0 / pi;
    const double end_angle_degrees = _end_angle * 180.0 / pi;

    // Pre-calculate all conditions, optimizing for the case where is_valid()
    // returns true
    std::array<bool, 7> conditions = {
        // Check that directions have length greater than LENGTH_TOL.
        are_vectors_nonzero_length({&v21, &v31}),

        // Check vectors are orthogonal
        are_vectors_orthogonal(v21, v31),

        // Check inner radius is positive and greater than LENGTH_TOL
        _inner_radius >= LENGTH_TOL,

        // Check outer radius is positive and greater than LENGTH_TOL
        _outer_radius >= LENGTH_TOL,

        // Check if start and end angles are valid
        start_angle_degrees >= -360 && start_angle_degrees <= 360 &&
            end_angle_degrees >= -360 && end_angle_degrees <= 360,

        // Check angle difference is above tolerance
        (end_angle_degrees - start_angle_degrees) >= ANGLE_TOL,

        // Check angle difference does not exceed limit
        (end_angle_degrees - start_angle_degrees) < 360 + ANGLE_TOL};

    // Since we are optimizing for the case is_valid() is true, we can aggregate
    // all conditions and return the result, avoiding early exit for efficiency
    // in the true case.
    const bool is_valid =
        std::all_of(std::begin(conditions), std::end(conditions),
                    [](bool condition) { return condition; });

    return is_valid;
}

inline bool Cylinder::is_valid() const {
    using std::numbers::pi;

    const Eigen::Vector3d v21 = _p2 - _p1;
    const Eigen::Vector3d v31 = _p3 - _p1;
    const double start_angle_degrees = _start_angle * 180.0 / pi;
    const double end_angle_degrees = _end_angle * 180.0 / pi;

    // Pre-calculate all conditions, optimizing for the case where is_valid()
    // returns true
    std::array<bool, 6> conditions = {
        // Check that directions (v21 and v31) have length greater than
        // LENGTH_TOL.
        are_vectors_nonzero_length({&v21, &v31}),

        // Check the axis (p2p1) is orthogonal to p3p1
        are_vectors_orthogonal(v21, v31),

        // Check radius is positive and greater than LENGTH_TOL
        _radius >= LENGTH_TOL,

        // Check if start and end angles are valid
        start_angle_degrees >= -360 && start_angle_degrees <= 360 &&
            end_angle_degrees >= -360 && end_angle_degrees <= 360,

        // Check angle difference is above tolerance
        (end_angle_degrees - start_angle_degrees) >= ANGLE_TOL,

        // Check angle difference does not exceed limit
        (end_angle_degrees - start_angle_degrees) < 360 + ANGLE_TOL};

    // Since we are optimizing for the case is_valid() is true, we can aggregate
    // all conditions and return the result, avoiding early exit for efficiency
    // in the true case.
    const bool is_valid =
        std::all_of(std::begin(conditions), std::end(conditions),
                    [](bool condition) { return condition; });

    return is_valid;
}

// 2D - 3D transformation methods

// Triangle
inline Point2D Triangle::from_3d_to_2d(const Point3D& p3d) const {
    // First direction in the 2D plane
    const Vector3D vx = v1().normalized();
    const Vector3D v3 = (_p3 - _p1);

    // Second direction in the 2D plane
    const Vector3D vy = (v3 - v3.dot(vx) * vx).normalized();

    return {(p3d - _p1).dot(vx), (p3d - _p1).dot(vy)};
}

inline Point3D Triangle::from_2d_to_3d(const Point2D& p2d) const {
    const Vector3D vx = v1().normalized();
    const Vector3D v3 = (_p3 - _p1);
    const Vector3D vy = (v3 - v3.dot(vx) * vx).normalized();
    return Point3D(_p1 + p2d.x() * vx + p2d.y() * vy);
}

// Rectangle
inline Point2D Rectangle::from_3d_to_2d(const Point3D& p3d) const {
    Point2D p2d((p3d - _p1).dot(v1().normalized()),
                (p3d - _p1).dot(v2().normalized()));
    return p2d;
}

inline Point3D Rectangle::from_2d_to_3d(const Point2D& p2d) const {
    Point3D p3d(_p1 + p2d.x() * v1().normalized() +
                p2d.y() * v2().normalized());
    return p3d;
}

// Quadrilateral
inline Point2D Quadrilateral::from_3d_to_2d(const Point3D& p3d) const {
    // First direction in the 2D plane
    const Vector3D vx = v1().normalized();

    // Second direction in the 2D plane
    const Vector3D vy = (v2() - v2().dot(vx) * vx).normalized();

    return {(p3d - _p1).dot(vx), (p3d - _p1).dot(vy)};
}

inline Point3D Quadrilateral::from_2d_to_3d(const Point2D& p2d) const {
    const Vector3D vx = v1().normalized();
    const Vector3D vy = (v2() - v2().dot(vx) * vx).normalized();
    return Point3D(_p1 + p2d.x() * vx + p2d.y() * vy);
}

// Disc
inline Point2D Disc::from_3d_to_2d(const Point3D& p3d) const {
    // First direction in the 2D plane
    const Vector3D vx = v1().normalized();
    const Vector3D vy = v2().normalized();

    return {(p3d - _p1).dot(vx), (p3d - _p1).dot(vy)};
}

inline Point3D Disc::from_2d_to_3d(const Point2D& p2d) const {
    const Vector3D vx = v1().normalized();
    const Vector3D vy = v2().normalized();
    return Point3D(_p1 + p2d.x() * vx + p2d.y() * vy);
}

// Cylinder
inline Point3D Cylinder::from_2d_to_3d(const Point2D& p2d) const {
    const Vector3D h_dir = (_p2 - _p1).normalized();
    const Vector3D r_dir = (_p3 - _p1).normalized();
    const double theta = p2d.x() / _radius;

    const Eigen::AngleAxisd rot(theta, h_dir);
    return _p1 + p2d.y() * h_dir + _radius * (rot * r_dir);
}

inline Point2D Cylinder::from_3d_to_2d(const Point3D& p3d) const {
    using std::numbers::pi;
    const Vector3D h_dir = (_p2 - _p1).normalized();
    const Vector3D r_dir = (_p3 - _p1).normalized();

    const double y = (p3d - _p1).dot(h_dir);

    const Vector3D p3d_r = (p3d - _p1) - y * h_dir;
    double theta = acos(p3d_r.normalized().dot(r_dir));

    if (r_dir.cross(p3d_r).dot(h_dir) < 0) {
        theta = 2 * pi - theta;
    }

    return {theta * _radius, y};
}

// Cone
inline Point2D Cone::from_3d_to_2d(const Point3D& p3d) const {
    if ((p3d - _p1).norm() < 1e-6) {
        return {0.0, 0.0};
    }

    using std::numbers::pi;

    // Main directions of the 3D cone
    const Eigen::Vector3d vx = (_p3 - _p1).normalized();
    const Eigen::Vector3d vz = (_p2 - _p1).normalized();
    const Eigen::Vector3d vy = (vz.cross(vx)).normalized();

    // Length of the side of the cone
    const double h = (_p2 - _p1).norm();
    const double s1 =
        sqrt(h * h + abs(_radius2 - _radius1) * abs(_radius2 - _radius1));
    const double s2 = (_radius1 < _radius2) ? _radius1 * s1 / _radius2
                                            : _radius2 * s1 / _radius1;
    const double s = s1 + s2;

    // Project the cone in the x-y plane
    Eigen::Vector2d p3d_xy((p3d - _p1).dot(vx), (p3d - _p1).dot(vy));
    double theta = atan2(p3d_xy[1], p3d_xy[0]);
    while (theta < 0.0) {
        theta += 2 * pi;
    }
    while (theta > 2 * pi) {
        theta -= 2 * pi;
    }

    // Project to the axis
    const double p3d_h = (p3d - _p1).dot(vz);

    // Length of the side of the cone at the height of the point
    const double p3d_s =
        (_radius1 < _radius2) ? s1 * p3d_h / h + s2 : s1 * (h - p3d_h) / h + s2;

    // Calculate the theta angle in the 2D projection
    const double theta_2d =
        (_radius1 < _radius2) ? theta * _radius2 / s : theta * _radius1 / s;

    // Change to 2D coordinates
    const double x = p3d_s * cos(theta_2d);
    const double y = p3d_s * sin(theta_2d);

    return {x, y};
}

inline Point3D Cone::from_2d_to_3d(const Point2D& p2d) const {
    using std::numbers::pi;

    const double x = p2d[0];
    const double y = p2d[1];

    // Main directions of the 3D cone
    const Eigen::Vector3d vx = (_p3 - _p1).normalized();
    const Eigen::Vector3d vz = (_p2 - _p1).normalized();
    const Eigen::Vector3d vy = (vz.cross(vx)).normalized();

    const double s_2d = sqrt(x * x + y * y);

    if (s_2d < 1e-6) {
        return (_radius1 == 0.0) ? _p1 : _p2;
    }

    // Length of the side of the cone
    const double h = (_p2 - _p1).norm();
    const double s1 =
        sqrt(h * h + abs(_radius2 - _radius1) * abs(_radius2 - _radius1));
    const double s2 = (_radius1 < _radius2) ? _radius1 * s1 / _radius2
                                            : _radius2 * s1 / _radius1;
    const double s = s1 + s2;

    // Calculate the angle theta
    double theta_2d = atan2(y, x);
    while (theta_2d < 0.0) {
        theta_2d += 2 * pi;
    }
    const double theta_3d = (_radius1 < _radius2) ? theta_2d * s / _radius2
                                                  : theta_2d * s / _radius1;

    // Calculate the height and radius of the 3D point
    const double p3d_h =
        (_radius1 < _radius2) ? h * (s_2d - s2) / s1 : h - (s_2d - s2) / s1 * h;
    const double p3d_r =
        (_radius1 < _radius2)
            ? _radius1 + (s_2d - s2) * (_radius2 - _radius1) / s1
            : _radius2 + (s_2d - s2) * (_radius1 - _radius2) / s1;

    Eigen::Vector3d p3d = _p1 + p3d_h * vz + p3d_r * cos(theta_3d) * vx +
                          p3d_r * sin(theta_3d) * vy;

    return p3d;
}

// Sphere
inline Point2D Sphere::from_3d_to_2d_mollweide(const Point3D& p3d) const {
    if ((p3d - _p1).norm() < 1e-6) {
        return {0.0, 0.0};
    }

    using std::numbers::pi;

    Eigen::Vector2d spherical_coordinates = from_cartesian_to_spherical(p3d);

    double theta = spherical_coordinates[1];
    double theta_pre = theta;
    for (int i = 0; i < 10; ++i) {
        theta = theta_pre - (2 * theta_pre + sin(2 * theta_pre) -
                             pi * sin(spherical_coordinates[1])) /
                                (4 * (cos(theta_pre) * cos(theta_pre)));
        if (abs(theta - theta_pre) / theta_pre <= 1e-9) {
            break;
        }
        theta_pre = theta;
    }

    const double x =
        _radius * 2 * sqrt(2) / pi * spherical_coordinates[0] * cos(theta);
    const double y = _radius * sqrt(2) * sin(theta);

    return {x, y};
}

/// @brief Inverse Mollweide projection
/// @param p2d Point in the Mollweide projection
/// @return Point in the 3D sphere
inline Point3D Sphere::from_2d_to_3d_mollweide(const Point2D& p2d) const {
    using std::numbers::pi;

    const double theta = asin(p2d[1] / (_radius * sqrt(2)));
    const double lon = pi * p2d[0] / (2 * _radius * sqrt(2) * cos(theta));
    const double lat = asin((2 * theta + sin(2 * theta)) / pi);

    return from_spherical_to_cartesian(Eigen::Vector2d(lon, lat));
}

/// @brief Apply the Albers projection to a 3D point
/// @param p3d 3D coordinates of the point
/// @return 2D coordinates of the point in the Albers projection
inline Point2D Sphere::from_3d_to_2d_albers(const Point3D& p3d, double lat1,
                                            double lat2) const {
    if ((p3d - _p1).norm() < 1e-6) {
        return {0.0, 0.0};
    }

    using std::numbers::pi;

    Eigen::Vector2d spherical_coordinates = from_cartesian_to_spherical(p3d);

    double lon = spherical_coordinates[0];
    double lat = spherical_coordinates[1];

    if (lon >= 2 * pi || lon < 0) {
        while (lon >= 2 * pi) {
            lon -= 2 * pi;
        }
        while (lon < 0) {
            lon += 2 * pi;
        }
    }

    if (lat >= pi / 2 || lat < -pi / 2) {
        while (lat >= pi / 2) {
            lat -= pi;
        }
        while (lat < -pi / 2) {
            lat += pi;
        }
    }

    const double n = (sin(lat1) + sin(lat2)) / 2;
    const double c = cos(lat1) * cos(lat1) + 2 * n * sin(lat1);
    const double rho = _radius * sqrt(c - 2 * n * sin(lat)) / n;
    const double theta = n * lon;

    const double y = -rho * cos(theta + pi / 2);
    const double x = rho * sin(theta + pi / 2);

    return {x, y};
}

inline Point3D Sphere::from_2d_to_3d_albers(const Point2D& p2d, double lat1,
                                            double lat2) const {
    using std::numbers::pi;

    const double x = p2d[0];
    const double y = p2d[1];

    const double n = (sin(lat1) + sin(lat2)) / 2;
    const double c = cos(lat1) * cos(lat1) + 2 * n * sin(lat1);
    const double rho = sqrt(x * x + y * y);
    const double theta = atan2(x, -y) - pi / 2;

    double lat =
        asin(((c - (rho * n / _radius) * (rho * n / _radius)) / (2 * n)));
    double lon = theta / n;

    if (lon >= 2 * pi || lon < 0) {
        while (lon >= 2 * pi) {
            lon -= 2 * pi;
        }
        while (lon < 0) {
            lon += 2 * pi;
        }
    }

    if (lat >= pi / 2 || lat < -pi / 2) {
        while (lat >= pi / 2) {
            lat = pi - lat;
        }
        while (lat < -pi / 2) {
            lat = -pi - lat;
        }
    }

    return from_spherical_to_cartesian(Eigen::Vector2d(lon, lat));
}

inline Point2D Sphere::from_3d_to_2d_sinusoidal(
    const Eigen::Vector2d& spherical_coordinates, double lon0) const {
    using std::numbers::pi;

    const double lon = spherical_coordinates[0];
    const double lat = spherical_coordinates[1];

    const double x = _radius * (lon - lon0) * cos(lat);
    const double y = _radius * lat;

    return {x, y};
}

inline Point3D Sphere::from_2d_to_3d_sinusoidal(const Point2D& p2d,
                                                double lon0) const {
    using std::numbers::pi;

    const double x = p2d[0];
    const double y = p2d[1];

    double lat = y / _radius;
    double lon = lon0 + x / (_radius * cos(lat));

    if (lon >= pi || lon < -pi) {
        while (lon >= pi) {
            lon -= 2 * pi;
        }
        while (lon < -pi) {
            lon += 2 * pi;
        }
    }

    if (lat > pi / 2 || lat < -pi / 2) {
        while (lat > pi / 2) {
            lat = pi - lat;
        }
        while (lat < -pi / 2) {
            lat = -pi - lat;
        }
    }

    return from_spherical_to_cartesian(Eigen::Vector2d(lon, lat));
}

// Face ID calculation methods
inline MeshIndex Triangle::get_faceid_from_uv(const ThermalMesh& thermal_mesh,
                                              const Point2D& point_uv) const {
    const auto* dir1_mesh_ptr = thermal_mesh.get_dir1_mesh_ptr();
    const auto* dir2_mesh_ptr = thermal_mesh.get_dir2_mesh_ptr();

    const auto& dir1_mesh = *dir1_mesh_ptr;
    const auto& dir2_mesh = *dir2_mesh_ptr;

    std::vector<double> dir1_mesh_normalized(dir1_mesh.size());
    std::vector<double> dir2_mesh_normalized(dir2_mesh.size());

    for (VectorIndex i = 0; i < dir1_mesh.size(); ++i) {
        dir1_mesh_normalized[i] =
            dir1_mesh[i] / dir1_mesh[dir1_mesh.size() - 1];
    }
    for (VectorIndex i = 0; i < dir2_mesh.size(); ++i) {
        dir2_mesh_normalized[i] =
            dir2_mesh[i] / dir2_mesh[dir2_mesh.size() - 1];
    }

    const double length_dir1 = v1().norm();
    const double length_dir2 = v2().norm();

    const Point2D p1_2d = from_3d_to_2d(_p1);
    const Point2D p2_2d = from_3d_to_2d(_p2);
    const Point2D p3_2d = from_3d_to_2d(_p3);

    Vector2D OP = point_uv - p1_2d;
    Vector2D v32 = (p3_2d - p2_2d).normalized();

    // Intersection in P2 - P1 for u
    Vector2D v21 = (p2_2d - p1_2d).normalized();
    const double term1_u = OP.x() * v21.y() - OP.y() * v21.x();
    const double term2_u = p1_2d.x() * v21.y() - p1_2d.y() * v21.x();
    const double term3_u = v32.x() * v21.y() - v32.y() * v21.x();
    const double t_u = (term2_u - term1_u) / term3_u;

    Point2D intersection_21 = OP + t_u * v32;
    Vector2D i = {intersection_21.x() - p1_2d.x(),
                  intersection_21.y() - p1_2d.y()};
    const double u = i.norm();
    i = i.normalized();

    assert(std::abs(i.x() - v21.x()) < 1e-6);
    assert(std::abs(i.y() - v21.y()) < 1e-6);

    // Intersection in P3 - P2 for v
    const double term1_v = OP.x() * v32.y() - OP.y() * v32.x();
    const double term2_v = p2_2d.x() * v32.y() - p2_2d.y() * v32.x();
    const double t_v = (term2_v - term1_v) / term1_v;

    Point2D intersection_32 = OP + t_v * OP;
    Vector2D j = {intersection_32.x() - p2_2d.x(),
                  intersection_32.y() - p2_2d.y()};
    const double v = j.norm();
    j = j.normalized();

    assert(std::abs(j.x() - v32.x()) < 1e-6);
    assert(std::abs(j.y() - v32.y()) < 1e-6);

    auto x_it = std::lower_bound(dir1_mesh_normalized.begin(),
                                 dir1_mesh_normalized.end(), u / length_dir1);
    auto y_it = std::lower_bound(dir2_mesh_normalized.begin(),
                                 dir2_mesh_normalized.end(), v / length_dir2);

    // TODO: Check if this is correct
    // If x_it or y_it are at the beginning or end of the vector, , throw
    // exception if (x_it == dir1_mesh.begin() || x_it == dir1_mesh.end() ||
    //     y_it == dir2_mesh.begin() || y_it == dir2_mesh.end()) {
    //     throw std::out_of_range("UV point is outside the triangle.");
    // }

    // Calculate indices
    const auto x_index = static_cast<MeshIndex>(
        std::distance(dir1_mesh_normalized.begin(), x_it) - 1);
    const auto y_index = static_cast<MeshIndex>(
        std::distance(dir2_mesh_normalized.begin(), y_it) - 1);

    // Compute face_id
    return (y_index * (static_cast<MeshIndex>(dir1_mesh.size()) - 1) +
            x_index) *
           2;
}

inline MeshIndex Rectangle::get_faceid_from_uv(const ThermalMesh& thermal_mesh,
                                               const Point2D& point_uv) const {
    const auto* dir1_mesh_ptr = thermal_mesh.get_dir1_mesh_ptr();
    const auto* dir2_mesh_ptr = thermal_mesh.get_dir2_mesh_ptr();

    const auto& dir1_mesh = *dir1_mesh_ptr;
    const auto& dir2_mesh = *dir2_mesh_ptr;

    const double length_dir1 = v1().norm();
    const double length_dir2 = v2().norm();

    auto x_it = std::lower_bound(dir1_mesh.begin(), dir1_mesh.end(),
                                 point_uv.x() / length_dir1);
    auto y_it = std::lower_bound(dir2_mesh.begin(), dir2_mesh.end(),
                                 point_uv.y() / length_dir2);

    // If x_it or y_it are at the beginning or end of the vector, throw
    // exception
    if (x_it == dir1_mesh.begin() || x_it == dir1_mesh.end() ||
        y_it == dir2_mesh.begin() || y_it == dir2_mesh.end()) {
        throw std::out_of_range("UV point is outside the rectangle.");
    }

    // Calculate indices
    const auto x_index =
        static_cast<MeshIndex>(std::distance(dir1_mesh.begin(), x_it) - 1);
    const auto y_index =
        static_cast<MeshIndex>(std::distance(dir2_mesh.begin(), y_it) - 1);

    // Compute face_id
    return (y_index * (static_cast<MeshIndex>(dir1_mesh.size()) - 1) +
            x_index) *
           2;
}

inline MeshIndex Disc::get_faceid_from_uv(const ThermalMesh& thermal_mesh,
                                          const Point2D& point_uv) const {
    using std::numbers::pi;

    auto dir1_mesh = thermal_mesh.get_dir1_mesh();
    auto dir2_mesh = thermal_mesh.get_dir2_mesh();

    auto inner_radius = _inner_radius;
    auto outer_radius = _outer_radius;

    if (inner_radius != 0.0) {
        for (VectorIndex i = 0; i < dir1_mesh.size() - 1; i++) {
            dir1_mesh[i] =
                dir1_mesh[i] * (outer_radius - inner_radius) / outer_radius +
                inner_radius / outer_radius;
        }
    }

    // std::cout << "Point " << point_uv.x() << ", " << point_uv.y() <<
    // '\n';
    auto r_uv = point_uv.norm();
    if (r_uv < _inner_radius) {
        r_uv = _inner_radius + (dir1_mesh[1] - dir1_mesh[0]) * outer_radius / 2;
    }
    auto angle_uv = std::atan2(point_uv.y(), point_uv.x());
    // std::cout << "atan2 " << angle_uv << '\n';
    if (angle_uv < 0) {
        angle_uv += 2 * pi;
    }

    // With auto I get a compiler error
    // NOLINTBEGIN(hicpp-use-auto,modernize-use-auto)
    const std::vector<double>::iterator r_it = std::lower_bound(
        dir1_mesh.begin(), dir1_mesh.end(), r_uv / outer_radius);

    // std::cout << "r_it: " << *r_it << " r_uv: " << r_uv << '\n';

    const std::vector<double>::iterator angle_it = std::lower_bound(
        dir2_mesh.begin(), dir2_mesh.end(), angle_uv / (2 * pi));

    // NOLINTEND(hicpp-use-auto,modernize-use-auto)

    // If r_it or angle_it are at the beginning or end of the vector, return -1
    if (r_it == dir1_mesh.begin() || r_it == dir1_mesh.end() ||
        angle_it == dir2_mesh.begin() || angle_it == dir2_mesh.end()) {
        throw std::out_of_range("UV point is outside the disc.");
    }

    // Calculate indices
    const auto r_index =
        static_cast<MeshIndex>(std::distance(dir1_mesh.begin(), r_it) - 1);
    const auto angle_index =
        static_cast<MeshIndex>(std::distance(dir2_mesh.begin(), angle_it) - 1);

    // Compute face_id
    return (angle_index * (static_cast<MeshIndex>(dir1_mesh.size()) - 1) +
            r_index) *
           2;
}

// Face ID calculation methods
inline MeshIndex Cylinder::get_faceid_from_uv(const ThermalMesh& thermal_mesh,
                                              const Point2D& point_uv) const {
    const auto* dir1_mesh_ptr = thermal_mesh.get_dir1_mesh_ptr();
    const auto* dir2_mesh_ptr = thermal_mesh.get_dir2_mesh_ptr();

    const auto& dir1_mesh = *dir1_mesh_ptr;
    const auto& dir2_mesh = *dir2_mesh_ptr;

    const double length_dir1 =
        (get_end_angle() - get_start_angle()) * get_radius();
    const double length_dir2 = (get_p2() - get_p1()).norm();

    auto x_it = std::lower_bound(dir1_mesh.begin(), dir1_mesh.end(),
                                 point_uv.x() / length_dir1);
    auto y_it = std::lower_bound(dir2_mesh.begin(), dir2_mesh.end(),
                                 point_uv.y() / length_dir2);

    // If x_it or y_it are at the beginning or end of the vector, return -1
    if (x_it == dir1_mesh.begin() || x_it == dir1_mesh.end() ||
        y_it == dir2_mesh.begin() || y_it == dir2_mesh.end()) {
        throw std::out_of_range("UV point is outside the cylinder.");
    }

    // Calculate indices
    const auto x_index =
        static_cast<MeshIndex>(std::distance(dir1_mesh.begin(), x_it) - 1);
    const auto y_index =
        static_cast<MeshIndex>(std::distance(dir2_mesh.begin(), y_it) - 1);

    // Compute face_id
    return (y_index * (static_cast<MeshIndex>(dir1_mesh.size()) - 1) +
            x_index) *
           2;
}

inline MeshIndex Cone::get_faceid_from_uv(const ThermalMesh& thermal_mesh,
                                          const Point2D& point_uv) const {
    using std::numbers::pi;

    // Length of the side of the cone
    const double h = (_p2 - _p1).norm();
    const double s1 =
        sqrt(h * h + abs(_radius2 - _radius1) * abs(_radius2 - _radius1));
    const double s2 = (_radius1 < _radius2) ? _radius1 * s1 / _radius2
                                            : _radius2 * s1 / _radius1;
    const double s = s1 + s2;

    auto dir1_mesh = thermal_mesh.get_dir1_mesh();
    auto dir2_mesh = thermal_mesh.get_dir2_mesh();
    // Remap dir2_mesh to the 2D plane
    // for (int i = 0; i < dir2_mesh.size(); ++i) {
    //     dir2_mesh[i] = (_radius1 < _radius2) ? dir2_mesh[i] * _radius2 / s :
    //                                            dir2_mesh[i] * _radius1 / s;
    // }

    for (auto& value : dir2_mesh) {
        value = (_start_angle + value * (_end_angle - _start_angle)) / (2 * pi);
        value =
            (_radius1 < _radius2) ? value * _radius2 / s : value * _radius1 / s;
    }

    auto inner_radius = s2;
    auto outer_radius = s;
    if (inner_radius != 0.0) {
        for (VectorIndex i = 0; i < dir1_mesh.size() - 1; i++) {
            dir1_mesh[i] =
                dir1_mesh[i] * (outer_radius - inner_radius) / outer_radius +
                inner_radius / outer_radius;
        }
    }

    // std::cout << "Point " << point_uv.x() << ", " << point_uv.y() <<
    // '\n';
    auto r_uv = point_uv.norm();
    if (r_uv < inner_radius) {
        r_uv = inner_radius + (dir1_mesh[1] - dir1_mesh[0]) * outer_radius / 2;
    }
    auto angle_uv = std::atan2(point_uv.y(), point_uv.x());
    // std::cout << "atan2 " << angle_uv << '\n';
    if (angle_uv < 0) {
        angle_uv += 2 * pi;
    }

    // With auto I get a compiler error
    // NOLINTBEGIN(hicpp-use-auto,modernize-use-auto)
    const std::vector<double>::iterator r_it = std::lower_bound(
        dir1_mesh.begin(), dir1_mesh.end(), r_uv / outer_radius);

    // std::cout << "r_it: " << *r_it << " r_uv: " << r_uv << '\n';

    const std::vector<double>::iterator angle_it = std::lower_bound(
        dir2_mesh.begin(), dir2_mesh.end(), angle_uv / (2 * pi));
    // std::cout << "angle_it: " << *angle_it << " angle_uv: " << angle_uv <<
    // '\n';

    // NOLINTEND(hicpp-use-auto,modernize-use-auto)

    // If r_it or angle_it are at the beginning or end of the vector, return -1
    if (r_it == dir1_mesh.begin() || r_it == dir1_mesh.end() ||
        angle_it == dir2_mesh.begin() || angle_it == dir2_mesh.end()) {
        throw std::out_of_range("UV point is outside the cone.");
    }

    // Calculate indices
    const auto r_index =
        static_cast<MeshIndex>(std::distance(dir1_mesh.begin(), r_it) - 1);
    const auto angle_index =
        static_cast<MeshIndex>(std::distance(dir2_mesh.begin(), angle_it) - 1);

    // Compute face_id
    return (angle_index * (static_cast<MeshIndex>(dir1_mesh.size()) - 1) +
            r_index) *
           2;
}

inline MeshIndex Sphere::get_faceid_from_uv(const ThermalMesh& thermal_mesh,
                                            const Point2D& point_uv) const {
    using std::numbers::pi;
    std::ofstream archivo("centroids.txt", std::ios::app);  // WTF?

    // Latitude and longitude of the point
    const double lon = point_uv.x();
    const double lat = point_uv.y();

    archivo << "lon: " << lon << " lat: " << lat << '\n';

    auto dir1_mesh = thermal_mesh.get_dir1_mesh();
    auto dir2_mesh = thermal_mesh.get_dir2_mesh();

    const double lon_start = _start_angle - pi;
    const double lon_end = _end_angle - pi;

    const double lat_start = asin(_base_truncation / _radius);
    const double lat_end = asin(_apex_truncation / _radius);

    // Remap dir1_mesh to latitude instead of z
    std::transform(
        dir1_mesh.begin(), dir1_mesh.end(), dir1_mesh.begin(),
        [this](auto value) {
            return asin((_base_truncation +
                         value * (_apex_truncation - _base_truncation)) /
                        _radius);
        });
    // Same as:
    // for (auto& value : dir1_mesh) {
    //     value = asin(
    //         (_base_truncation + value * (_apex_truncation -
    //         _base_truncation)) / _radius);
    // }

    archivo << "lon_start: " << lon_start << " lon_end: " << lon_end << '\n';
    archivo << "lat_start: " << lat_start << " lat_end: " << lat_end << '\n';

    const double lon_uv = (lon - lon_start) / (lon_end - lon_start);
    const double lat_uv = lat;

    archivo << "lon_uv: " << lon_uv << " lat_uv: " << lat_uv << '\n';

    auto lat_it = std::lower_bound(dir1_mesh.begin(), dir1_mesh.end(), lat_uv);

    auto lon_it = std::lower_bound(dir2_mesh.begin(), dir2_mesh.end(), lon_uv);

    // If r_it or angle_it are at the beginning or end of the vector, return -1
    // if (lat_it == dir1_mesh.begin() || lat_it == dir1_mesh.end() ||
    //     lon_it == dir2_mesh.begin() || lon_it == dir2_mesh.end()) {
    //     return -1;
    // }

    // Calculate indices
    const auto lat_index =
        static_cast<MeshIndex>(std::distance(dir1_mesh.begin(), lat_it) - 1);
    const auto lon_index =
        static_cast<MeshIndex>(std::distance(dir2_mesh.begin(), lon_it) - 1);

    archivo << "lat_index: " << static_cast<int>(lat_index)
            << " lon_index: " << static_cast<int>(lon_index) << '\n';

    archivo.close();

    // Compute face_id
    // std::cout << static_cast<int>((lon_index * (dir1_mesh.size() - 1) +
    // lat_index) * 2) << '\n';
    return (lon_index * (static_cast<MeshIndex>(dir1_mesh.size()) - 1) +
            lat_index) *
           2;
}

// Triangular mesh creation methods
// Triangle
inline TriMesh Triangle::create_mesh(const ThermalMesh& thermal_mesh,
                                     double tolerance) const {
    // Get the triangle
    const Triangle& triangle = *this;

    const double length_dir1 = triangle.v1().norm();
    const double length_dir2 = triangle.v2().norm();

    // Copy the mesh vectors
    Eigen::VectorXd dir1_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir1_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir1_mesh().size()));

    Eigen::VectorXd dir2_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir2_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir2_mesh().size()));

    // Scale the vectors with the size of the triangle
    dir1_mesh *= length_dir1;
    dir2_mesh *= length_dir2;

    TriMesh trimesh = trimesher::create_2d_triangular_mesh(
        dir1_mesh, dir2_mesh, triangle.from_3d_to_2d(triangle.get_p1()),
        triangle.from_3d_to_2d(triangle.get_p2()),
        triangle.from_3d_to_2d(triangle.get_p3()), tolerance, tolerance);

    std::cout << trimesh.get_faces_edges().size() << '\n';

    // Triangulate the mesh. TODO: This triangulation is trivial. Should be done
    // differently.
    trimesher::cdt_trimesher(trimesh);

    // Assign face ids
    for (Eigen::Index i = 0; i < trimesh.get_triangles().rows(); i++) {
        auto tri = trimesh.get_triangles().row(i);
        const Point2D p0_2d{trimesh.get_vertices()(tri[0], 0),
                            trimesh.get_vertices()(tri[0], 1)};
        const Point2D p1_2d{trimesh.get_vertices()(tri[1], 0),
                            trimesh.get_vertices()(tri[1], 1)};
        const Point2D p2_2d{trimesh.get_vertices()(tri[2], 0),
                            trimesh.get_vertices()(tri[2], 1)};

        const Point2D centroid = (p0_2d + p1_2d + p2_2d) / 3.0;

        // Get the face id
        // i >= 0, so it is safe to cast to VectorIndex
        trimesh.get_face_ids()[i] =
            triangle.get_faceid_from_uv(thermal_mesh, centroid);
    }

    // Transform 2D points to 3D
    for (int i = 0; i < trimesh.get_vertices().rows(); i++) {
        // Be careful with the order of the _vertices matrix
        // This would fail: Point2D point2d = trimesh.get_vertices().row(i)
        //                      Eigen::Map<Point2D>(trimesh.get_vertices().row(i).data(),
        //                      2);
        // Because _vertices is column major
        const Point2D point2d{trimesh.get_vertices()(i, 0),
                              trimesh.get_vertices()(i, 1)};
        trimesh.get_vertices().row(i) = triangle.from_2d_to_3d(point2d);
    }

    // Common operations for all meshes. TODO: create a dedicated function
    trimesh.set_surface1_color(thermal_mesh.get_side1_color().get_rgb());
    trimesh.set_surface2_color(thermal_mesh.get_side2_color().get_rgb());

    // TODO(PERFORMANCE): The mesh should be sorted during creation
    //  sort the mesh
    trimesh.sort_triangles();
    // compute areas
    trimesh.compute_areas();

    return trimesh;
}

// Rectangle
inline TriMesh Rectangle::create_mesh(const ThermalMesh& thermal_mesh,
                                      double /*tolerance*/) const {
    // Get the rectangle
    const Rectangle& rectangle = *this;

    const double length_dir1 = rectangle.v1().norm();
    const double length_dir2 = rectangle.v2().norm();

    // Copy the mesh vectors
    Eigen::VectorXd dir1_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir1_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir1_mesh().size()));

    Eigen::VectorXd dir2_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir2_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir2_mesh().size()));

    // Scale the vectors with the size of the rectangle
    dir1_mesh *= length_dir1;
    dir2_mesh *= length_dir2;

    TriMesh trimesh =
        trimesher::create_2d_rectangular_mesh(dir1_mesh, dir2_mesh, -1.0, -1.0);

    // Triangulate the mesh. TODO: This triangulation is trivial. Should be done
    // differently.
    trimesher::cdt_trimesher(trimesh);

    // Assign face ids
    for (Eigen::Index i = 0; i < trimesh.get_triangles().rows(); i++) {
        auto tri = trimesh.get_triangles().row(i);
        const Point2D p0_2d{trimesh.get_vertices()(tri[0], 0),
                            trimesh.get_vertices()(tri[0], 1)};
        const Point2D p1_2d{trimesh.get_vertices()(tri[1], 0),
                            trimesh.get_vertices()(tri[1], 1)};
        const Point2D p2_2d{trimesh.get_vertices()(tri[2], 0),
                            trimesh.get_vertices()(tri[2], 1)};

        const Point2D centroid = (p0_2d + p1_2d + p2_2d) / 3.0;

        // Get the face id
        // i >= 0, so it is safe to cast to VectorIndex
        trimesh.get_face_ids()[i] =
            rectangle.get_faceid_from_uv(thermal_mesh, centroid);
    }

    // Transform 2D points to 3D
    for (int i = 0; i < trimesh.get_vertices().rows(); i++) {
        // Be careful with the order of the _vertices matrix
        // This would fail: Point2D point2d = trimesh.get_vertices().row(i)
        //                      Eigen::Map<Point2D>(trimesh.get_vertices().row(i).data(),
        //                      2);
        // Because _vertices is column major
        const Point2D point2d{trimesh.get_vertices()(i, 0),
                              trimesh.get_vertices()(i, 1)};
        trimesh.get_vertices().row(i) = rectangle.from_2d_to_3d(point2d);
    }

    // Common operations for all meshes. TODO: create a dedicated function
    trimesh.set_surface1_color(thermal_mesh.get_side1_color().get_rgb());
    trimesh.set_surface2_color(thermal_mesh.get_side2_color().get_rgb());

    // TODO(PERFORMANCE): The mesh should be sorted during creation
    //  sort the mesh
    trimesh.sort_triangles();
    // compute areas
    trimesh.compute_areas();

    return trimesh;
}

inline TriMesh Disc::create_mesh(const ThermalMesh& thermal_mesh,
                                 double tolerance) const {
    // Get the disc
    const Disc& disc = *this;

    using std::numbers::pi;

    // TODO Add checks to validate tolerance and have a minimum/maximum
    // tolerance
    const double max_length_points_dir1 =
        std::sqrt(tolerance * (2 * disc.get_outer_radius() - tolerance));
    const double max_length_points_dir2 =
        std::sqrt(tolerance * (2 * disc.get_end_angle() - tolerance));

    std::cout << "Max length points dir1: " << max_length_points_dir1
              << '\n';  // Copy the mesh vectors
    std::cout << "Max length points dir2: " << max_length_points_dir2
              << '\n';  // Copy the mesh vectors
    Eigen::VectorXd dir1_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir1_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir1_mesh().size()));

    const Eigen::VectorXd dir2_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir2_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir2_mesh().size()));

    Point2D center = disc.from_3d_to_2d(disc.get_p1());
    // Calculate the 2D coordinates of p3 in the local disc plane

    auto start_angle = disc.get_start_angle();
    Point2D outer_point = disc.from_3d_to_2d(disc.get_p3());
    if (start_angle != 0.0) {
        Point2D p3_2d = disc.from_3d_to_2d(disc.get_p3());
        // Point3D vx = v1().normalized();
        // Point3D vy = v2().normalized();

        // Calculate the new 2D coordinates after rotation
        const double new_x_2d =
            p3_2d.x() * cos(start_angle) - p3_2d.y() * sin(start_angle);
        const double new_y_2d =
            p3_2d.x() * sin(start_angle) + p3_2d.y() * cos(start_angle);

        // Calculate the new 3D point by transforming the rotated 2D coordinates
        // back to 3D
        outer_point.x() = new_x_2d;
        outer_point.y() = new_y_2d;
    }

    auto inner_radius = disc.get_inner_radius();
    auto outer_radius = disc.get_outer_radius();
    if (inner_radius != 0.0) {
        for (int i = 0; i < dir1_mesh.size() - 1; i++) {
            dir1_mesh[i] =
                dir1_mesh[i] * (outer_radius - inner_radius) / outer_radius +
                inner_radius / outer_radius;
        }
    }

    std::cout << "Creating mesh" << '\n';
    std::cout << "dir1_mesh" << '\n';
    for (const auto& i : dir1_mesh) {
        std::cout << i << " ";
    }
    std::cout << '\n';
    std::cout << "dir2_mesh" << '\n';
    for (const auto& i : dir2_mesh) {
        std::cout << i << " ";
    }
    std::cout << '\n';
    std::cout << "Center: " << center.x() << ", " << center.y() << '\n';
    std::cout << "Outer point: " << outer_point.x() << ", " << outer_point.y()
              << '\n';
    TriMesh trimesh = trimesher::create_2d_disc_mesh(
        dir1_mesh, dir2_mesh, center, outer_point, max_length_points_dir1,
        max_length_points_dir2);
    // Triangulate the mesh. TODO: This triangulation is trivial. Should be done
    // differently.

    std::cout << "Triangulating mesh" << '\n';
    trimesher::cdt_trimesher(trimesh);

    std::cout << "Assigning face ids" << '\n';
    // Assign face ids
    for (Eigen::Index i = 0; i < trimesh.get_triangles().rows(); i++) {
        auto tri = trimesh.get_triangles().row(i);
        const Point2D p0_2d{trimesh.get_vertices()(tri[0], 0),
                            trimesh.get_vertices()(tri[0], 1)};
        const Point2D p1_2d{trimesh.get_vertices()(tri[1], 0),
                            trimesh.get_vertices()(tri[1], 1)};
        const Point2D p2_2d{trimesh.get_vertices()(tri[2], 0),
                            trimesh.get_vertices()(tri[2], 1)};

        const Point2D centroid = (p0_2d + p1_2d + p2_2d) / 3.0;
        // std::cout << "[" << centroid.x() << ", " << centroid.y() << "]" <<
        // '\n';

        // if (centroid.norm() <= inner_radius) {
        //     std::cout << "[" << p0_2d.x() << ", " << p0_2d.y() << "]," <<
        //     '\n'; std::cout << "[" << p1_2d.x() << ", " << p1_2d.y() <<
        //     "]," << '\n'; std::cout << "[" << p2_2d.x() << ", " <<
        //     p2_2d.y() << "]," << '\n';
        // }

        // Get the face id
        // i >= 0, so it is safe to cast to VectorIndex
        trimesh.get_face_ids()[i] =
            disc.get_faceid_from_uv(thermal_mesh, centroid);
        std::cout << "Face " << i << " id: " << trimesh.get_face_ids()[i]
                  << '\n';
    }

    // Transform 2D points to 3D
    for (int i = 0; i < trimesh.get_vertices().rows(); i++) {
        // Be careful with the order of the _vertices matrix
        // This would fail: Eigen::Map<Point2D>
        // point2d(trimesh.get_vertices().row(i).data(), 2);; Because _vertices
        // is column major
        const Point2D point2d{trimesh.get_vertices()(i, 0),
                              trimesh.get_vertices()(i, 1)};
        trimesh.get_vertices().row(i) = disc.from_2d_to_3d(point2d);
    }

    // Common operations for all meshes. TODO: create a dedicated function
    trimesh.set_surface1_color(thermal_mesh.get_side1_color().get_rgb());
    trimesh.set_surface2_color(thermal_mesh.get_side2_color().get_rgb());
    // sort the mesh
    trimesh.sort_triangles();
    // compute areas
    trimesh.compute_areas();

    return trimesh;
}

inline TriMesh Cylinder::create_mesh(const ThermalMesh& thermal_mesh,
                                     double tolerance) const {
    // Get the cylinder
    const Cylinder& cylinder = *this;

    using std::numbers::pi;

    // TODO Add checks to validate tolerance and have a minimum/maximum
    // tolerance
    const double max_length_points_dir1 =
        std::sqrt(tolerance * (2 * cylinder.get_radius() - tolerance));

    const double length_dir1 =
        (cylinder.get_end_angle() - cylinder.get_start_angle()) *
        cylinder.get_radius();
    const double length_dir2 = (cylinder.get_p2() - cylinder.get_p1()).norm();

    // Copy the mesh vectors
    Eigen::VectorXd dir1_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir1_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir1_mesh().size()));

    Eigen::VectorXd dir2_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir2_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir2_mesh().size()));

    // Scale the vectors with the size of the rectangle
    dir1_mesh *= length_dir1;
    dir2_mesh *= length_dir2;

    TriMesh trimesh = trimesher::create_2d_rectangular_mesh(
        dir1_mesh, dir2_mesh, max_length_points_dir1, -1.0);
    // Triangulate the mesh. TODO: This triangulation is trivial. Should be done
    // differently.
    trimesher::cdt_trimesher(trimesh);

    // Assign face ids
    for (Eigen::Index i = 0; i < trimesh.get_triangles().rows(); i++) {
        auto tri = trimesh.get_triangles().row(i);
        const Point2D p0_2d{trimesh.get_vertices()(tri[0], 0),
                            trimesh.get_vertices()(tri[0], 1)};
        const Point2D p1_2d{trimesh.get_vertices()(tri[1], 0),
                            trimesh.get_vertices()(tri[1], 1)};
        const Point2D p2_2d{trimesh.get_vertices()(tri[2], 0),
                            trimesh.get_vertices()(tri[2], 1)};

        const Point2D centroid = (p0_2d + p1_2d + p2_2d) / 3.0;

        // Get the face id
        // i >= 0, so it is safe to cast to VectorIndex
        trimesh.get_face_ids()[i] =
            cylinder.get_faceid_from_uv(thermal_mesh, centroid);
    }

    // Transform 2D points to 3D
    for (int i = 0; i < trimesh.get_vertices().rows(); i++) {
        // Be careful with the order of the _vertices matrix
        // This would fail: Eigen::Map<Point2D>
        // point2d(trimesh.get_vertices().row(i).data(), 2);; Because _vertices
        // is column major
        const Point2D point2d{trimesh.get_vertices()(i, 0),
                              trimesh.get_vertices()(i, 1)};
        trimesh.get_vertices().row(i) = cylinder.from_2d_to_3d(point2d);
    }

    // Common operations for all meshes. TODO: create a dedicated function
    trimesh.set_surface1_color(thermal_mesh.get_side1_color().get_rgb());
    trimesh.set_surface2_color(thermal_mesh.get_side2_color().get_rgb());
    // sort the mesh
    trimesh.sort_triangles();
    // compute areas
    trimesh.compute_areas();

    return trimesh;
}

// TODO: SOLVE THE CLANG-TIDY WARNING. REDUCE COMPLEXITY OF THE FUNCTION
// NOLINTBEGIN(readability-function-cognitive-complexity)

inline TriMesh Cone::create_mesh(const ThermalMesh& thermal_mesh,
                                 double tolerance) const {
    // Get the cone
    const Cone& cone = *this;

    using std::numbers::pi;

    // Main directions of the 3D cone
    const Eigen::Vector3d vx = (_p3 - _p1).normalized();
    const Eigen::Vector3d vz = (_p2 - _p1).normalized();
    const Eigen::Vector3d vy = (vz.cross(vx)).normalized();

    // Length of the side of the cone
    const double h = (_p2 - _p1).norm();
    const double s1 =
        sqrt(h * h + abs(_radius2 - _radius1) * abs(_radius2 - _radius1));
    const double s2 = (_radius1 < _radius2) ? _radius1 * s1 / _radius2
                                            : _radius2 * s1 / _radius1;
    const double s = s1 + s2;

    // TODO Add checks to validate tolerance and have a minimum/maximum
    // tolerance
    const double max_length_points_dir1 =
        (_radius1 < _radius2)
            ? std::sqrt(tolerance * (2 * cone.get_radius2() - tolerance))
            : std::sqrt(tolerance * (2 * cone.get_radius1() - tolerance));
    const double max_length_points_dir2 =
        std::sqrt(tolerance * (2 * cone.get_end_angle() - tolerance));

    // double length_dir1 = abs(cone.get_radius2() - cone.get_radius1());
    // double angle_dir2 = cone.get_end_angle() - cone.get_start_angle();

    // Copy the normalized mesh vectors
    Eigen::VectorXd dir1_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir1_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir1_mesh().size()));

    Eigen::VectorXd dir2_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir2_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir2_mesh().size()));

    // Remap dir1_mesh to the 2D plane
    std::transform(dir1_mesh.begin(), dir1_mesh.end(), dir1_mesh.begin(),
                   [s1, s2, s](auto value) { return (value * s1 + s2) / s; });
    // Same as:
    // for (auto& value : dir1_mesh) {
    //    value = (value * s1 + s2) / s;
    //}

    // Remap dir2_mesh to the 2D plane
    // TODO: Check if the value of the mesh is relative to the angle of the cone
    // or to the full circle
    for (auto& value : dir2_mesh) {
        value = (_start_angle + value * (_end_angle - _start_angle)) / (2 * pi);
        value =
            (_radius1 < _radius2) ? value * _radius2 / s : value * _radius1 / s;
    }

    Point2D center = (_radius1 < _radius2) ? cone.from_3d_to_2d(_p1)
                                           : cone.from_3d_to_2d(_p2);
    // Rotate p3-p1 start angle rads around p2-p1
    Point2D outer_point = (_radius1 < _radius2)
                              ? cone.from_3d_to_2d(vx * _radius2 + _p2)
                              : cone.from_3d_to_2d(vx * _radius1 + _p1);

    if (_start_angle != 0.0) {
        const double theta = cone.get_start_angle();
        const Point3D p3d_r = vx * cos(theta) + vy * sin(theta);
        // Calculate the 2D coordinates of p3 in the local cone planeñ
        outer_point = (_radius1 < _radius2)
                          ? from_3d_to_2d(p3d_r * _radius2 + _p2)
                          : from_3d_to_2d(p3d_r * _radius1 + _p1);
    }

    // TODO: Check if when using start and end angle the mesh takes them into
    // account

    std::cout << "Center: " << center.x() << ", " << center.y() << '\n';
    std::cout << "Outer point: " << outer_point.x() << ", " << outer_point.y()
              << '\n';
    std::cout << "Max length points dir1: " << max_length_points_dir1 << '\n';
    std::cout << "Max length points dir2: " << max_length_points_dir2 << '\n';

    std::cout << "dir1_mesh: [";
    for (const auto& i : dir1_mesh) {
        std::cout << i << ", ";
    }
    std::cout << "]" << '\n';
    std::cout << "dir2_mesh: [";
    for (const auto& i : dir2_mesh) {
        std::cout << i << ", ";
    }
    std::cout << "]" << '\n';

    TriMesh trimesh = trimesher::create_2d_disc_mesh(
        dir1_mesh, dir2_mesh, center, outer_point, max_length_points_dir1,
        max_length_points_dir2);

    // Triangulate the mesh. TODO: This triangulation is trivial. Should be done
    // differently.
    trimesher::cdt_trimesher(trimesh);

    // Assign face ids
    for (Index i = 0; i < trimesh.get_triangles().rows(); i++) {
        auto tri = trimesh.get_triangles().row(static_cast<Eigen::Index>(i));
        const Point2D p0_2d{trimesh.get_vertices()(tri[0], 0),
                            trimesh.get_vertices()(tri[0], 1)};
        const Point2D p1_2d{trimesh.get_vertices()(tri[1], 0),
                            trimesh.get_vertices()(tri[1], 1)};
        const Point2D p2_2d{trimesh.get_vertices()(tri[2], 0),
                            trimesh.get_vertices()(tri[2], 1)};

        const Point2D centroid = (p0_2d + p1_2d + p2_2d) / 3.0;

        // Get the face id
        // i >= 0, so it is safe to cast to VectorIndex
        trimesh.get_face_ids()[i] =
            cone.get_faceid_from_uv(thermal_mesh, centroid);
    }

    // sort the mesh
    trimesh.sort_triangles();

    // If the cone is a full circle, remove the repeating edges and points
    if (_end_angle - _start_angle == 2 * pi) {
        auto old_vertices = trimesh.get_vertices();
        auto old_edges = trimesh.get_edges();
        auto perimeter_edges = trimesh.get_perimeter_edges();
        auto faces_edges = trimesh.get_faces_edges();

        auto dir1_size = static_cast<MeshIndex>(dir1_mesh.size());
        auto dir2_size = static_cast<MeshIndex>(dir2_mesh.size());

        std::vector<MeshIndex> edges_to_remove(dir1_size - 1);
        for (MeshIndex i = 0; i < dir1_size - 1; ++i) {
            edges_to_remove[i] =
                static_cast<MeshIndex>((dir1_size - 1) * (dir2_size - 1) + i);
        }

        // int edge_idx = 0;
        std::set<MeshIndex> vertices_to_remove_set;
        for (const auto& edge_idx : edges_to_remove) {
            auto edge = old_edges[edge_idx];
            for (const auto& vert_idx : edge) {
                if ((_radius1 == 0.0 || _radius2 == 0.0) && vert_idx == 0) {
                    continue;
                }
                vertices_to_remove_set.insert(vert_idx);
            }
        }

        // Create a new matrix without the contact points
        const auto num_rows_to_remove =
            static_cast<Index>(vertices_to_remove_set.size());
        const auto num_rows_in_original_matrix = old_vertices.rows();
        const auto num_rows_in_new_matrix =
            num_rows_in_original_matrix - num_rows_to_remove;

        VerticesList reduced_vertices(num_rows_in_new_matrix, 3);
        int new_row = 0;

        for (MeshIndex row = 0; row < num_rows_in_original_matrix; ++row) {
            if (vertices_to_remove_set.find(row) ==
                vertices_to_remove_set.end()) {
                reduced_vertices.row(new_row) = old_vertices.row(row);
                ++new_row;
            }
        }

        // Remove the contact edges
        auto reduced_edges = old_edges;
        reduced_edges.erase(reduced_edges.begin() + edges_to_remove[0],
                            reduced_edges.begin() +
                                edges_to_remove[edges_to_remove.size() - 1] +
                                1);

        const MeshIndex last_idx_edge = (dir1_size - 1) * (dir2_size - 1);
        auto last_dir1_point = *vertices_to_remove_set.begin() - 1;
        auto points_skip =
            static_cast<MeshIndex>(vertices_to_remove_set.size());

        for (auto& face_edge : faces_edges) {
            for (auto& edge : face_edge) {
                if (edge >= static_cast<MeshIndex>(last_idx_edge) &&
                    edge < static_cast<MeshIndex>(last_idx_edge +
                                                  (dir1_size - 1))) {
                    edge -= last_idx_edge;
                } else if (edge >= static_cast<MeshIndex>(last_idx_edge +
                                                          (dir1_size - 1))) {
                    edge -= (dir1_size - 1);
                }
            }
        }

        // Update the perimeter edges of the cone
        // If the cone is not a full circle, the perimeter edges are not updated

        // Calculate the start index for the elements to keep
        const MeshIndex start_idx = (dir1_size - 1) * 2;
        const MeshIndex num_elements_to_keep =
            static_cast<MeshIndex>(perimeter_edges.rows()) - start_idx;
        // Create a new matrix with the elements to keep
        const EdgesIdsList temp =
            perimeter_edges.block(start_idx, 0, num_elements_to_keep, 1);
        // Move the temp matrix back to perimeter_edges
        perimeter_edges = temp;

        // The radial edges do not need to be updated
        // Update circumferencial edges
        const MeshIndex start = (_radius1 == 0.0 || _radius2 == 0.0) ? 1 : 0;
        for (MeshIndex edge_id = (dir1_size - 1) * (dir2_size - 1);
             edge_id < reduced_edges.size(); ++edge_id) {
            auto& edge = reduced_edges[edge_id];  // Use a reference here if you
                                                  // need to modify the edge
            for (auto& vert :
                 edge) {  // Use a reference to modify elements in-place
                if (vert > last_dir1_point &&
                    vert <= last_dir1_point + points_skip) {
                    vert += start - (last_dir1_point + 1);
                } else if (vert > last_dir1_point) {
                    vert -= points_skip;
                }
            }
        }

        // Update the triangles
        auto triangles = trimesh.get_triangles();
        for (int tri = 0; tri < triangles.rows(); ++tri) {
            for (int col = 0; col < 3; ++col) {
                if (triangles(tri, col) > last_dir1_point &&
                    triangles(tri, col) <= last_dir1_point + points_skip) {
                    triangles(tri, col) =
                        triangles(tri, col) - (last_dir1_point + 1) + start;
                } else if (triangles(tri, col) >
                           last_dir1_point + points_skip) {
                    triangles(tri, col) -= points_skip;
                }
            }
        }

        trimesh.set_vertices(std::move(reduced_vertices));
        trimesh.set_edges(std::move(reduced_edges));
        trimesh.set_perimeter_edges(std::move(perimeter_edges));
        trimesh.set_faces_edges(std::move(faces_edges));
        trimesh.set_triangles(std::move(triangles));
    }

    // Transform 2D points to 3D
    for (int i = 0; i < trimesh.get_vertices().rows(); i++) {
        // Be careful with the order of the _vertices matrix
        // This would fail: Eigen::Map<Point2D>
        // point2d(trimesh.get_vertices().row(i).data(), 2);; Because _vertices
        // is column major
        const Point2D point2d{trimesh.get_vertices()(i, 0),
                              trimesh.get_vertices()(i, 1)};
        trimesh.get_vertices().row(i) = cone.from_2d_to_3d(point2d);
    }

    // Common operations for all meshes. TODO: create a dedicated function
    trimesh.set_surface1_color(thermal_mesh.get_side1_color().get_rgb());
    trimesh.set_surface2_color(thermal_mesh.get_side2_color().get_rgb());
    // compute areas
    trimesh.compute_areas();

    return trimesh;
}

// NOLINTEND(readability-function-cognitive-complexity)

// TODO: SOLVE THE CLANG-TIDY WARNING. REDUCE COMPLEXITY OF THE FUNCTION
// NOLINTBEGIN(readability-function-cognitive-complexity)
inline TriMesh Sphere::create_mesh1(const ThermalMesh& thermal_mesh,
                                    double tolerance) const {
    using std::numbers::pi;

    // return create_mesh2(thermal_mesh, tolerance);

    // Main directions of the 3D cone
    // Eigen::Vector3d vx = (_p3 - _p1).normalized();
    // Eigen::Vector3d vz = (_p2 - _p1).normalized();
    // Eigen::Vector3d vy = (vz.cross(vx)).normalized();

    // TODO Add checks to validate tolerance and have a minimum/maximum
    // tolerance
    const double max_length_points_dir1 = std::sqrt(
        tolerance *
        (2 * (_apex_truncation - _base_truncation) * _radius - tolerance));
    const double max_length_points_dir2 = std::sqrt(
        tolerance * (2 * (_end_angle - _start_angle) * _radius - tolerance));

    // Copy the normalized mesh vectors
    Eigen::VectorXd dir1_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir1_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir1_mesh().size()));

    Eigen::VectorXd dir2_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir2_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir2_mesh().size()));

    std::vector<double> dir2_mesh_normalized(
        static_cast<VectorIndex>(dir2_mesh.size()));
    for (MeshIndex i = 0; i < dir2_mesh.size(); ++i) {
        dir2_mesh_normalized[i] =
            (_start_angle + dir2_mesh[i] * (_end_angle - _start_angle)) /
            (2 * pi);
    }

    // 1. Determine the number of points to reserve space
    const auto dir1_size = static_cast<MeshIndex>(dir1_mesh.size());
    const auto dir2_size = static_cast<MeshIndex>(dir2_mesh.size());

    MeshIndex num_points = 0;

    // Dir1 is the vz direction. The number of points in each meridian is the
    // same
    std::vector<MeshIndex> additional_points_dir1(dir1_size - 1, 0);

    // Dir1 start is 0 if the base is not truncated
    const MeshIndex dir1_start = (_base_truncation != -_radius) ? 0 : 1;
    // Dir1 end is 0 if the apex is not truncated
    const MeshIndex dir1_end = (_apex_truncation != _radius) ? 0 : 1;
    // Dir2 end is 0 if the end angle is not 2*pi
    const MeshIndex dir2_end = (_end_angle - _start_angle != 2 * pi) ? 0 : 1;

    // Dir2 is the angular direction. Because the length of the circumference
    // changes for each of the divisions in dir1, a different number of points
    // might be needed
    std::vector<std::vector<MeshIndex>> additional_points_dir2(
        dir1_size - (dir1_start + dir1_end),
        std::vector<MeshIndex>(dir2_size - 1));

    MeshIndex num_points_row_dir1 = dir1_size;
    MeshIndex num_points_dir2 = 0;

    // Points in direction 1, the same for each radius
    if (max_length_points_dir1 > LENGTH_TOL) {
        for (MeshIndex i = 0; i < dir1_size - 1; ++i) {
            const double z1 =
                _base_truncation +
                dir1_mesh[i] * (_apex_truncation - _base_truncation);
            const double z2 =
                _base_truncation +
                dir1_mesh[i + 1] * (_apex_truncation - _base_truncation);
            const double ph1 = asin(z1 / _radius);
            const double ph2 = asin(z2 / _radius);
            const double distance = _radius * abs(ph2 - ph1);
            const auto num_additional_points = static_cast<MeshIndex>(
                std::floor(distance / max_length_points_dir1));
            num_points_row_dir1 += num_additional_points;
            additional_points_dir1[i] = num_additional_points;
        }
    }

    // Points in direction 2, different for each radius and angle
    if (max_length_points_dir2 > LENGTH_TOL) {
        for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
            const double angle_i =
                (dir2_mesh_normalized[i + 1] - dir2_mesh_normalized[i]) * 2 *
                pi;
            for (MeshIndex j = 0; j < dir1_size - (dir1_start + dir1_end);
                 ++j) {
                const double z =
                    (j < dir1_size - 1 - (dir1_start + dir1_end))
                        ? _base_truncation +
                              dir1_mesh[j + dir1_start] *
                                  (_apex_truncation - _base_truncation)
                        : _base_truncation + dir1_mesh[j] * (_apex_truncation -
                                                             _base_truncation);

                const double distance =
                    angle_i * _radius * sin(acos(z / _radius));
                const auto num_additional_points = static_cast<MeshIndex>(
                    std::floor(distance / max_length_points_dir2));
                num_points_dir2 += num_additional_points;
                additional_points_dir2[j][i] = num_additional_points;
            }
        }
    }

    // Determine if there are interior points. For the disc
    // only the dir1 points generate interior points and only
    // they are the same for a given radius
    // std::vector<std::vector<MeshIndex>> interior_points_dir2(
    //    dir1_size - 1, std::vector<MeshIndex>(dir2_size - 1));
    MeshIndex num_interior_points = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        if (additional_points_dir1[i_dir1] > 0) {
            auto add_pi1 = additional_points_dir1[i_dir1];
            for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
                MeshIndex num_interior_points_i = 0;
                num_interior_points_i =
                    (i_dir1 == dir1_size - 2 && dir1_end == 1)
                        ? add_pi1 * additional_points_dir2[i_dir1 - 1][j_dir2]
                        : add_pi1 * additional_points_dir2[i_dir1][j_dir2];
                num_interior_points += num_interior_points_i;
                // interior_points_dir2[i_dir1][j_dir2] = num_interior_points_i;
            }
        }
    }

    // Calculate the total number of points
    MeshIndex num_points_dir1 = num_points_row_dir1;
    num_points_dir1 =
        (num_points_dir1 - (dir1_start + dir1_end)) * (dir2_size - dir2_end) +
        1 * dir1_start + 1 * dir1_end;

    num_points = num_points_dir1 + num_points_dir2 + num_interior_points;
    VerticesList points(num_points, 3);
    std::vector<double> full_dir1_mesh(num_points_row_dir1);
    std::vector<double> full_dir2_mesh(num_points_dir2);
    // Fill the full mesh arrays with the original and additional mesh
    // points
    // Dir 1
    MeshIndex p_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        const double distance = (dir1_mesh[i_dir1 + 1] - dir1_mesh[i_dir1]) *
                                (_apex_truncation - _base_truncation) /
                                (additional_points_dir1[i_dir1] + 1);
        if ((_base_truncation == -_radius) && (i_dir1 == 0)) {
            full_dir1_mesh[0] = _base_truncation;
            ++p_idx;
            for (MeshIndex i_add1 = 1; i_add1 <= additional_points_dir1[i_dir1];
                 ++i_add1) {
                full_dir1_mesh[p_idx] =
                    _base_truncation +
                    dir1_mesh[i_dir1] * (_apex_truncation - _base_truncation) +
                    i_add1 * distance;
                ++p_idx;
            }
        } else {
            for (MeshIndex i_add1 = 0; i_add1 <= additional_points_dir1[i_dir1];
                 ++i_add1) {
                full_dir1_mesh[p_idx] =
                    _base_truncation +
                    dir1_mesh[i_dir1] * (_apex_truncation - _base_truncation) +
                    i_add1 * distance;
                ++p_idx;
            }
        }
    }
    full_dir1_mesh[full_dir1_mesh.size() - 1] = _apex_truncation;

    // Dir 2
    p_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - (dir1_start + dir1_end);
         ++i_dir1) {
        // Angle of the dir2 division
        for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
            const double angle_i = (dir2_mesh_normalized[j_dir2 + 1] -
                                    dir2_mesh_normalized[j_dir2]) *
                                   2 * pi;

            // Angle between points of the dir2 division
            const double distance =
                angle_i / (additional_points_dir2[i_dir1][j_dir2] + 1);
            for (MeshIndex i_add1 = 1;
                 i_add1 <= additional_points_dir2[i_dir1][j_dir2]; ++i_add1) {
                full_dir2_mesh[p_idx] =
                    dir2_mesh_normalized[j_dir2] * 2 * pi + i_add1 * distance;
                ++p_idx;
            }
        }
    }

    // Fill the points array in dir1
    p_idx = 0;
    if (_base_truncation == -_radius) {
        points(0, 0) = 0.0;
        points(0, 1) = 0.0;
        points(0, 2) = -_radius;
        ++p_idx;
    }

    const auto full_dir1_mesh_size = static_cast<Index>(full_dir1_mesh.size());

    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - dir2_end; ++i_dir2) {
        auto angle_i = dir2_mesh_normalized[i_dir2] * 2 * pi;
        for (MeshIndex i_dir1 = dir1_start;
             i_dir1 < full_dir1_mesh_size - dir1_end; ++i_dir1) {
            const double phi = acos(full_dir1_mesh[i_dir1] / _radius);
            points(p_idx, 0) = _p1.x() + round(_radius * sin(phi) *
                                               cos(angle_i) / LENGTH_TOL) *
                                             LENGTH_TOL;
            points(p_idx, 1) = _p1.y() + round(_radius * sin(phi) *
                                               sin(angle_i) / LENGTH_TOL) *
                                             LENGTH_TOL;
            points(p_idx, 2) = _p1.z() + full_dir1_mesh[i_dir1];
            ++p_idx;
        }
    }

    const MeshIndex north_pole_idx = p_idx;
    if (_apex_truncation == _radius) {
        points(p_idx, 0) = 0.0;
        points(p_idx, 1) = 0.0;
        points(p_idx, 2) = _radius;
        ++p_idx;
    }

    // Fill the points array in dir2
    MeshIndex dir2_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - (dir1_start + dir1_end);
         ++i_dir1) {
        for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
            for (MeshIndex i_add2 = 0;
                 i_add2 < additional_points_dir2[i_dir1][i_dir2]; ++i_add2) {
                const double z = _base_truncation +
                                 dir1_mesh[i_dir1 + dir1_start] *
                                     (_apex_truncation - _base_truncation);
                auto angle_i = full_dir2_mesh[dir2_idx];
                const double phi = acos(z / _radius);
                points(p_idx, 0) = _p1.x() + round(_radius * sin(phi) *
                                                   cos(angle_i) / LENGTH_TOL) *
                                                 LENGTH_TOL;
                points(p_idx, 1) = _p1.y() + round(_radius * sin(phi) *
                                                   sin(angle_i) / LENGTH_TOL) *
                                                 LENGTH_TOL;
                points(p_idx, 2) = _p1.z() + z;
                ++p_idx;
                ++dir2_idx;
            }
        }
    }

    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        auto add_pi1 = additional_points_dir1[i_dir1];
        MeshIndex a_i = 0;
        a_i = (i_dir1 == dir1_size - 2 && dir1_end == 1)
                  ? i_dir1 - 1
                  : i_dir1 + 1 - dir1_start;
        const double z1 =
            _base_truncation +
            dir1_mesh[i_dir1] * (_apex_truncation - _base_truncation);
        const double z2 =
            _base_truncation +
            dir1_mesh[i_dir1 + 1] * (_apex_truncation - _base_truncation);
        const double ph1 = asin(z1 / _radius);
        const double ph2 = asin(z2 / _radius);
        const double d_ph = ph2 - ph1;
        for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
            const double d_angle = (dir2_mesh_normalized[j_dir2 + 1] -
                                    dir2_mesh_normalized[j_dir2]);
            for (MeshIndex p = 1; p <= add_pi1; ++p) {
                const double phi_i = ph1 + (p - 1) * d_ph / (add_pi1 + 1) +
                                     d_ph / (2 * (add_pi1 + 1));
                const double z = _radius * sin(phi_i);
                for (MeshIndex a = 1; a <= additional_points_dir2[a_i][j_dir2];
                     ++a) {
                    auto angle_a =
                        (dir2_mesh_normalized[j_dir2] +
                         d_angle / (1 + additional_points_dir2[a_i][j_dir2]) *
                             a) *
                        2 * pi;
                    const double phi = acos(z / _radius);
                    points(p_idx, 0) =
                        _p1.x() +
                        round(_radius * sin(phi) * cos(angle_a) / LENGTH_TOL) *
                            LENGTH_TOL;
                    points(p_idx, 1) =
                        _p1.y() +
                        round(_radius * sin(phi) * sin(angle_a) / LENGTH_TOL) *
                            LENGTH_TOL;
                    points(p_idx, 2) = _p1.z() + z;
                    ++p_idx;
                }
            }
        }
    }

    // 3. Create the edges

    // Reserve space for the edges
    MeshIndex edges_size =
        (dir1_size - 1) * (dir2_size) + (dir1_size) * (dir2_size - 1);
    if (_apex_truncation == _radius) {
        edges_size -= dir2_size - 1;
    }
    if (_base_truncation == -_radius) {
        edges_size -= dir2_size - 1;
    }
    if (_end_angle - _start_angle == 2 * pi) {
        edges_size -= dir1_size - 1;
    }
    EdgesList edges(edges_size);

    // Fill the edges
    // First the edges in direction 1
    MeshIndex e_idx = 0;
    p_idx = 0;
    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - dir2_end; ++i_dir2) {
        for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
            const MeshIndex num_edge_points =
                additional_points_dir1[i_dir1] + 2;

            Edges edge(num_edge_points);
            if (_base_truncation == -_radius && i_dir1 == 0) {
                edge[0] = 0;
            } else {
                edge[0] = p_idx;
            }
            for (MeshIndex i_add1 = 0; i_add1 <= additional_points_dir1[i_dir1];
                 ++i_add1) {
                edge[i_add1 + 1] = p_idx + i_add1 + 1;
            }
            if (_apex_truncation == _radius && i_dir1 == dir1_size - 2) {
                edge[num_edge_points - 1] = north_pole_idx;
            }
            edges[e_idx] = edge;
            e_idx++;
            p_idx += num_edge_points - 1;
        }
        if (_base_truncation != -_radius && _apex_truncation != _radius) {
            p_idx += 1;
        }
        if (_base_truncation == -_radius && _apex_truncation == _radius) {
            p_idx -= 1;
        }
    }

    if (_apex_truncation == _radius) {
        p_idx += 1;
    }
    if (_base_truncation == -_radius) {
        p_idx += 1;
    }

    // Then the edges in direction 2
    MeshIndex dir1_idx = 0;
    if (_base_truncation == -_radius) {
        dir1_idx = 1 + additional_points_dir1[0];
    }
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - (dir1_start + dir1_end);
         ++i_dir1) {
        MeshIndex end_idx = dir1_idx;
        for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
            const MeshIndex num_edge_points =
                additional_points_dir2[i_dir1][j_dir2] + 2;
            Edges edge(num_edge_points);
            edge[0] = end_idx;
            for (MeshIndex i_add2 = 0;
                 i_add2 < additional_points_dir2[i_dir1][j_dir2]; ++i_add2) {
                edge[i_add2 + 1] = p_idx;
                ++p_idx;
            }
            end_idx += num_points_dir1 / (dir2_size - dir2_end);
            if ((_end_angle - _start_angle == 2 * pi) &&
                (j_dir2 == dir2_size - dir2_end - 1)) {
                edge[num_edge_points - 1] = dir1_idx;
            } else {
                edge[num_edge_points - 1] = end_idx;
            }
            edges[e_idx] = edge;
            e_idx++;
        }
        if (i_dir1 < dir1_size - (dir1_start + dir1_end) - 1) {
            if (_base_truncation == -_radius) {
                dir1_idx += additional_points_dir1[i_dir1 + 1] + 1;
            } else {
                dir1_idx += additional_points_dir1[i_dir1] + 1;
            }
        }
    }

    // 4. Create the permiter edges. Anti-clockwise order starting from P01
    // -> dir1 edge
    MeshIndex num_perimeter_edges = dir1_size - 1;
    if (_end_angle - _start_angle != 2 * pi) {
        num_perimeter_edges += dir1_size - 1;
    }
    if (_base_truncation != -_radius) {
        num_perimeter_edges += dir2_size - 1;
    }
    if (_apex_truncation != _radius) {
        num_perimeter_edges += dir2_size - 1;
    }

    EdgesIdsList perimeter_edges(num_perimeter_edges);
    MeshIndex per_edge_idx = 0;

    // _start_angle edge
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        perimeter_edges[per_edge_idx] = i_dir1;
        ++per_edge_idx;
    }

    // _apex_truncation edge
    if (_apex_truncation != _radius) {
        for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
            perimeter_edges[per_edge_idx] =
                static_cast<MeshIndex>(edges.size()) - (dir2_size - 1) + i_dir2;
            ++per_edge_idx;
        }
    }

    // _end_angle edge
    if (_end_angle - _start_angle != 2 * pi) {
        for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
            perimeter_edges[per_edge_idx] =
                i_dir1 + (dir1_size - 1) * (dir2_size - 1);
            ++per_edge_idx;
        }
    }

    // _base_truncation edge
    if (_base_truncation != -_radius) {
        for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
            perimeter_edges[per_edge_idx] =
                (_end_angle - _start_angle != 2 * pi)
                    ? i_dir2 + dir2_size * (dir1_size - 1)
                    : i_dir2 + (dir2_size - 1) * (dir1_size - 1);
            ++per_edge_idx;
        }
    }

    // 5. Create the faces edges
    const MeshIndex num_faces = (dir1_size - 1) * (dir2_size - 1);
    FaceEdges faces_edges(num_faces);
    MeshIndex face_idx = 0;
    const MeshIndex skip_horizontal_edges =
        (dir2_size - dir2_end) * (dir1_size - 1);
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
            if (_base_truncation == -_radius && (i_dir1 == 0)) {
                EdgesIdsList face(3);
                face[0] = i_dir1 + (dir1_size - 1) * j_dir2;
                face[1] = j_dir2 + skip_horizontal_edges;
                face[2] = (_end_angle - _start_angle == 2 * pi &&
                           j_dir2 == dir2_size - 2)
                              ? 0
                              : i_dir1 + (dir1_size - 1) * (j_dir2 + 1);
                faces_edges[face_idx] = face;
            } else if (_apex_truncation == _radius && i_dir1 == dir1_size - 2) {
                EdgesIdsList face(3);
                face[0] = i_dir1 + (dir1_size - 1) * j_dir2;
                face[1] = (_end_angle - _start_angle == 2 * pi &&
                           j_dir2 == dir2_size - 2)
                              ? 0
                              : i_dir1 + (dir1_size - 1) * (j_dir2 + 1);
                face[2] = j_dir2 + (dir2_size - 1) * (i_dir1 - dir1_start) +
                          skip_horizontal_edges;
                faces_edges[face_idx] = face;
            } else {
                EdgesIdsList face(4);
                face[0] = i_dir1 + (dir1_size - 1) * j_dir2;
                face[1] = j_dir2 + (dir2_size - 1) * (i_dir1 - dir1_start + 1) +
                          skip_horizontal_edges;
                face[2] = (_end_angle - _start_angle == 2 * pi &&
                           j_dir2 == dir2_size - 2)
                              ? i_dir1
                              : i_dir1 + (dir1_size - 1) * (j_dir2 + 1);
                face[3] = j_dir2 + (dir2_size - 1) * (i_dir1 - dir1_start) +
                          skip_horizontal_edges;
                faces_edges[face_idx] = face;
            }

            ++face_idx;
        }
    }

    TriMesh trimesh;

    trimesh.set_faces_edges(std::move(faces_edges));
    trimesh.set_number_of_faces(num_faces * 2);

    // Transform 3D points to 2D
    auto num_points_2d = static_cast<MeshIndex>(points.rows());
    if (_end_angle - _start_angle == 2 * pi) {
        num_points_2d += num_points_row_dir1;
        if (_base_truncation == -_radius) {
            num_points_2d -= 1;
        }
        if (_apex_truncation == _radius) {
            num_points_2d -= 1;
        }
    }

    VerticesList points_2d(num_points_2d, 3);
    for (MeshIndex i = 0; i < num_points; ++i) {
        auto lon_lat = from_cartesian_to_spherical(points.row(i));
        auto p2d = from_3d_to_2d_sinusoidal(
            lon_lat, (_end_angle - _start_angle) / 2 - pi);
        points_2d(i, 0) = p2d.x();
        points_2d(i, 1) = p2d.y();
        points_2d(i, 2) = 0;
    }

    if (_end_angle - _start_angle == 2 * pi) {
        Edges edge(num_points_2d - num_points + dir1_start + dir1_end);
        if (_base_truncation == -_radius) {
            edge[0] = 0;
        }
        if (_apex_truncation == _radius) {
            edge[edge.size() - 1] = north_pole_idx;
        }
        for (MeshIndex i = 0; i < num_points_2d - num_points; ++i) {
            points_2d(num_points + i, 0) = -points_2d(i + dir1_start, 0);
            points_2d(num_points + i, 1) = points_2d(i + dir1_start, 1);
            points_2d(num_points + i, 2) = points_2d(i + dir1_start, 2);
            edge[i + dir1_start] = num_points + i;
        }
        edges.insert(edges.end(), edge);

        // Insert edges.size() - 1 to perimeter_edges
        perimeter_edges.conservativeResize(perimeter_edges.size() + 1,
                                           Eigen::NoChange);
        perimeter_edges.tail<1>()(0) = static_cast<MeshIndex>(edges.size() - 1);

        // Update the latitude edges that close the sphere
        for (MeshIndex i = 0; i < dir1_size - (dir1_start + dir1_end); ++i) {
            const MeshIndex index = (dir1_size - 1) * (dir2_size - 1) +
                                    (dir2_size - 1) * i + dir2_size - 2;
            edges[index][edges[index].size() - 1] += (num_points - dir1_start);
        }
    }

    trimesh.set_vertices(points_2d);  // points 2D are used later, don't know
                                      // why. Can't move them.
    trimesh.set_edges(std::move(edges));
    trimesh.set_perimeter_edges(std::move(perimeter_edges));

    // Triangulate the mesh
    trimesher::cdt_trimesher(trimesh);

    if (_end_angle - _start_angle == 2 * pi) {
        // Erase added edge
        trimesh.get_edges().erase(trimesh.get_edges().end() - 1,
                                  trimesh.get_edges().end());

        // Update the latitude edges that close the sphere
        for (MeshIndex i = 0; i < dir1_size - (dir1_start + dir1_end); ++i) {
            const MeshIndex index = (dir1_size - 1) * (dir2_size - 1) +
                                    (dir2_size - 1) * i + dir2_size - 2;
            trimesh.get_edges()[index][trimesh.get_edges()[index].size() - 1] -=
                (num_points - dir1_start);
        }
        // Erase added permiter edge
        trimesh.get_perimeter_edges().conservativeResize(
            trimesh.get_perimeter_edges().size() - 1, Eigen::NoChange);

        // Renumber last triangles
        for (MeshIndex i = 0; i < trimesh.get_triangles().rows(); ++i) {
            for (MeshIndex j = 0; j < trimesh.get_triangles().cols(); ++j) {
                if (trimesh.get_triangles()(i, j) >= num_points) {
                    if (trimesh.get_triangles()(i, j) == points_2d.rows() - 1 &&
                        _apex_truncation == _radius) {
                        trimesh.get_triangles()(i, j) = north_pole_idx;
                    } else {
                        trimesh.get_triangles()(i, j) -=
                            (num_points - dir1_start);
                    }
                }
            }
        }
    }

    // Set points to 3D
    trimesh.set_vertices(points);

    // Assign face ids
    for (Index i = 0; i < static_cast<Index>(trimesh.get_triangles().rows());
         i++) {
        auto tri = trimesh.get_triangles().row(i);
        auto p0_sph = from_cartesian_to_spherical(points.row(tri[0]));
        auto p1_sph = from_cartesian_to_spherical(points.row(tri[1]));
        auto p2_sph = from_cartesian_to_spherical(points.row(tri[2]));

        if (_end_angle - _start_angle == 2 * pi) {
            if (abs(p0_sph[0] + pi) < LENGTH_TOL) {
                if (p1_sph[0] > pi / 2 || p2_sph[0] > pi / 2) {
                    p0_sph[0] = pi;
                }
            }
            if (abs(p1_sph[0] + pi) < LENGTH_TOL) {
                if (p0_sph[0] > pi / 2 || p2_sph[0] > pi / 2) {
                    p1_sph[0] = pi;
                }
            }
            if (abs(p2_sph[0] + pi) < LENGTH_TOL) {
                if (p0_sph[0] > pi / 2 || p1_sph[0] > pi / 2) {
                    p2_sph[0] = pi;
                }
            }
        }

        auto centroid = (p0_sph + p1_sph + p2_sph) / 3.0;

        // Get the face id
        trimesh.get_face_ids()[i] = get_faceid_from_uv(thermal_mesh, centroid);
    }

    // Common operations for all meshes. TODO: create a dedicated function
    trimesh.set_surface1_color(thermal_mesh.get_side1_color().get_rgb());
    trimesh.set_surface2_color(thermal_mesh.get_side2_color().get_rgb());
    // sort the mesh
    trimesh.sort_triangles();
    // compute areas
    trimesh.compute_areas();

    return trimesh;
}

// NOLINTEND(readability-function-cognitive-complexity)

// TODO: clang-tidy: This function is too long and complex. It should be
// refactored.

// NOLINTBEGIN(readability-function-cognitive-complexity)

inline TriMesh Sphere::create_mesh2(const ThermalMesh& thermal_mesh,
                                    double tolerance) const {
    using std::numbers::pi;

    // Main directions of the 3D cone
    // Eigen::Vector3d vx = (_p3 - _p1).normalized();
    // Eigen::Vector3d vz = (_p2 - _p1).normalized();
    // Eigen::Vector3d vy = (vz.cross(vx)).normalized();

    // TODO Add checks to validate tolerance and have a minimum/maximum
    // tolerance
    const double max_length_points_dir1 = std::sqrt(
        tolerance *
        (2 * (_apex_truncation - _base_truncation) * _radius - tolerance));
    const double max_length_points_dir2 = std::sqrt(
        tolerance * (2 * (_end_angle - _start_angle) * _radius - tolerance));

    // Copy the normalized mesh vectors
    Eigen::VectorXd dir1_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir1_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir1_mesh().size()));

    Eigen::VectorXd dir2_mesh = Eigen::Map<const Eigen::VectorXd>(
        thermal_mesh.get_dir2_mesh().data(),
        static_cast<Eigen::Index>(thermal_mesh.get_dir2_mesh().size()));

    std::vector<double> dir2_mesh_normalized(
        static_cast<VectorIndex>(dir2_mesh.size()));
    for (MeshIndex i = 0; i < dir2_mesh.size(); ++i) {
        dir2_mesh_normalized[i] =
            (_start_angle + dir2_mesh[i] * (_end_angle - _start_angle)) /
            (2 * pi);
    }

    // 1. Determine the number of points to reserve space
    const auto dir1_size = static_cast<MeshIndex>(dir1_mesh.size());
    const auto dir2_size = static_cast<MeshIndex>(dir2_mesh.size());

    MeshIndex num_points = 0;

    // Dir1 is the vz direction. The number of points in each meridian is the
    // same
    std::vector<MeshIndex> additional_points_dir1(dir1_size - 1, 0);

    // Dir1 start is 0 if the base is not truncated
    const MeshIndex dir1_start = (_base_truncation != -_radius) ? 0 : 1;
    // Dir1 end is 0 if the apex is not truncated
    const MeshIndex dir1_end = (_apex_truncation != _radius) ? 0 : 1;
    // Dir2 end is 0 if the end angle is not 2*pi
    const MeshIndex dir2_end = (_end_angle - _start_angle != 2 * pi) ? 0 : 1;

    std::cout << "add2" << '\n';
    // Dir2 is the angular direction. Because the length of the circumference
    // changes for each of the divisions in dir1, a different number of points
    // might be needed
    MeshIndex num_points_row_dir1 = dir1_size;
    MeshIndex num_points_dir2 = 0;

    // Points in direction 1, the same for each radius
    if (max_length_points_dir1 > LENGTH_TOL) {
        for (MeshIndex i = 0; i < dir1_size - 1; ++i) {
            const double z1 =
                _base_truncation +
                dir1_mesh[i] * (_apex_truncation - _base_truncation);
            const double z2 =
                _base_truncation +
                dir1_mesh[i + 1] * (_apex_truncation - _base_truncation);
            const double ph1 = asin(z1 / _radius);
            const double ph2 = asin(z2 / _radius);
            const double distance = _radius * abs(ph2 - ph1);
            const auto num_additional_points = static_cast<MeshIndex>(
                std::floor(distance / max_length_points_dir1));
            num_points_row_dir1 += num_additional_points;
            additional_points_dir1[i] = num_additional_points;
        }
    }

    const bool single_node = dir1_size - (dir1_start + dir1_end) == 0;

    const MeshIndex add2_1 =
        single_node ? 1 : dir1_size - (dir1_start + dir1_end);
    const MeshIndex add2_2 = single_node ? 1 : dir2_size - 1;
    std::vector<std::vector<MeshIndex>> additional_points_dir2(
        add2_1, std::vector<MeshIndex>(add2_2));

    // if (dir1_size-(dir1_start+dir1_end) > 0) {
    // Points in direction 2, different for each radius and angle
    if (max_length_points_dir2 > LENGTH_TOL) {
        if (single_node) {
            additional_points_dir2[0][0] = static_cast<MeshIndex>(
                std::floor(2 * pi * _radius / max_length_points_dir2));
        } else {
            for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
                const double angle_i =
                    (dir2_mesh_normalized[i + 1] - dir2_mesh_normalized[i]) *
                    2 * pi;
                for (MeshIndex j = 0; j < dir1_size - (dir1_start + dir1_end);
                     ++j) {
                    const double z =
                        (j < dir1_size - 1 - (dir1_start + dir1_end))
                            ? _base_truncation +
                                  dir1_mesh[j + dir1_start] *
                                      (_apex_truncation - _base_truncation)
                            : _base_truncation +
                                  dir1_mesh[j] *
                                      (_apex_truncation - _base_truncation);

                    const double distance =
                        angle_i * _radius * sin(acos(z / _radius));
                    const auto num_additional_points = static_cast<MeshIndex>(
                        std::floor(distance / max_length_points_dir2));
                    num_points_dir2 += num_additional_points;
                    additional_points_dir2[j][i] = num_additional_points;
                }
            }
        }
    }

    for (const auto& v : additional_points_dir2) {
        for (const auto& v2 : v) {
            std::cout << v2 << " ";
        }
        std::cout << '\n';
    }
    // } else {
    //     // std::vector<std::vector<int>> additional_points_dir2(1,
    //     std::vector<int>(1)); additional_points_dir2[0][0] = 0;
    // }

    // Determine if there are interior points. For the disc
    // only the dir1 points generate interior points and only
    // they are the same for a given radius
    std::cout << "interior pts" << '\n';
    // std::vector<std::vector<MeshIndex>> interior_points_dir2(
    //     dir1_size - 1, std::vector<MeshIndex>(dir2_size - 1));
    MeshIndex num_interior_points = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        auto add_pi1 = additional_points_dir1[i_dir1];
        if (!single_node) {
            for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
                MeshIndex num_interior_points_i = 0;
                num_interior_points_i =
                    (i_dir1 == dir1_size - 2 && dir1_end == 1)
                        ? add_pi1 * additional_points_dir2[i_dir1 - 1][j_dir2]
                        : add_pi1 * additional_points_dir2[i_dir1][j_dir2];
                num_interior_points_i +=
                    (i_dir1 == dir1_size - 2 && dir1_end == 1)
                        ? (add_pi1 + 1) *
                              (additional_points_dir2[i_dir1 - 1][j_dir2] + 1)
                        : (add_pi1 + 1) *
                              (additional_points_dir2[i_dir1][j_dir2] + 1);
                num_interior_points += num_interior_points_i;
                // interior_points_dir2[i_dir1][j_dir2] = num_interior_points_i;
            }
        } else {
            num_interior_points =
                additional_points_dir2[0][0] * add_pi1 +
                (add_pi1 + 1) * (additional_points_dir2[0][0] + 1);
        }
    }

    std::cout << "total pts" << '\n';
    // Calculate the total number of points
    MeshIndex num_points_dir1 = num_points_row_dir1;
    num_points_dir1 =
        (num_points_dir1 - (dir1_start + dir1_end)) * (dir2_size - dir2_end) +
        1 * dir1_start + 1 * dir1_end;

    num_points = num_points_dir1 + num_points_dir2 + num_interior_points;
    VerticesList points(num_points, 3);
    std::vector<double> full_dir1_mesh(num_points_row_dir1);
    std::vector<double> full_dir2_mesh(num_points_dir2);
    // Fill the full mesh arrays with the original and additional mesh
    // points
    // Dir 1
    MeshIndex p_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        const double z1 =
            _base_truncation +
            dir1_mesh[i_dir1] * (_apex_truncation - _base_truncation);
        const double z2 =
            _base_truncation +
            dir1_mesh[i_dir1 + 1] * (_apex_truncation - _base_truncation);
        const double ph1 = asin(z1 / _radius);
        const double ph2 = asin(z2 / _radius);
        const double d_ph = ph2 - ph1;
        // double distance = (dir1_mesh[i_dir1 + 1] -
        // dir1_mesh[i_dir1])*(_apex_truncation-_base_truncation) /
        //                   (additional_points_dir1[i_dir1] + 1);
        if ((_base_truncation == -_radius) && (i_dir1 == 0)) {
            full_dir1_mesh[0] = _base_truncation;
            ++p_idx;
            for (MeshIndex i_add1 = 1; i_add1 <= additional_points_dir1[i_dir1];
                 ++i_add1) {
                const double phi_i =
                    ph1 + i_add1 * d_ph / (additional_points_dir1[i_dir1] + 1);
                full_dir1_mesh[p_idx] = _radius * sin(phi_i);
                ++p_idx;
            }
        } else {
            for (MeshIndex i_add1 = 0; i_add1 <= additional_points_dir1[i_dir1];
                 ++i_add1) {
                const double phi_i =
                    ph1 + i_add1 * d_ph / (additional_points_dir1[i_dir1] + 1);
                full_dir1_mesh[p_idx] = _radius * sin(phi_i);
                ++p_idx;
            }
        }
    }
    full_dir1_mesh[full_dir1_mesh.size() - 1] = _apex_truncation;

    // Dir 2
    p_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - (dir1_start + dir1_end);
         ++i_dir1) {
        // Angle of the dir2 division
        for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
            const double angle_i = (dir2_mesh_normalized[j_dir2 + 1] -
                                    dir2_mesh_normalized[j_dir2]) *
                                   2 * pi;

            // Angle between points of the dir2 division
            const double distance =
                angle_i / (additional_points_dir2[i_dir1][j_dir2] + 1);
            for (MeshIndex i_add1 = 1;
                 i_add1 <= additional_points_dir2[i_dir1][j_dir2]; ++i_add1) {
                full_dir2_mesh[p_idx] =
                    dir2_mesh_normalized[j_dir2] * 2 * pi + i_add1 * distance;
                ++p_idx;
            }
        }
    }

    std::cout << "fill pts" << '\n';
    // Fill the points array in dir1
    p_idx = 0;
    if (_base_truncation == -_radius) {
        points(0, 0) = 0.0;
        points(0, 1) = 0.0;
        points(0, 2) = -_radius;
        ++p_idx;
    }
    const auto full_dir1_mesh_size =
        static_cast<MeshIndex>(full_dir1_mesh.size());

    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - dir2_end; ++i_dir2) {
        auto angle_i = dir2_mesh_normalized[i_dir2] * 2 * pi;
        for (MeshIndex i_dir1 = dir1_start;
             i_dir1 < full_dir1_mesh_size - dir1_end; ++i_dir1) {
            const double phi = acos(full_dir1_mesh[i_dir1] / _radius);
            points(p_idx, 0) = _p1.x() + round(_radius * sin(phi) *
                                               cos(angle_i) / LENGTH_TOL) *
                                             LENGTH_TOL;
            points(p_idx, 1) = _p1.y() + round(_radius * sin(phi) *
                                               sin(angle_i) / LENGTH_TOL) *
                                             LENGTH_TOL;
            points(p_idx, 2) = _p1.z() + full_dir1_mesh[i_dir1];
            ++p_idx;
        }
    }

    const MeshIndex north_pole_idx = p_idx;
    if (_apex_truncation == _radius) {
        points(p_idx, 0) = 0.0;
        points(p_idx, 1) = 0.0;
        points(p_idx, 2) = _radius;
        ++p_idx;
    }

    std::cout << "fill pts2" << '\n';
    // Fill the points array in dir2
    MeshIndex dir2_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - (dir1_start + dir1_end);
         ++i_dir1) {
        for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
            for (MeshIndex i_add2 = 0;
                 i_add2 < additional_points_dir2[i_dir1][i_dir2]; ++i_add2) {
                const double z = _base_truncation +
                                 dir1_mesh[i_dir1 + dir1_start] *
                                     (_apex_truncation - _base_truncation);
                auto angle_i = full_dir2_mesh[dir2_idx];
                const double phi = acos(z / _radius);
                points(p_idx, 0) = _p1.x() + round(_radius * sin(phi) *
                                                   cos(angle_i) / LENGTH_TOL) *
                                                 LENGTH_TOL;
                points(p_idx, 1) = _p1.y() + round(_radius * sin(phi) *
                                                   sin(angle_i) / LENGTH_TOL) *
                                                 LENGTH_TOL;
                points(p_idx, 2) = _p1.z() + z;
                ++p_idx;
                ++dir2_idx;
            }
        }
    }

    std::cout << "fill pts3" << '\n';
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        auto add_pi1 = additional_points_dir1[i_dir1];
        MeshIndex a_i = 0;
        if (single_node) {
            a_i = 0;
        } else {
            a_i = (i_dir1 == dir1_size - 2 && dir1_end == 1)
                      ? i_dir1 - 1
                      : i_dir1 + 1 - dir1_start;
        }
        const double z1 =
            _base_truncation +
            dir1_mesh[i_dir1] * (_apex_truncation - _base_truncation);
        const double z2 =
            _base_truncation +
            dir1_mesh[i_dir1 + 1] * (_apex_truncation - _base_truncation);
        const double ph1 = asin(z1 / _radius);
        const double ph2 = asin(z2 / _radius);
        const double d_ph = ph2 - ph1;
        for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
            const double d_angle = (dir2_mesh_normalized[j_dir2 + 1] -
                                    dir2_mesh_normalized[j_dir2]);
            std::cout << "inline pts" << '\n';
            for (MeshIndex p = 1; p <= add_pi1; ++p) {
                const double phi_i =
                    ph1 + p * d_ph / (additional_points_dir1[i_dir1] + 1);
                const double z = _radius * sin(phi_i);
                // double z = _base_truncation + (dir1_mesh[i_dir1]+
                //     (dir1_mesh[i_dir1+1]-dir1_mesh[i_dir1])*p/(1+add_pi1))*(_apex_truncation-_base_truncation);
                // std::cout << "a_i: " << a_i << '\n';
                for (MeshIndex a = 1; a <= additional_points_dir2[a_i][j_dir2];
                     ++a) {
                    auto angle_a =
                        (dir2_mesh_normalized[j_dir2] +
                         d_angle / (1 + additional_points_dir2[a_i][j_dir2]) *
                             a) *
                        2 * pi;
                    const double phi = acos(z / _radius);
                    points(p_idx, 0) =
                        _p1.x() +
                        round(_radius * sin(phi) * cos(angle_a) / LENGTH_TOL) *
                            LENGTH_TOL;
                    points(p_idx, 1) =
                        _p1.y() +
                        round(_radius * sin(phi) * sin(angle_a) / LENGTH_TOL) *
                            LENGTH_TOL;
                    points(p_idx, 2) = _p1.z() + z;
                    ++p_idx;
                }
            }
            std::cout << "crossline pts" << '\n';
            for (MeshIndex p = 1; p <= add_pi1 + 1; ++p) {
                const double phi_i = ph1 + (p - 1) * d_ph / (add_pi1 + 1) +
                                     d_ph / (2 * (add_pi1 + 1));
                const double z = _radius * sin(phi_i);
                // double z = _base_truncation + (dir1_mesh[i_dir1]+
                //     (dir1_mesh[i_dir1+1]-dir1_mesh[i_dir1])*p/(2+add_pi1))*(_apex_truncation-_base_truncation);
                for (MeshIndex a = 1;
                     a <= additional_points_dir2[a_i][j_dir2] + 1; ++a) {
                    auto angle_a =
                        (dir2_mesh_normalized[j_dir2] +
                         d_angle / (1 + additional_points_dir2[a_i][j_dir2]) *
                             (a - 1) +
                         d_angle / (1 + additional_points_dir2[a_i][j_dir2]) /
                             2) *
                        2 * pi;
                    const double phi = acos(z / _radius);
                    // std::cout << "p_idx: " << p_idx << '\n';
                    points(p_idx, 0) =
                        _p1.x() +
                        round(_radius * sin(phi) * cos(angle_a) / LENGTH_TOL) *
                            LENGTH_TOL;
                    points(p_idx, 1) =
                        _p1.y() +
                        round(_radius * sin(phi) * sin(angle_a) / LENGTH_TOL) *
                            LENGTH_TOL;
                    points(p_idx, 2) = _p1.z() + z;
                    // std::cout << "p_idx: " << p_idx << '\n';
                    ++p_idx;
                }
            }
            // int num_interior_points_i =
            // add_pi1*additional_points_dir2[a_i][j_dir2]; num_interior_points
            // += num_interior_points_i; interior_points_dir2[i_dir1][j_dir2] =
            // num_interior_points_i;
        }
    }

    // 3. Create the edges

    std::cout << "edges" << '\n';
    // Reserve space for the edges
    MeshIndex edges_size =
        (dir1_size - 1) * (dir2_size) + (dir1_size) * (dir2_size - 1);
    if (_apex_truncation == _radius) {
        edges_size -= dir2_size - 1;
    }
    if (_base_truncation == -_radius) {
        edges_size -= dir2_size - 1;
    }
    if (_end_angle - _start_angle == 2 * pi) {
        edges_size -= dir1_size - 1;
    }
    EdgesList edges(edges_size);

    // Fill the edges
    // First the edges in direction 1
    MeshIndex e_idx = 0;
    p_idx = 0;
    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - dir2_end; ++i_dir2) {
        for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
            const MeshIndex num_edge_points =
                additional_points_dir1[i_dir1] + 2;

            Edges edge(num_edge_points);
            if (_base_truncation == -_radius && i_dir1 == 0) {
                edge[0] = 0;
            } else {
                edge[0] = p_idx;
            }
            for (MeshIndex i_add1 = 0; i_add1 <= additional_points_dir1[i_dir1];
                 ++i_add1) {
                edge[i_add1 + 1] = p_idx + i_add1 + 1;
            }
            if (_apex_truncation == _radius && i_dir1 == dir1_size - 2) {
                edge[num_edge_points - 1] = north_pole_idx;
            }
            edges[e_idx] = edge;
            e_idx++;
            p_idx += num_edge_points - 1;
        }
        if (_base_truncation != -_radius && _apex_truncation != _radius) {
            p_idx += 1;
        }
        if (_base_truncation == -_radius && _apex_truncation == _radius) {
            p_idx -= 1;
        }
    }

    if (_apex_truncation == _radius) {
        p_idx += 1;
    }
    if (_base_truncation == -_radius) {
        p_idx += 1;
    }

    // Then the edges in direction 2
    MeshIndex dir1_idx = 0;
    if (_base_truncation == -_radius) {
        dir1_idx = 1 + additional_points_dir1[0];
    }
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - (dir1_start + dir1_end);
         ++i_dir1) {
        MeshIndex end_idx = dir1_idx;
        for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
            const MeshIndex num_edge_points =
                additional_points_dir2[i_dir1][j_dir2] + 2;
            Edges edge(num_edge_points);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
            // GCC doesn't like additional_points_dir2[i_dir1][j_dir2]. If
            // num_edge_points where a compiled constant, it would be fine
            edge[0] = end_idx;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
            for (MeshIndex i_add2 = 0;
                 i_add2 < additional_points_dir2[i_dir1][j_dir2]; ++i_add2) {
                edge[i_add2 + 1] = p_idx;
                ++p_idx;
            }
            end_idx += num_points_dir1 / (dir2_size - dir2_end);
            if ((_end_angle - _start_angle == 2 * pi) &&
                (j_dir2 == dir2_size - dir2_end - 1)) {
                edge[num_edge_points - 1] = dir1_idx;
            } else {
                edge[num_edge_points - 1] = end_idx;
            }
            edges[e_idx] = edge;
            e_idx++;
        }
        if (i_dir1 < dir1_size - (dir1_start + dir1_end) - 1) {
            if (_base_truncation == -_radius) {
                dir1_idx += additional_points_dir1[i_dir1 + 1] + 1;
            } else {
                dir1_idx += additional_points_dir1[i_dir1] + 1;
            }
        }
    }

    // 4. Create the permiter edges. Anti-clockwise order starting from P01
    // -> dir1 edge
    MeshIndex num_perimeter_edges = dir1_size - 1;
    if (_end_angle - _start_angle != 2 * pi) {
        num_perimeter_edges += dir1_size - 1;
    }
    if (_base_truncation != -_radius) {
        num_perimeter_edges += dir2_size - 1;
    }
    if (_apex_truncation != _radius) {
        num_perimeter_edges += dir2_size - 1;
    }

    EdgesIdsList perimeter_edges(num_perimeter_edges);
    MeshIndex per_edge_idx = 0;

    // _start_angle edge
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        perimeter_edges[per_edge_idx] = i_dir1;
        ++per_edge_idx;
    }

    // _apex_truncation edge
    if (_apex_truncation != _radius) {
        for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
            perimeter_edges[per_edge_idx] =
                static_cast<MeshIndex>(edges.size()) - (dir2_size - 1) + i_dir2;
            ++per_edge_idx;
        }
    }

    // _end_angle edge
    if (_end_angle - _start_angle != 2 * pi) {
        for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
            perimeter_edges[per_edge_idx] =
                i_dir1 + (dir1_size - 1) * (dir2_size - 1);
            ++per_edge_idx;
        }
    }

    // _base_truncation edge
    if (_base_truncation != -_radius) {
        for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
            perimeter_edges[per_edge_idx] =
                (_end_angle - _start_angle != 2 * pi)
                    ? i_dir2 + dir2_size * (dir1_size - 1)
                    : i_dir2 + (dir2_size - 1) * (dir1_size - 1);
            ++per_edge_idx;
        }
    }

    // 5. Create the faces edges
    const MeshIndex num_faces = (dir1_size - 1) * (dir2_size - 1);
    FaceEdges faces_edges(num_faces);
    const MeshIndex skip_horizontal_edges =
        (dir2_size - dir2_end) * (dir1_size - 1);
    if (!single_node) {
        MeshIndex face_idx = 0;
        for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
            for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
                if (_base_truncation == -_radius && (i_dir1 == 0)) {
                    EdgesIdsList face(3);
                    face[0] = i_dir1 + (dir1_size - 1) * j_dir2;
                    face[1] = j_dir2 + skip_horizontal_edges;
                    face[2] = (_end_angle - _start_angle == 2 * pi &&
                               j_dir2 == dir2_size - 2)
                                  ? 0
                                  : i_dir1 + (dir1_size - 1) * (j_dir2 + 1);
                    faces_edges[face_idx] = face;
                } else if (_apex_truncation == _radius &&
                           i_dir1 == dir1_size - 2) {
                    EdgesIdsList face(3);
                    face[0] = i_dir1 + (dir1_size - 1) * j_dir2;
                    face[1] = (_end_angle - _start_angle == 2 * pi &&
                               j_dir2 == dir2_size - 2)
                                  ? 0
                                  : i_dir1 + (dir1_size - 1) * (j_dir2 + 1);
                    face[2] = j_dir2 + (dir2_size - 1) * (i_dir1 - dir1_start) +
                              skip_horizontal_edges;
                    faces_edges[face_idx] = face;
                } else {
                    EdgesIdsList face(4);
                    face[0] = i_dir1 + (dir1_size - 1) * j_dir2;
                    face[1] = j_dir2 +
                              (dir2_size - 1) * (i_dir1 - dir1_start + 1) +
                              skip_horizontal_edges;
                    face[2] = (_end_angle - _start_angle == 2 * pi &&
                               j_dir2 == dir2_size - 2)
                                  ? i_dir1
                                  : i_dir1 + (dir1_size - 1) * (j_dir2 + 1);
                    face[3] = j_dir2 + (dir2_size - 1) * (i_dir1 - dir1_start) +
                              skip_horizontal_edges;
                    faces_edges[face_idx] = face;
                }

                ++face_idx;
            }
        }
    } else {
        EdgesIdsList face(1);
        face[0] = 0;
        faces_edges[0] = face;
    }

    TriMesh trimesh;

    trimesh.set_faces_edges(std::move(faces_edges));
    trimesh.set_number_of_faces(num_faces * 2);

    // Transform 3D points to 2D
    auto num_points_2d = static_cast<MeshIndex>(points.rows());
    if (_end_angle - _start_angle == 2 * pi) {
        num_points_2d += num_points_row_dir1;
        if (_base_truncation == -_radius) {
            num_points_2d -= 1;
        }
        if (_apex_truncation == _radius) {
            num_points_2d -= 1;
        }
    }

    VerticesList points_2d(num_points_2d, 3);
    for (MeshIndex i = 0; i < num_points; ++i) {
        auto lon_lat = from_cartesian_to_spherical(points.row(i));
        auto p2d = from_3d_to_2d_sinusoidal(
            lon_lat, (_end_angle - _start_angle) / 2 - pi);
        points_2d(i, 0) = p2d.x();
        points_2d(i, 1) = p2d.y();
        points_2d(i, 2) = 0;
    }

    if (_end_angle - _start_angle == 2 * pi) {
        Edges edge(num_points_2d - num_points + dir1_start + dir1_end);
        if (_base_truncation == -_radius) {
            edge[0] = 0;
        }
        if (_apex_truncation == _radius) {
            edge[edge.size() - 1] = north_pole_idx;
        }
        for (MeshIndex i = 0; i < num_points_2d - num_points; ++i) {
            points_2d(num_points + i, 0) = -points_2d(i + dir1_start, 0);
            points_2d(num_points + i, 1) = points_2d(i + dir1_start, 1);
            points_2d(num_points + i, 2) = points_2d(i + dir1_start, 2);
            edge[i + dir1_start] = num_points + i;
        }
        edges.insert(edges.end(), edge);

        // Insert edges.size() - 1 to perimeter_edges
        perimeter_edges.conservativeResize(perimeter_edges.size() + 1,
                                           Eigen::NoChange);
        perimeter_edges.tail<1>()(0) = static_cast<MeshIndex>(edges.size() - 1);

        // Update the latitude edges that close the sphere
        for (MeshIndex i = 0; i < dir1_size - (dir1_start + dir1_end); ++i) {
            const MeshIndex index = (dir1_size - 1) * (dir2_size - 1) +
                                    (dir2_size - 1) * i + dir2_size - 2;
            edges[index][edges[index].size() - 1] += (num_points - dir1_start);
        }
    }

    trimesh.set_vertices(points_2d);  // points 2D are used later, don't know
                                      // why. Can't move them.
    trimesh.set_edges(std::move(edges));
    trimesh.set_perimeter_edges(std::move(perimeter_edges));

    // Triangulate the mesh
    trimesher::cdt_trimesher(trimesh);

    if (_end_angle - _start_angle == 2 * pi) {
        // Erase added edge
        trimesh.get_edges().erase(trimesh.get_edges().end() - 1,
                                  trimesh.get_edges().end());

        // Update the latitude edges that close the sphere
        for (MeshIndex i = 0; i < dir1_size - (dir1_start + dir1_end); ++i) {
            const MeshIndex index = (dir1_size - 1) * (dir2_size - 1) +
                                    (dir2_size - 1) * i + dir2_size - 2;
            trimesh.get_edges()[index][trimesh.get_edges()[index].size() - 1] -=
                (num_points - dir1_start);
        }
        // Erase added permiter edge
        trimesh.get_perimeter_edges().conservativeResize(
            trimesh.get_perimeter_edges().size() - 1, Eigen::NoChange);

        // Renumber last triangles
        for (MeshIndex i = 0; i < trimesh.get_triangles().rows(); ++i) {
            for (MeshIndex j = 0; j < trimesh.get_triangles().cols(); ++j) {
                if (trimesh.get_triangles()(i, j) >= num_points) {
                    if (trimesh.get_triangles()(i, j) == points_2d.rows() - 1 &&
                        _apex_truncation == _radius) {
                        trimesh.get_triangles()(i, j) = north_pole_idx;
                    } else {
                        trimesh.get_triangles()(i, j) -=
                            (num_points - dir1_start);
                    }
                }
            }
        }
    }

    // Set points to 3D
    trimesh.set_vertices(points);

    // Assign face ids
    for (MeshIndex i = 0; i < trimesh.get_triangles().rows(); i++) {
        auto tri = trimesh.get_triangles().row(i);
        auto p0_sph = from_cartesian_to_spherical(points.row(tri[0]));
        auto p1_sph = from_cartesian_to_spherical(points.row(tri[1]));
        auto p2_sph = from_cartesian_to_spherical(points.row(tri[2]));

        if (_end_angle - _start_angle == 2 * pi) {
            if (abs(p0_sph[0] + pi) < LENGTH_TOL) {
                if (p1_sph[0] > pi / 2 || p2_sph[0] > pi / 2) {
                    p0_sph[0] = pi;
                }
            }
            if (abs(p1_sph[0] + pi) < LENGTH_TOL) {
                if (p0_sph[0] > pi / 2 || p2_sph[0] > pi / 2) {
                    p1_sph[0] = pi;
                }
            }
            if (abs(p2_sph[0] + pi) < LENGTH_TOL) {
                if (p0_sph[0] > pi / 2 || p1_sph[0] > pi / 2) {
                    p2_sph[0] = pi;
                }
            }
        }

        auto centroid = (p0_sph + p1_sph + p2_sph) / 3.0;

        // Get the face id
        trimesh.get_face_ids()[i] = get_faceid_from_uv(thermal_mesh, centroid);
    }

    // Common operations for all meshes. TODO: create a dedicated function
    trimesh.set_surface1_color(thermal_mesh.get_side1_color().get_rgb());
    trimesh.set_surface2_color(thermal_mesh.get_side2_color().get_rgb());
    // sort the mesh
    trimesh.sort_triangles();
    // compute areas
    trimesh.compute_areas();

    return trimesh;
}

// NOLINTEND(readability-function-cognitive-complexity)

}  // namespace pycanha::gmm
