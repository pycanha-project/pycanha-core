#pragma once
#include <utility>

#include "../parameters.hpp"

using namespace pycanha;  // NOLINT

namespace pycanha::gmm {

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
class Quadrilateral {
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
    bool is_valid() const;

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
    void set_p1(Point3D p1) { _p1 = std::move(p1); }

    /**
     * @brief Updates the second vertex of the quadrilateral.
     * @param p2 The new second vertex of the quadrilateral.
     */
    void set_p2(Point3D p2) { _p2 = std::move(p2); }

    /**
     * @brief Updates the third vertex of the quadrilateral.
     * @param p3 The new third vertex of the quadrilateral.
     */
    void set_p3(Point3D p3) { _p3 = std::move(p3); }

    /**
     * @brief Updates the fourth vertex of the quadrilateral.
     * @param p4 The new fourth vertex of the quadrilateral.
     */
    void set_p4(Point3D p4) { _p4 = std::move(p4); }

  private:
    Point3D _p1;
    Point3D _p2;
    Point3D _p3;
    Point3D _p4;
};

bool Quadrilateral::is_valid() const {
    // Compute and store the vectors v1 and v2
    Vector3D v1(this->v1());
    Vector3D v2(this->v2());

    // Check if v1 or v2 is too small
    if (v1.norm() <= LENGTH_TOL || v2.norm() <= LENGTH_TOL) return false;

    // Compute the dot product of the normalized vectors v1 and v2
    double dot = v1.dot(v2) / (v1.norm() * v2.norm());

    // Check if the angle between v1 and v2 is too close to 0 or 180 degrees
    if (dot <= (-1.0 + ANGLE_TOL) || dot >= (1.0 - ANGLE_TOL)) return false;

    // Compute the normal vector n via cross product
    Vector3D n = v1.cross(v2);
    double nlen = n.norm();

    // Check if the length of n is too small
    if (nlen <= LENGTH_TOL) return false;

    // Normalize n
    n = n / nlen;

    // Check if _p3 is in the plane spanned by _p1, _p2, and _p4
    Vector3D m(_p3 - _p1);
    if (std::abs(m.dot(n)) > LENGTH_TOL) return false;

    // Verify correct interior angles at points _p2, _p3, and _p4
    for (int i = 2; i <= 4; ++i) {
        Point3D curr, prev, next;
        switch (i) {
            case 2:
                curr = _p2;
                prev = _p3;
                next = _p1;
                break;
            case 3:
                curr = _p3;
                prev = _p4;
                next = _p2;
                break;
            case 4:
                curr = _p4;
                prev = _p1;
                next = _p3;
                break;
        }
        v1 = prev - curr;
        v2 = next - curr;

        if (v1.norm() <= LENGTH_TOL || v2.norm() <= LENGTH_TOL) return false;

        dot = v1.dot(v2) / (v1.norm() * v2.norm());

        if (dot <= (-1.0 + ANGLE_TOL) || dot >= (1.0 - ANGLE_TOL)) return false;
    }

    return true;
}

}  // namespace pycanha::gmm
