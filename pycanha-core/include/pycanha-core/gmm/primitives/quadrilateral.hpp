#pragma once

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

class Quadrilateral {
  public:
    Quadrilateral(Point3D p1, Point3D p2, Point3D p3, Point3D p4) noexcept;

    [[nodiscard]] const Point3D& p1() const noexcept;
    [[nodiscard]] const Point3D& p2() const noexcept;
    [[nodiscard]] const Point3D& p3() const noexcept;
    [[nodiscard]] const Point3D& p4() const noexcept;

    void set_p1(Point3D p1) noexcept;
    void set_p2(Point3D p2) noexcept;
    void set_p3(Point3D p3) noexcept;
    void set_p4(Point3D p4) noexcept;

    [[nodiscard]] bool is_valid() const noexcept;
    [[nodiscard]] Point2D to_uv(const Point3D& point) const;
    [[nodiscard]] Point3D to_cartesian(const Point2D& uv) const;
    [[nodiscard]] Vector3D normal_at_uv(const Point2D& uv) const noexcept;
    [[nodiscard]] double surface_area() const noexcept;

  private:
    Point3D _p1;
    Point3D _p2;
    Point3D _p3;
    Point3D _p4;
};

}  // namespace pycanha::gmm
