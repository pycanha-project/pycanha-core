#pragma once

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

class Disc {
  public:
    Disc(Point3D p1, Point3D p2, Point3D p3, double inner_radius,
         double outer_radius, double start_angle, double end_angle) noexcept;

    [[nodiscard]] const Point3D& p1() const noexcept;
    [[nodiscard]] const Point3D& p2() const noexcept;
    [[nodiscard]] const Point3D& p3() const noexcept;
    [[nodiscard]] double inner_radius() const noexcept;
    [[nodiscard]] double outer_radius() const noexcept;
    [[nodiscard]] double start_angle() const noexcept;
    [[nodiscard]] double end_angle() const noexcept;

    void set_p1(Point3D p1) noexcept;
    void set_p2(Point3D p2) noexcept;
    void set_p3(Point3D p3) noexcept;
    void set_inner_radius(double inner_radius) noexcept;
    void set_outer_radius(double outer_radius) noexcept;
    void set_start_angle(double start_angle) noexcept;
    void set_end_angle(double end_angle) noexcept;

    [[nodiscard]] bool is_valid() const noexcept;
    [[nodiscard]] Point2D to_uv(const Point3D& point) const;
    [[nodiscard]] Point3D to_cartesian(const Point2D& uv) const;
    [[nodiscard]] Vector3D normal_at_uv(const Point2D& uv) const noexcept;
    [[nodiscard]] double surface_area() const noexcept;

  private:
    Point3D _p1;
    Point3D _p2;
    Point3D _p3;
    double _inner_radius;
    double _outer_radius;
    double _start_angle;
    double _end_angle;
};

}  // namespace pycanha::gmm
