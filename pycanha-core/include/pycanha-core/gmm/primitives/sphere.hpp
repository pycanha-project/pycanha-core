#pragma once

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

class Sphere {
  public:
    Sphere(Point3D p1, Point3D p2, Point3D p3, double radius,
           double base_truncation, double apex_truncation, double start_angle,
           double end_angle) noexcept;

    [[nodiscard]] const Point3D& p1() const noexcept;
    [[nodiscard]] const Point3D& p2() const noexcept;
    [[nodiscard]] const Point3D& p3() const noexcept;
    [[nodiscard]] double radius() const noexcept;
    [[nodiscard]] double base_truncation() const noexcept;
    [[nodiscard]] double apex_truncation() const noexcept;
    [[nodiscard]] double start_angle() const noexcept;
    [[nodiscard]] double end_angle() const noexcept;

    void set_p1(Point3D p1) noexcept;
    void set_p2(Point3D p2) noexcept;
    void set_p3(Point3D p3) noexcept;
    void set_radius(double radius) noexcept;
    void set_base_truncation(double base_truncation) noexcept;
    void set_apex_truncation(double apex_truncation) noexcept;
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
    double _radius;
    double _base_truncation;
    double _apex_truncation;
    double _start_angle;
    double _end_angle;
};

}  // namespace pycanha::gmm
