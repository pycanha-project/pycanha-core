#pragma once

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

class Cube {
  public:
    Cube(Point3D center, Vector3D extent) noexcept;

    [[nodiscard]] const Point3D& center() const noexcept;
    [[nodiscard]] const Vector3D& extent() const noexcept;

    void set_center(Point3D center) noexcept;
    void set_extent(Vector3D extent) noexcept;

    [[nodiscard]] bool is_valid() const noexcept;
    [[nodiscard]] Point2D to_uv(const Point3D& point) const;
    [[nodiscard]] Point3D to_cartesian(const Point2D& uv) const;
    [[nodiscard]] Vector3D normal_at_uv(const Point2D& uv) const noexcept;
    [[nodiscard]] double surface_area() const noexcept;

  private:
    Point3D _center;
    Vector3D _extent;
};

}  // namespace pycanha::gmm
