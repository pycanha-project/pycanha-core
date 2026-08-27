#pragma once

#include <array>

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

// A solid triangular prism: the base triangle p1-p2-p3 extruded along
// p4 - p1. This is the STEP-TAS mgm_solid_triangular_prism, and it spells its
// points the way ESATAN's SHELL_TRIANGULAR_PRISM does, with the base vertices
// ordered so that (p2 - p1) x (p3 - p1) points along p4 - p1. The extrusion
// need not be perpendicular to the base.
//
// Cutter-only, exactly like Cube: it is admitted by is_closed_solid and built
// as a Manifold solid, and the UvMesher refuses it, so it never meshes,
// radiates or conducts. The SHELL form of a prism is a different object --
// three wall rectangles and no end caps -- and lives in the readers, not here.
// The two triangular bases exist only in this solid form, where they close the
// volume and are never seen.
class TriangularPrism {
  public:
    TriangularPrism(Point3D p1, Point3D p2, Point3D p3, Point3D p4) noexcept;

    [[nodiscard]] const Point3D& p1() const noexcept;
    [[nodiscard]] const Point3D& p2() const noexcept;
    [[nodiscard]] const Point3D& p3() const noexcept;
    [[nodiscard]] const Point3D& p4() const noexcept;

    void set_p1(Point3D p1) noexcept;
    void set_p2(Point3D p2) noexcept;
    void set_p3(Point3D p3) noexcept;
    void set_p4(Point3D p4) noexcept;

    /// The extrusion vector p4 - p1.
    [[nodiscard]] Vector3D height() const noexcept;

    /// The three base corners, in their defining order.
    [[nodiscard]] std::array<Point3D, 3> base() const noexcept;

    /// True when the base is a non-degenerate triangle and the extrusion
    /// leaves its plane, so the prism encloses a volume.
    [[nodiscard]] bool is_valid() const noexcept;

    /// uv follows the Cube convention: the integer part of u selects one of
    /// the five faces -- walls 0, 1 and 2 on the three base edges, then the
    /// base (3) and the top (4) -- and its fraction is the first surface
    /// parameter, with v the second. The walls span their base edge and the
    /// extrusion; the two triangular ends use the strip parametrisation a
    /// Triangle does. Normals point out of the solid.
    [[nodiscard]] Point2D to_uv(const Point3D& point) const;
    [[nodiscard]] Point3D to_cartesian(const Point2D& uv) const;
    [[nodiscard]] Vector3D normal_at_uv(const Point2D& uv) const noexcept;

    /// Total area of the closed solid: two bases plus three walls.
    [[nodiscard]] double surface_area() const noexcept;

  private:
    Point3D _p1;
    Point3D _p2;
    Point3D _p3;
    Point3D _p4;
};

}  // namespace pycanha::gmm
