#pragma once

#include <cstdint>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::gmm {

/// Exact area and area-weighted centroid of one face pair.
struct FacePairGeometry {
    double area = 0.0;
    Point3D centroid = Point3D::Zero();
};

/**
 * @brief The exact geometry of every face pair of one primitive under its
 * thermal mesh, in the primitive's own frame.
 *
 * Exact for the geometry definition, not for its triangulation: no mesh is
 * built. Face pair (i, j) is the part of the surface between cuts i and i + 1
 * of direction 1 and j and j + 1 of direction 2, in the parametrisation the
 * mesher samples.
 *
 * - Rectangle, Triangle and Quadrilateral are one planar bilinear patch, whose
 *   constant-parameter lines are straight: a face pair is a planar
 *   quadrilateral with straight edges (a triangle where it touches the
 *   triangle's apex), and its area and centroid are those of its two
 *   triangles.
 * - Disc, Cylinder, Cone, Sphere and Paraboloid are surfaces of revolution:
 *   the angular direction integrates in closed form, and so does the meridian
 *   one for all five.
 *
 * Construction is O(n1 + n2); each face pair then costs O(1), so a caller can
 * visit millions of them without storing any per-face-pair array.
 */
class FacePairGeometryEvaluator {
  public:
    FacePairGeometryEvaluator(const Primitive& primitive,
                              const ThermalMesh& thermal_mesh);

    /// False for the cutter-only primitives (Cube, TriangularPrism), which
    /// have no face pairs.
    [[nodiscard]] bool is_supported() const noexcept {
        return _kind != Kind::None;
    }

    /// Face pair (i, j), direction 1 first.
    [[nodiscard]] FacePairGeometry operator()(MeshIndex i, MeshIndex j) const;

  private:
    enum class Kind : std::uint8_t { None, Planar, Revolution };

    Kind _kind = Kind::None;

    // Planar: P(u, v) = _origin + u * _du + v * _dv + u * v * _duv, with u and
    // v the direction-1 and direction-2 cut fractions.
    Point3D _origin = Point3D::Zero();
    Vector3D _du = Vector3D::Zero();
    Vector3D _dv = Vector3D::Zero();
    Vector3D _duv = Vector3D::Zero();
    std::vector<double> _dir1_cuts;
    std::vector<double> _dir2_cuts;

    // Revolution: P = _origin + z * _axis + rho * (cos(t) _ref + sin(t)
    // _tangent). Per angular interval: its angle span and the integrals of
    // cos and sin over it; per meridian band: the integrals of rho, z * rho
    // and rho^2 over the meridian arc length.
    Vector3D _axis = Vector3D::UnitZ();
    Vector3D _ref = Vector3D::UnitX();
    Vector3D _tangent = Vector3D::UnitY();
    std::vector<double> _angle_span;
    std::vector<double> _cos_integral;
    std::vector<double> _sin_integral;
    std::vector<double> _rho_integral;
    std::vector<double> _z_rho_integral;
    std::vector<double> _rho2_integral;
};

/// Convenience for a single face pair; builds the evaluator each call.
[[nodiscard]] FacePairGeometry face_pair_geometry(
    const Primitive& primitive, const ThermalMesh& thermal_mesh, MeshIndex i,
    MeshIndex j);

}  // namespace pycanha::gmm
