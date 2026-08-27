#pragma once

#include <cstdint>
#include <optional>

#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::conduction {

/**
 * @brief The two scalar maps one-dimensional conduction needs from a primitive.
 *
 * Every non-planar pycanha primitive is a surface of revolution: mesh direction
 * 1 is an angle and direction 2 runs along the meridian at a distance @c rho
 * from the axis. Writing @c dl2 for the meridional arc element, the whole
 * conduction model is expressed with
 *
 *   dir1_coordinate(f)  the coordinate heat flows along in direction 1,
 *                       measured so that a physical arc at distance rho is
 *                       rho * delta(dir1_coordinate) long. For a surface of
 *                       revolution that is the angle in radians; a planar
 *                       primitive is the degenerate case rho == 1, where the
 *                       coordinate is simply the in-plane length.
 *   potential(f)        Phi = integral of dl2 / rho along the meridian.
 *
 * Both directions are ratios of those two maps. Around the axis the strips at
 * different rho sit side by side, so their conductances add and the direction-1
 * conductance is proportional to the potential span of the band. Along the
 * meridian the strips sit end to end, so their resistances add and the
 * direction-2 conductance is inversely proportional to the potential span.
 *
 * @c f is always a cut fraction in [0, 1] of the primitive's own native
 * parameter, which is what the ThermalMesh cut vectors store.
 */
class MeridianProfile {
  public:
    enum class Kind : std::uint8_t {
        Planar,
        Disc,
        Cylinder,
        Cone,
        Sphere,
        Paraboloid,
    };

    [[nodiscard]] static MeridianProfile make_planar(double dir1_length,
                                                     double dir2_length);
    [[nodiscard]] static MeridianProfile make_disc(double angle_span,
                                                   bool closes_ring,
                                                   double inner_radius,
                                                   double outer_radius);
    [[nodiscard]] static MeridianProfile make_cylinder(double angle_span,
                                                       bool closes_ring,
                                                       double radius,
                                                       double height);
    [[nodiscard]] static MeridianProfile make_cone(double angle_span,
                                                   bool closes_ring,
                                                   double radius1,
                                                   double radius2,
                                                   double height);
    [[nodiscard]] static MeridianProfile make_sphere(double angle_span,
                                                     bool closes_ring,
                                                     double radius,
                                                     double min_latitude,
                                                     double max_latitude);
    [[nodiscard]] static MeridianProfile make_paraboloid(double angle_span,
                                                         bool closes_ring,
                                                         double radius,
                                                         double height);

    [[nodiscard]] Kind kind() const noexcept { return _kind; }

    /// True when direction 1 spans a full revolution, so the last angular face
    /// pair is adjacent to the first one.
    [[nodiscard]] bool closes_ring() const noexcept { return _closes_ring; }

    /// Coordinate along direction 1 at cut fraction @p fraction: radians for a
    /// surface of revolution, metres for a planar primitive.
    [[nodiscard]] double dir1_coordinate(double fraction) const noexcept;

    /// Total direction-1 extent, i.e. dir1_coordinate(1) - dir1_coordinate(0).
    [[nodiscard]] double dir1_extent() const noexcept { return _dir1_extent; }

    /// Distance from the axis of revolution at cut fraction @p fraction.
    /// Identically one for a planar primitive, where it is dimensionless.
    [[nodiscard]] double rho(double fraction) const noexcept;

    /// True where @c rho vanishes: the potential is unbounded there and the
    /// caller must integrate the direction-1 band from its reference line.
    [[nodiscard]] bool on_axis(double fraction) const noexcept;

    /// Phi at cut fraction @p fraction, up to an additive constant that cancels
    /// in every difference the conduction model takes. Unbounded on the axis.
    [[nodiscard]] double potential(double fraction) const noexcept;

    /// Meridian arc length between two cut fractions. This is what the
    /// around-the-axis conductance of a band needs once the band reaches the
    /// axis, where the potential itself is unbounded.
    [[nodiscard]] double meridian_length(double low,
                                         double high) const noexcept;

  private:
    Kind _kind = Kind::Planar;
    bool _closes_ring = false;
    double _dir1_extent = 1.0;
    // Native direction-2 parameter at the two ends of the cut range: the two
    // radii of a disc, 0 and the height of a cylinder or cone, the two
    // latitudes of a sphere, the in-plane length of a planar primitive.
    double _dir2_start = 0.0;
    double _dir2_end = 1.0;
    double _radius = 0.0;
    double _radius_end = 0.0;
    double _height = 0.0;
    // sqrt(1 + (dr/dh)^2): the meridian arc element per unit of the native
    // direction-2 parameter, constant for a cone and one everywhere else.
    double _slant = 1.0;
};

/// The conduction profile of @p primitive, or std::nullopt when it has no
/// closed form: a Triangle (its fan parametrisation is not orthogonal), a
/// Quadrilateral (a bilinear patch, so its faces vary in width along direction
/// 2 and no constant-width planar profile represents it) -- both handled by
/// the discrete shared-edge path -- or a Cube, which is cutter-only and never
/// meshes.
[[nodiscard]] std::optional<MeridianProfile> profile_of(
    const gmm::Primitive& primitive);

}  // namespace pycanha::conduction
