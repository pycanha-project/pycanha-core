#include "pycanha-core/gmm/primitives/face_pair_geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "detail.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"

namespace pycanha::gmm {

namespace {

// The three meridian integrals of one band of a surface of revolution:
// integral of rho, z * rho and rho^2 over the band's meridian arc length.
struct BandIntegrals {
    double rho = 0.0;
    double z_rho = 0.0;
    double rho2 = 0.0;
};

// 8-point Gauss-Legendre rule on [-1, 1]: (node, weight).
constexpr std::array<std::pair<double, double>, 8> gauss_rule{{
    {-0.9602898564975363, 0.1012285362903763},
    {-0.7966664774136267, 0.2223810344533745},
    {-0.5255324099163290, 0.3137066458778873},
    {-0.1834346424956498, 0.3626837833783620},
    {0.1834346424956498, 0.3626837833783620},
    {0.5255324099163290, 0.3137066458778873},
    {0.7966664774136267, 0.2223810344533745},
    {0.9602898564975363, 0.1012285362903763},
}};

// A disc band between radii r0 and r1: rho = r, z = 0, dm = dr.
[[nodiscard]] BandIntegrals disc_band(double r0, double r1) {
    const double width = r1 - r0;
    return {.rho = width * (r0 + r1) / 2.0,
            .z_rho = 0.0,
            .rho2 = width * ((r0 * r0) + (r0 * r1) + (r1 * r1)) / 3.0};
}

// A cone (or cylinder) band between heights h0 and h1, with rho linear in h
// and dm = slant * dh. Every integrand is a polynomial of degree 2 in h, so
// the trapezoid-like closed forms and Simpson's rule are exact.
[[nodiscard]] BandIntegrals cone_band(double h0, double h1, double rho0,
                                      double rho1, double slant) {
    const double width = h1 - h0;
    const double h_mid = (h0 + h1) / 2.0;
    const double rho_mid = (rho0 + rho1) / 2.0;
    return {.rho = slant * width * rho_mid,
            .z_rho = slant * width *
                     ((h0 * rho0) + (4.0 * h_mid * rho_mid) + (h1 * rho1)) /
                     6.0,
            .rho2 = slant * width *
                    ((rho0 * rho0) + (rho0 * rho1) + (rho1 * rho1)) / 3.0};
}

// A sphere band between latitudes lat0 and lat1: rho = R cos(lat),
// z = R sin(lat), dm = R dlat. Differences are written as products so that a
// thin band keeps its relative precision.
[[nodiscard]] BandIntegrals sphere_band(double radius, double lat0,
                                        double lat1) {
    const double width = lat1 - lat0;
    const double mid = (lat0 + lat1) / 2.0;
    const double r2 = radius * radius;
    const double r3 = r2 * radius;
    const double sin_difference = 2.0 * std::cos(mid) * std::sin(width / 2.0);
    return {
        .rho = r2 * sin_difference,
        .z_rho = r3 * sin_difference * (std::sin(lat0) + std::sin(lat1)) / 2.0,
        .rho2 = r3 * ((width / 2.0) +
                      (std::cos(2.0 * mid) * std::sin(width) / 2.0))};
}

// A paraboloid band between height fractions f0 and f1. With u = sqrt(f):
// rho = R u, z = H u^2, dm = sqrt(R^2 + 4 H^2 u^2) du. The closed forms lose
// relative precision on a thin band (a difference of two nearly equal
// values). There an 8-point Gauss-Legendre rule is exact to rounding
// instead, as long as the band is narrow next to its distance from the
// integrand's complex singularities at u = +-i R / (2 H).
[[nodiscard]] BandIntegrals paraboloid_band(double radius, double height,
                                            double f0, double f1) {
    const double u0 = std::sqrt(std::max(f0, 0.0));
    const double u1 = std::sqrt(std::max(f1, 0.0));
    const double r2 = radius * radius;
    const double c2 = 4.0 * height * height;
    const auto arc = [r2, c2](double u) {
        return std::sqrt(r2 + (c2 * u * u));
    };

    const double half = (u1 - u0) / 2.0;
    const double centre = (u1 + u0) / 2.0;
    const double apex_slope = radius / (2.0 * height);
    if (half <= 0.1 * std::hypot(centre, apex_slope)) {
        BandIntegrals band;
        for (const auto& [node, node_weight] : gauss_rule) {
            const double u = centre + (half * node);
            const double weight = half * node_weight * arc(u);
            const double rho = radius * u;
            band.rho += weight * rho;
            band.z_rho += weight * height * u * u * rho;
            band.rho2 += weight * rho * rho;
        }
        return band;
    }

    const double c = 2.0 * height;
    const auto w = [r2, c2](double u) { return r2 + (c2 * u * u); };
    const auto rho_primitive = [&](double u) {
        return std::pow(w(u), 1.5) / (3.0 * c2);
    };
    const auto z_rho_primitive = [&](double u) {
        return ((std::pow(w(u), 2.5) / 5.0) -
                (r2 * std::pow(w(u), 1.5) / 3.0)) /
               (c2 * c2);
    };
    const auto rho2_primitive = [&](double u) {
        return ((u * ((2.0 * c2 * u * u) + r2) * arc(u)) / (8.0 * c2)) -
               ((r2 * r2) * std::asinh(c * u / radius) / (8.0 * c2 * c));
    };
    return {
        .rho = radius * (rho_primitive(u1) - rho_primitive(u0)),
        .z_rho = height * radius * (z_rho_primitive(u1) - z_rho_primitive(u0)),
        .rho2 = r2 * (rho2_primitive(u1) - rho2_primitive(u0))};
}

[[nodiscard]] double lerp(double start, double end, double fraction) {
    return start + ((end - start) * fraction);
}

}  // namespace

FacePairGeometryEvaluator::FacePairGeometryEvaluator(
    const Primitive& primitive, const ThermalMesh& thermal_mesh)
    : _dir1_cuts(thermal_mesh.get_dir1_mesh().begin(),
                 thermal_mesh.get_dir1_mesh().end()),
      _dir2_cuts(thermal_mesh.get_dir2_mesh().begin(),
                 thermal_mesh.get_dir2_mesh().end()) {
    // Angular intervals of a surface of revolution spanning [start, end].
    const auto set_angles = [this](double start, double end) {
        const std::size_t count = _dir1_cuts.size() - 1U;
        _angle_span.resize(count);
        _cos_integral.resize(count);
        _sin_integral.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            const double angle0 = lerp(start, end, _dir1_cuts[i]);
            const double angle1 = lerp(start, end, _dir1_cuts[i + 1U]);
            const double span = angle1 - angle0;
            const double mid = (angle0 + angle1) / 2.0;
            const double half_chord = 2.0 * std::sin(span / 2.0);
            _angle_span[i] = span;
            _cos_integral[i] = std::cos(mid) * half_chord;
            _sin_integral[i] = std::sin(mid) * half_chord;
        }
    };
    const auto set_bands = [this](const auto& band_of) {
        const std::size_t count = _dir2_cuts.size() - 1U;
        _rho_integral.resize(count);
        _z_rho_integral.resize(count);
        _rho2_integral.resize(count);
        for (std::size_t j = 0; j < count; ++j) {
            const BandIntegrals band =
                band_of(_dir2_cuts[j], _dir2_cuts[j + 1U]);
            _rho_integral[j] = band.rho;
            _z_rho_integral[j] = band.z_rho;
            _rho2_integral[j] = band.rho2;
        }
    };
    const auto set_axis_frame = [this](const Point3D& p1, const Point3D& p2,
                                       const Point3D& p3) {
        _origin = p1;
        _axis = detail::axis_direction(p1, p2);
        _ref = detail::radial_reference(p1, p3, _axis);
        _tangent = detail::tangential_direction(_axis, _ref);
    };

    std::visit(
        [&](const auto& concrete) {
            using T = std::decay_t<decltype(concrete)>;
            if constexpr (std::is_same_v<T, Rectangle>) {
                _kind = Kind::Planar;
                _origin = concrete.p1();
                _du = concrete.p2() - concrete.p1();
                _dv = concrete.p3() - concrete.p1();
            } else if constexpr (std::is_same_v<T, Quadrilateral>) {
                _kind = Kind::Planar;
                _origin = concrete.p1();
                _du = concrete.p2() - concrete.p1();
                _dv = concrete.p4() - concrete.p1();
                _duv = concrete.p1() - concrete.p2() + concrete.p3() -
                       concrete.p4();
            } else if constexpr (std::is_same_v<T, Triangle>) {
                // p1 + u * ((1 - v) * e1 + v * e2) = p1 + u e1 + u v (e2 - e1)
                _kind = Kind::Planar;
                _origin = concrete.p1();
                _du = concrete.p2() - concrete.p1();
                _duv = concrete.p3() - concrete.p2();
            } else if constexpr (std::is_same_v<T, Disc>) {
                _kind = Kind::Revolution;
                set_axis_frame(concrete.p1(), concrete.p2(), concrete.p3());
                set_angles(concrete.start_angle(), concrete.end_angle());
                set_bands([&concrete](double f0, double f1) {
                    return disc_band(lerp(concrete.inner_radius(),
                                          concrete.outer_radius(), f0),
                                     lerp(concrete.inner_radius(),
                                          concrete.outer_radius(), f1));
                });
            } else if constexpr (std::is_same_v<T, Cylinder>) {
                _kind = Kind::Revolution;
                set_axis_frame(concrete.p1(), concrete.p2(), concrete.p3());
                set_angles(concrete.start_angle(), concrete.end_angle());
                const double height = (concrete.p2() - concrete.p1()).norm();
                set_bands([&concrete, height](double f0, double f1) {
                    return cone_band(f0 * height, f1 * height,
                                     concrete.radius(), concrete.radius(), 1.0);
                });
            } else if constexpr (std::is_same_v<T, Cone>) {
                _kind = Kind::Revolution;
                set_axis_frame(concrete.p1(), concrete.p2(), concrete.p3());
                set_angles(concrete.start_angle(), concrete.end_angle());
                const double height = (concrete.p2() - concrete.p1()).norm();
                const double slope =
                    height > LENGTH_TOL
                        ? (concrete.radius2() - concrete.radius1()) / height
                        : 0.0;
                const double slant = std::sqrt(1.0 + (slope * slope));
                set_bands([&concrete, height, slant](double f0, double f1) {
                    return cone_band(
                        f0 * height, f1 * height,
                        lerp(concrete.radius1(), concrete.radius2(), f0),
                        lerp(concrete.radius1(), concrete.radius2(), f1),
                        slant);
                });
            } else if constexpr (std::is_same_v<T, Sphere>) {
                _kind = Kind::Revolution;
                const auto frame = detail::make_sphere_frame(
                    concrete.p1(), concrete.p2(), concrete.p3());
                _origin = concrete.p1();
                _axis = frame.axis;
                _ref = frame.ref;
                _tangent = frame.tangent;
                set_angles(concrete.start_angle(), concrete.end_angle());
                const double min_latitude =
                    std::asin(concrete.base_truncation() / concrete.radius());
                const double max_latitude =
                    std::asin(concrete.apex_truncation() / concrete.radius());
                set_bands([&concrete, min_latitude, max_latitude](double f0,
                                                                  double f1) {
                    return sphere_band(concrete.radius(),
                                       lerp(min_latitude, max_latitude, f0),
                                       lerp(min_latitude, max_latitude, f1));
                });
            } else if constexpr (std::is_same_v<T, Paraboloid>) {
                _kind = Kind::Revolution;
                set_axis_frame(concrete.p1(), concrete.p2(), concrete.p3());
                set_angles(concrete.start_angle(), concrete.end_angle());
                const double height = (concrete.p2() - concrete.p1()).norm();
                set_bands([&concrete, height](double f0, double f1) {
                    return paraboloid_band(concrete.radius(), height, f0, f1);
                });
            } else {
                // Cube and TriangularPrism are cutter-only: no face pairs.
                _kind = Kind::None;
            }
        },
        primitive);
}

FacePairGeometry FacePairGeometryEvaluator::operator()(MeshIndex i,
                                                       MeshIndex j) const {
    FacePairGeometry geometry;
    if (_kind == Kind::Planar) {
        const auto point = [this](double u, double v) -> Point3D {
            return _origin + (u * _du) + (v * _dv) + ((u * v) * _duv);
        };
        const double u0 = _dir1_cuts[i];
        const double u1 = _dir1_cuts[i + 1U];
        const double v0 = _dir2_cuts[j];
        const double v1 = _dir2_cuts[j + 1U];
        const Point3D p00 = point(u0, v0);
        const Point3D p10 = point(u1, v0);
        const Point3D p11 = point(u1, v1);
        const Point3D p01 = point(u0, v1);
        // The mesher's two triangles of the face pair.
        const double area_1 = 0.5 * (p10 - p00).cross(p11 - p00).norm();
        const double area_2 = 0.5 * (p11 - p00).cross(p01 - p00).norm();
        geometry.area = area_1 + area_2;
        if (geometry.area > 0.0) {
            geometry.centroid =
                ((area_1 * (p00 + p10 + p11)) + (area_2 * (p00 + p11 + p01))) /
                (3.0 * geometry.area);
        } else {
            geometry.centroid = p00;
        }
    } else if (_kind == Kind::Revolution) {
        geometry.area = _angle_span[i] * _rho_integral[j];
        if (geometry.area > 0.0) {
            const Vector3D moment =
                ((_angle_span[i] * _z_rho_integral[j]) * _axis) +
                ((_cos_integral[i] * _rho2_integral[j]) * _ref) +
                ((_sin_integral[i] * _rho2_integral[j]) * _tangent);
            geometry.centroid = _origin + (moment / geometry.area);
        } else {
            geometry.centroid = _origin;
        }
    }
    return geometry;
}

FacePairGeometry face_pair_geometry(const Primitive& primitive,
                                    const ThermalMesh& thermal_mesh,
                                    MeshIndex i, MeshIndex j) {
    return FacePairGeometryEvaluator(primitive, thermal_mesh)(i, j);
}

}  // namespace pycanha::gmm
