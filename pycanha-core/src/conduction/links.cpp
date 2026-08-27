#include "pycanha-core/conduction/links.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/conduction/profile.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"

namespace pycanha::conduction {

namespace {

// Conductivity-thickness product of one side: the only material quantity an
// in-plane conductance depends on. Zero means "this sheet does not conduct".
[[nodiscard]] double side_conductance_thickness(
    const gmm::ThermalMesh& thermal_mesh, unsigned side) {
    if (!thermal_mesh.is_conductive_active(side)) {
        return 0.0;
    }
    const auto& material = side == 1U ? thermal_mesh.get_side1_material()
                                      : thermal_mesh.get_side2_material();
    if (material == nullptr) {
        return 0.0;
    }
    const double thickness = side == 1U ? thermal_mesh.get_side1_thick()
                                        : thermal_mesh.get_side2_thick();
    const double conductivity = material->get_conductivity();
    if (thickness <= 0.0 || conductivity <= 0.0) {
        return 0.0;
    }
    return conductivity * thickness;
}

[[nodiscard]] double midpoint(double low, double high) noexcept {
    return 0.5 * (low + high);
}

[[nodiscard]] pycanha::MeshIndex face_pair_index(
    std::size_t dir1_idx, std::size_t dir2_idx,
    std::size_t dir1_face_pairs) noexcept {
    return to_meshidx((dir2_idx * dir1_face_pairs) + dir1_idx);
}

// Transverse extent of a direction-2 band: what the around-the-axis
// conductances of that band integrate to, the strips at different rho being
// parallel resistors.
//
// Away from the axis that is the band's potential span, which assumes the
// around-the-axis temperature difference is the same at every rho of the band.
// That assumption fails where the band reaches the axis (a disc down to r = 0,
// a cone or paraboloid apex, a sphere pole): the temperature field is analytic
// there, so the difference between two neighbouring angular face pairs vanishes
// linearly with rho instead of staying constant, and the naive integral
// diverges. Imposing the correct behaviour, dT(rho) = dT(rho_ref) * rho /
// rho_ref, cancels the 1/rho and leaves the band's meridian arc length over the
// distance of its own reference line from the axis.
[[nodiscard]] double band_transverse_extent(const MeridianProfile& profile,
                                            double low, double high) {
    const double mid = midpoint(low, high);
    if (profile.on_axis(low) || profile.on_axis(high)) {
        const double reference_rho = profile.rho(mid);
        if (reference_rho <= 0.0) {
            return 0.0;  // a band with no extent at all
        }
        return profile.meridian_length(low, high) / reference_rho;
    }
    return profile.potential(high) - profile.potential(low);
}

// Two half-resistances in series, each from a face pair's reference line to the
// shared edge, each divided by the transverse extent shared by both face pairs.
[[nodiscard]] double series_conductance(double half_a, double half_b,
                                        double conductance_thickness,
                                        double transverse_extent) {
    if (transverse_extent <= 0.0 || conductance_thickness <= 0.0) {
        return 0.0;
    }
    const double resistance =
        (half_a + half_b) / (conductance_thickness * transverse_extent);
    if (!(resistance > 0.0) || !std::isfinite(resistance)) {
        return 0.0;
    }
    return 1.0 / resistance;
}

void append_link(std::vector<FacePairLink>& links,
                 pycanha::MeshIndex face_pair_a, pycanha::MeshIndex face_pair_b,
                 unsigned side, double conductance) {
    if (conductance > 0.0 && std::isfinite(conductance)) {
        links.push_back(FacePairLink{.face_pair_a = face_pair_a,
                                     .face_pair_b = face_pair_b,
                                     .side = side,
                                     .conductance = conductance});
    }
}

void profile_links(const MeridianProfile& profile,
                   std::span<const double> dir1_cuts,
                   std::span<const double> dir2_cuts, unsigned side,
                   double conductance_thickness, const TmmBuildOptions& options,
                   std::vector<FacePairLink>& links) {
    const std::size_t dir1_face_pairs = dir1_cuts.size() - 1U;
    const std::size_t dir2_face_pairs = dir2_cuts.size() - 1U;

    // Direction 1: heat flows around the axis. The strips at different rho are
    // parallel resistors, so the band's potential span multiplies the
    // conductance and the angular distance divides it.
    for (std::size_t dir2_idx = 0; dir2_idx < dir2_face_pairs; ++dir2_idx) {
        const double band = band_transverse_extent(profile, dir2_cuts[dir2_idx],
                                                   dir2_cuts[dir2_idx + 1U]);

        const auto angular_link = [&](std::size_t cell_lo, std::size_t cell_hi,
                                      double edge_fraction) {
            const double edge = profile.dir1_coordinate(edge_fraction);
            const double ref_lo = profile.dir1_coordinate(
                midpoint(dir1_cuts[cell_lo], dir1_cuts[cell_lo + 1U]));
            const double ref_hi = profile.dir1_coordinate(
                midpoint(dir1_cuts[cell_hi], dir1_cuts[cell_hi + 1U]));
            return series_conductance(edge - ref_lo, ref_hi - edge,
                                      conductance_thickness, band);
        };

        for (std::size_t dir1_idx = 0; dir1_idx + 1U < dir1_face_pairs;
             ++dir1_idx) {
            append_link(
                links, face_pair_index(dir1_idx, dir2_idx, dir1_face_pairs),
                face_pair_index(dir1_idx + 1U, dir2_idx, dir1_face_pairs), side,
                angular_link(dir1_idx, dir1_idx + 1U,
                             dir1_cuts[dir1_idx + 1U]));
        }

        // A full revolution wraps: the seam at the start angle is an ordinary
        // interior edge, with the last face pair's half-distance measured to
        // the end of the range and the first face pair's from its start.
        if (options.close_full_revolution && profile.closes_ring() &&
            dir1_face_pairs >= 2U) {
            const double last_ref = profile.dir1_coordinate(midpoint(
                dir1_cuts[dir1_face_pairs - 1U], dir1_cuts[dir1_face_pairs]));
            const double first_ref =
                profile.dir1_coordinate(midpoint(dir1_cuts[0], dir1_cuts[1]));
            const double half_last =
                profile.dir1_coordinate(dir1_cuts[dir1_face_pairs]) - last_ref;
            const double half_first =
                first_ref - profile.dir1_coordinate(dir1_cuts[0]);
            append_link(links,
                        face_pair_index(dir1_face_pairs - 1U, dir2_idx,
                                        dir1_face_pairs),
                        face_pair_index(0U, dir2_idx, dir1_face_pairs), side,
                        series_conductance(half_last, half_first,
                                           conductance_thickness, band));
        }
    }

    // Direction 2: heat flows along the meridian. The strips are in series, so
    // the potential span divides the conductance and the face pair's own
    // angular span multiplies it.
    for (std::size_t dir1_idx = 0; dir1_idx < dir1_face_pairs; ++dir1_idx) {
        const double angular_extent =
            profile.dir1_coordinate(dir1_cuts[dir1_idx + 1U]) -
            profile.dir1_coordinate(dir1_cuts[dir1_idx]);

        for (std::size_t dir2_idx = 0; dir2_idx + 1U < dir2_face_pairs;
             ++dir2_idx) {
            const double edge = profile.potential(dir2_cuts[dir2_idx + 1U]);
            const double ref_lo = profile.potential(
                midpoint(dir2_cuts[dir2_idx], dir2_cuts[dir2_idx + 1U]));
            const double ref_hi = profile.potential(
                midpoint(dir2_cuts[dir2_idx + 1U], dir2_cuts[dir2_idx + 2U]));
            append_link(
                links, face_pair_index(dir1_idx, dir2_idx, dir1_face_pairs),
                face_pair_index(dir1_idx, dir2_idx + 1U, dir1_face_pairs), side,
                series_conductance(edge - ref_lo, ref_hi - edge,
                                   conductance_thickness, angular_extent));
        }
    }
}

// Discrete path for a planar primitive whose face grid is not a uniform
// rectangle: the triangle's fan, whose parametrisation is not orthogonal, and
// the quadrilateral's bilinear patch, whose faces change width along direction
// 2. The primitive is planar either way, so the standard finite-difference
// form applies: the shared edge carries the flow and the two reference points
// sit at their own distances from its midpoint. `point_at` maps the
// primitive's normalised (dir1, dir2) parameters to a point on it, which is
// exactly its to_cartesian.
template <typename PointFunction>
void planar_patch_links(const PointFunction& point_at,
                        std::span<const double> dir1_cuts,
                        std::span<const double> dir2_cuts, unsigned side,
                        double conductance_thickness,
                        std::vector<FacePairLink>& links) {
    const std::size_t dir1_face_pairs = dir1_cuts.size() - 1U;
    const std::size_t dir2_face_pairs = dir2_cuts.size() - 1U;

    const auto discrete_link = [&](const Point3D& edge_start,
                                   const Point3D& edge_end,
                                   const Point3D& ref_a, const Point3D& ref_b) {
        const double shared_length = (edge_end - edge_start).norm();
        const Point3D edge_mid = 0.5 * (edge_start + edge_end);
        const double distance_a = (ref_a - edge_mid).norm();
        const double distance_b = (ref_b - edge_mid).norm();
        return series_conductance(distance_a, distance_b, conductance_thickness,
                                  shared_length);
    };

    for (std::size_t dir2_idx = 0; dir2_idx < dir2_face_pairs; ++dir2_idx) {
        const double blend_ref =
            midpoint(dir2_cuts[dir2_idx], dir2_cuts[dir2_idx + 1U]);
        for (std::size_t dir1_idx = 0; dir1_idx + 1U < dir1_face_pairs;
             ++dir1_idx) {
            const double fan_edge = dir1_cuts[dir1_idx + 1U];
            append_link(
                links, face_pair_index(dir1_idx, dir2_idx, dir1_face_pairs),
                face_pair_index(dir1_idx + 1U, dir2_idx, dir1_face_pairs), side,
                discrete_link(
                    point_at(fan_edge, dir2_cuts[dir2_idx]),
                    point_at(fan_edge, dir2_cuts[dir2_idx + 1U]),
                    point_at(midpoint(dir1_cuts[dir1_idx], fan_edge),
                             blend_ref),
                    point_at(midpoint(fan_edge, dir1_cuts[dir1_idx + 2U]),
                             blend_ref)));
        }
    }

    for (std::size_t dir1_idx = 0; dir1_idx < dir1_face_pairs; ++dir1_idx) {
        const double fan_ref =
            midpoint(dir1_cuts[dir1_idx], dir1_cuts[dir1_idx + 1U]);
        for (std::size_t dir2_idx = 0; dir2_idx + 1U < dir2_face_pairs;
             ++dir2_idx) {
            const double blend_edge = dir2_cuts[dir2_idx + 1U];
            append_link(
                links, face_pair_index(dir1_idx, dir2_idx, dir1_face_pairs),
                face_pair_index(dir1_idx, dir2_idx + 1U, dir1_face_pairs), side,
                discrete_link(
                    point_at(dir1_cuts[dir1_idx], blend_edge),
                    point_at(dir1_cuts[dir1_idx + 1U], blend_edge),
                    point_at(fan_ref,
                             midpoint(dir2_cuts[dir2_idx], blend_edge)),
                    point_at(fan_ref,
                             midpoint(blend_edge, dir2_cuts[dir2_idx + 2U]))));
        }
    }
}

}  // namespace

std::vector<FacePairLink> intra_primitive_links(
    const gmm::Primitive& primitive, const gmm::ThermalMesh& thermal_mesh,
    const TmmBuildOptions& options) {
    std::vector<FacePairLink> links;
    if (!thermal_mesh.is_valid()) {
        return links;
    }

    const std::optional<MeridianProfile> profile = profile_of(primitive);
    const auto* triangle = std::get_if<gmm::Triangle>(&primitive);
    const auto* quadrilateral = std::get_if<gmm::Quadrilateral>(&primitive);
    if (!profile.has_value() && triangle == nullptr &&
        quadrilateral == nullptr) {
        return links;  // Cube: cutter-only, it never produces faces.
    }

    for (const unsigned side : {1U, 2U}) {
        const double conductance_thickness =
            side_conductance_thickness(thermal_mesh, side);
        if (conductance_thickness <= 0.0) {
            continue;
        }
        if (profile.has_value()) {
            profile_links(*profile, thermal_mesh.get_dir1_mesh(),
                          thermal_mesh.get_dir2_mesh(), side,
                          conductance_thickness, options, links);
        } else if (triangle != nullptr) {
            planar_patch_links(
                [triangle](double dir1, double dir2) {
                    return triangle->to_cartesian({dir1, dir2});
                },
                thermal_mesh.get_dir1_mesh(), thermal_mesh.get_dir2_mesh(),
                side, conductance_thickness, links);
        } else {
            planar_patch_links(
                [quadrilateral](double dir1, double dir2) {
                    return quadrilateral->to_cartesian({dir1, dir2});
                },
                thermal_mesh.get_dir1_mesh(), thermal_mesh.get_dir2_mesh(),
                side, conductance_thickness, links);
        }
    }
    return links;
}

double through_thickness_conductance(const gmm::ThermalMesh& thermal_mesh,
                                     double pair_area) {
    if (pair_area <= 0.0) {
        return 0.0;
    }
    if (!thermal_mesh.is_conductive_active(1U) ||
        !thermal_mesh.is_conductive_active(2U)) {
        return 0.0;
    }
    const auto& material_1 = thermal_mesh.get_side1_material();
    const auto& material_2 = thermal_mesh.get_side2_material();
    if (material_1 == nullptr || material_2 == nullptr) {
        return 0.0;
    }
    const double conductivity_1 = material_1->get_conductivity();
    const double conductivity_2 = material_2->get_conductivity();
    if (conductivity_1 <= 0.0 || conductivity_2 <= 0.0) {
        return 0.0;
    }
    const double resistance =
        (thermal_mesh.get_side1_thick() / conductivity_1) +
        (thermal_mesh.get_side2_thick() / conductivity_2);
    if (resistance <= 0.0) {
        return 0.0;
    }
    return pair_area / resistance;
}

}  // namespace pycanha::conduction
