#pragma once

#include <vector>

#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::conduction {

/**
 * @brief An in-plane conductor between two adjacent face pairs of one item.
 *
 * Face pairs are the linear indices of the ThermalMesh face-pair grid,
 * @c k = i + j * (n1 - 1) with direction 1 varying fastest, and the link
 * always joins the SAME side of both face pairs: side 1 and side 2 are two
 * separate conducting sheets, each with its own thickness and bulk material.
 * When both sides of a face pair carry the same node number the two sheets end
 * up as parallel conductors of that node pair, which is exactly the
 * "a dual-surfaced node conducts through t1 + t2" rule.
 */
struct FacePairLink {
    pycanha::MeshIndex face_pair_a = 0;
    pycanha::MeshIndex face_pair_b = 0;
    /// 1 or 2: which of the two sheets this conductor belongs to.
    unsigned side = 1U;
    /// W/K.
    double conductance = 0.0;
};

/**
 * @brief In-plane conductors of one item's face-pair grid.
 *
 * Pure geometry and material: no model, no nodes, no node numbers. A side
 * contributes only when it is conductively active and carries both a bulk
 * material with non-zero conductivity and a non-zero thickness. A Triangle and
 * a Quadrilateral go through the discrete shared-edge path -- the fan
 * parametrisation is not orthogonal and the bilinear patch's face pairs vary
 * in width, so no closed form applies to either -- while a Cube and a
 * TriangularPrism are cutter-only and produce nothing at all.
 *
 * The link between two face pairs is two Fourier half-resistances in series,
 * each measured from its own face pair's reference line to the shared edge.
 * The reference line is the midpoint of the face pair's cut interval in the
 * primitive's native parameter, which is what makes the half-resistances of a
 * chain telescope into the single analytic resistance between its two ends.
 */
[[nodiscard]] std::vector<FacePairLink> intra_primitive_links(
    const gmm::Primitive& primitive, const gmm::ThermalMesh& thermal_mesh,
    const TmmBuildOptions& options);

/**
 * @brief Conductance through the thickness of one face pair of @p pair_area.
 *
 * The two half-slabs are in series: A / (t1/k1 + t2/k2). Returns zero when a
 * side is conductively inactive, has no bulk material, or has zero thickness
 * or conductivity — such a pair simply produces no conductor.
 */
[[nodiscard]] double through_thickness_conductance(
    const gmm::ThermalMesh& thermal_mesh, double pair_area);

}  // namespace pycanha::conduction
