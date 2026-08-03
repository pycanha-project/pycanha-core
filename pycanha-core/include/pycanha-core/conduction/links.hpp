#pragma once

#include <vector>

#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::conduction {

/**
 * @brief An in-plane conductor between two adjacent cells of one item.
 *
 * Cells are the linear indices of the ThermalMesh cell grid,
 * @c k = i + j * (n1 - 1) with direction 1 varying fastest, and the link
 * always joins the SAME side of both cells: side 1 and side 2 are two separate
 * conducting sheets, each with its own thickness and bulk material. When both
 * sides of a cell carry the same node number the two sheets end up as parallel
 * conductors of that node pair, which is exactly the "a dual-surfaced node
 * conducts through t1 + t2" rule.
 */
struct CellLink {
    pycanha::MeshIndex cell_a = 0;
    pycanha::MeshIndex cell_b = 0;
    /// 1 or 2: which of the two sheets this conductor belongs to.
    unsigned side = 1U;
    /// W/K.
    double conductance = 0.0;
};

/**
 * @brief In-plane conductors of one item's cell grid.
 *
 * Pure geometry and material: no model, no nodes, no node numbers. A side
 * contributes only when it is conductively active and carries both a bulk
 * material with non-zero conductivity and a non-zero thickness. A Triangle is
 * handled by a discrete shared-edge fallback (its fan parametrisation is not
 * orthogonal, so no closed form applies) and a Cube produces nothing at all.
 *
 * The link between two cells is two Fourier half-resistances in series, each
 * measured from its own cell's reference line to the shared edge. The
 * reference line is the midpoint of the cell's cut interval in the primitive's
 * native parameter, which is what makes the half-resistances of a chain
 * telescope into the single analytic resistance between its two ends.
 */
[[nodiscard]] std::vector<CellLink> intra_primitive_links(
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
