#pragma once

#include "pycanha-core/conduction/options.hpp"

namespace pycanha {
class ThermalModel;
}  // namespace pycanha

namespace pycanha::conduction {

/**
 * @brief Populates a model's tmm from its gmm: nodes and conductive couplings.
 *
 * Every item builds its own network part from its definition alone (see
 * build_network_part): exact thermal capacities from the face pairs' exact
 * areas, the parametric in-plane conductances and the through-thickness ones.
 * No triangulation is made for an item that is not cut. The parts are then
 * merged and written with the bulk calls (commit_network_parts).
 *
 * Every active face that carries a node number contributes its capacitance
 * and its centroid to that node. A face is active when its side takes part in
 * either physics, so a radiative-only side gets its nodes too; a side that
 * takes part in neither contributes nothing, even when the other side of the
 * same face pair carries the same node number.
 *
 * Conductors are conduction-only. Every pair of adjacent face pairs on a
 * conductively active side contributes an in-plane conductor, and face pairs
 * whose two sides conduct and carry different node numbers additionally get a
 * through-thickness conductor. A node fed by radiative-only faces therefore
 * exists, with capacitance, but with no conductor attached.
 *
 * An item inside a boolean-cut group gets its nodes too: each face pair's
 * capacity and through-thickness conductance are scaled by the fraction of its
 * area that survives the cut, read off the cut triangulation, and the in-plane
 * conductors touching a cut face pair are removed and reported (a correction
 * model for them is future work). The node area `a` is not set: it is the
 * triangulated area, see assign_node_areas. Radiative couplings, parameters,
 * formulas and thermal data are left untouched. Nothing connects two
 * different geometries -- conductive interfaces need a gmm entity that does
 * not exist yet.
 *
 * The whole node and coupling set is assembled before anything is written, so
 * a throwing build leaves the tmm untouched.
 *
 * @throws std::invalid_argument if the tmm already holds nodes or conductive
 *         couplings. There is no merge semantics and no provenance tracking.
 */
[[nodiscard]] TmmBuildReport build_tmm_from_gmm(
    ThermalModel& model, const TmmBuildOptions& options = {});

}  // namespace pycanha::conduction
