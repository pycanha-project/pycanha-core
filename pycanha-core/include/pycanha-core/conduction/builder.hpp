#pragma once

#include "pycanha-core/conduction/options.hpp"

namespace pycanha {
class ThermalModel;
}  // namespace pycanha

namespace pycanha::conduction {

/**
 * @brief Populates a model's tmm from its gmm: nodes and conductive couplings.
 *
 * Every active face that carries a node number contributes its
 * capacitance, its area and its centroid to that node. A face is active when
 * its side takes part in either physics, so a radiative-only side gets its
 * nodes too; a side that takes part in neither contributes nothing, even when
 * the other side of the same face pair carries the same node number.
 *
 * Conductors are conduction-only. Every pair of adjacent face pairs on a
 * conductively active side contributes an in-plane conductor, and face pairs
 * whose two sides conduct and carry different node numbers additionally get a
 * through-thickness conductor. A node fed by radiative-only faces therefore
 * exists, with capacitance and area, but with no conductor attached.
 *
 * Radiative couplings, parameters, formulas and thermal data are left
 * untouched. Geometry inside a boolean-cut group is skipped: its face-pair grid
 * no longer exists, so the parametric conduction integrals do not apply.
 * Nothing connects two different geometries — conductive interfaces need a gmm
 * entity that does not exist yet.
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
