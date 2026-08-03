#pragma once

#include "pycanha-core/conduction/options.hpp"

namespace pycanha {
class ThermalModel;
}  // namespace pycanha

namespace pycanha::conduction {

/**
 * @brief Populates a model's tmm from its gmm: nodes and conductive couplings.
 *
 * Every conductively active face slot that carries a node number contributes
 * its capacitance, its area and its centroid to that node, and every pair of
 * adjacent cells contributes an in-plane conductor. Face pairs whose two sides
 * carry different node numbers additionally get a through-thickness conductor.
 *
 * Radiative couplings, parameters, formulas and thermal data are left
 * untouched. Geometry inside a boolean-cut group is skipped: its cell grid no
 * longer exists, so the parametric conduction integrals do not apply. Nothing
 * connects two different geometries — conductive interfaces need a gmm entity
 * that does not exist yet.
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
