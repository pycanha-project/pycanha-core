#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/tmm/bulk.hpp"

namespace pycanha {
class ThermalMathematicalModel;
class ThermalModel;
}  // namespace pycanha

namespace pycanha::conduction {

/**
 * @brief One geometry item's contribution to the conduction network.
 *
 * Built from the item's definition alone: exact thermal capacities (the
 * face pairs' exact areas, never a triangulation) and the same in-plane
 * conductances as intra_primitive_links, plus the through-thickness ones.
 * Nothing in it depends on other items, so parts are built independently and
 * merged by commit_network_parts.
 *
 * Nodes are sorted by number and unique. Couplings join two nodes of the
 * part by their position in node_numbers (index_1 < index_2), sorted by
 * (index_1, index_2) and unique: exactly the order the coupling matrices
 * store, so a part is committed without translating or sorting anything.
 */
struct NetworkPart {
    std::vector<NodeNum> node_numbers;
    /// [J/K]
    std::vector<double> thermal_capacity;
    /// Area-weighted centroid of each node's faces, in the root frame.
    std::vector<double> position_x;
    std::vector<double> position_y;
    std::vector<double> position_z;
    /// The weight behind each position: the node's (surviving) face area
    /// over its active sides. Parts that share a node combine their
    /// positions with it.
    std::vector<double> position_weight;

    std::vector<std::int32_t> coupling_index_1;
    std::vector<std::int32_t> coupling_index_2;
    /// [W/K]
    std::vector<double> conductance;

    /// Which sides fed each node (bit 0: side 1, bit 1: side 2), with each
    /// side's bulk material: what tells a node that gathers two different
    /// materials, within the item or across parts.
    std::vector<std::uint8_t> node_sides;
    std::array<const gmm::BulkMaterial*, 2> side_bulk{nullptr, nullptr};

    /// This item's share of the build report: whether it was processed,
    /// its diagnostics, the face-pair links computed, and the cut accounting.
    TmmBuildReport report;
};

/**
 * @brief The network part of one item.
 *
 * @p to_root places the item (its own frame to the root frame); it moves the
 * positions only, areas and conductances do not change under a rigid
 * transform.
 *
 * An item cut by other geometry passes, per face pair (direction 1 fastest),
 * the fraction of its area that survives the cut and the area-weighted
 * centroid of what survives, in the root frame; both empty for an uncut item.
 * A face pair's capacity and through-thickness conductance are scaled by its
 * fraction; a fraction at or below 1e-9 means the face pair is gone and
 * contributes nothing (a node left with nothing is not created). In-plane
 * conductances touching a face pair that is cut at all are dropped and
 * counted: a cut face pair has no parametric neighbour to integrate to.
 *
 * @throws std::invalid_argument if a non-empty fraction or centroid span does
 *         not have one entry per face pair.
 */
[[nodiscard]] NetworkPart build_network_part(
    const gmm::GeometryItem& item, const gmm::CoordinateTransformation& to_root,
    const TmmBuildOptions& options = {},
    std::span<const double> surviving_fraction = {},
    std::span<const Vector3D> surviving_centroid = {});

/**
 * @brief Merges network parts and writes them into an empty tmm.
 *
 * Parts whose node numbers do not interleave are appended one after the
 * other, straight from their arrays, with the bulk calls. Parts that share or
 * interleave node numbers are first merged: capacities and conductances are
 * summed and positions combined with their weights. Every node is diffusive,
 * at options.initial_temperature; conductances at or below
 * options.min_conductance are dropped. The node area is not set, see
 * assign_node_areas. The parts' diagnostics are logged here, in part order.
 *
 * @throws std::invalid_argument if the tmm already holds nodes or conductive
 *         couplings.
 */
[[nodiscard]] TmmBuildReport commit_network_parts(
    ThermalMathematicalModel& tmm, std::span<const NetworkPart> parts,
    const TmmBuildOptions& options = {});

/// The same, for parts held elsewhere (Python objects, say), so that none has
/// to be copied into one array first.
[[nodiscard]] TmmBuildReport commit_network_parts(
    ThermalMathematicalModel& tmm, std::span<const NetworkPart* const> parts,
    const TmmBuildOptions& options = {});

/**
 * @brief Sets the node area `a` from the model's triangulation.
 *
 * The area of every triangulated face on an active side, summed per node:
 * each item's cached triangulation, or its cut one. A node of the gmm that is
 * not in the tmm is skipped and reported. This is the one place the
 * conduction path triangulates; build_tmm_from_gmm never sets `a`.
 */
BulkReport assign_node_areas(ThermalModel& model);

}  // namespace pycanha::conduction
