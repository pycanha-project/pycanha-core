#pragma once

#include <cstdint>

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

// Strong uint64 wrapper. A single global atomic counter assigns a unique id to
// every gmm object (GeometryItem, GeometryGroup, GeometryGroupCutted).
//   - The counter starts at 1; the value 0 means "unregistered / unassigned".
//   - No Kind / index packing, no helpers (kind_of / index_of / to_raw /
//     make_geometry_id are all removed).
enum class GeometryId : std::uint64_t {};

// Strong MeshIndex-sized id, INTERNAL to the gmm. Even values address side 1,
// odd values address side 2. Users never construct one; the gmm computes them.
enum class FaceId : pycanha::MeshIndex {};

// NodeNum is the global pycanha::NodeNum (int32, see globals.hpp). The gmm
// previously shadowed it as int64; that shadow has been removed.

// Sentinel NodeNum meaning "no node associated" (face cut away, no thermal
// mesh, or unassigned). Distinct from any real user node number (0 is a legal
// node), and consistent with the tmm invalid-node convention (NodeNum{-1}).
inline constexpr pycanha::NodeNum NO_NODE = -1;

// Returns the next unused GeometryId from the process-wide atomic counter.
[[nodiscard]] GeometryId next_geometry_id() noexcept;

}  // namespace pycanha::gmm
