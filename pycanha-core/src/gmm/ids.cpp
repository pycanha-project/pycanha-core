#include "pycanha-core/gmm/ids.hpp"

#include <atomic>
#include <cstdint>

namespace pycanha::gmm {

namespace {
// Process-wide source of GeometryId values. Starts at 1; 0 is reserved to mean
// "unregistered / unassigned".
std::atomic<std::uint64_t> g_next_geometry_id{1};
}  // namespace

GeometryId next_geometry_id() noexcept {
    return static_cast<GeometryId>(
        g_next_geometry_id.fetch_add(1, std::memory_order_relaxed));
}

}  // namespace pycanha::gmm
