#include "pycanha-core/gmm/ids.hpp"

#include <atomic>
#include <cstdint>

namespace pycanha::gmm {

GeometryId next_geometry_id() noexcept {
    // Process-wide source of GeometryId values. Starts at 1; 0 is reserved to
    // mean "unregistered / unassigned". A function-local static keeps the
    // mutable counter out of namespace scope.
    static std::atomic<std::uint64_t> next_id{1};
    return static_cast<GeometryId>(
        next_id.fetch_add(1, std::memory_order_relaxed));
}

}  // namespace pycanha::gmm
