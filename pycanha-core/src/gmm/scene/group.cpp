#include "pycanha-core/gmm/scene/group.hpp"

#include <cstdint>
#include <span>
#include <utility>

#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::gmm {

Group::Group(CoordinateTransformation transform)
    : _transform(std::move(transform)) {}

const CoordinateTransformation& Group::transform() const noexcept {
    return _transform;
}

std::span<const std::uint32_t> Group::child_item_indices() const noexcept {
    return _items;
}

std::span<const std::uint32_t> Group::child_group_indices() const noexcept {
    return _groups;
}

std::span<const std::uint32_t> Group::child_cut_group_indices() const noexcept {
    return _cut_groups;
}

}  // namespace pycanha::gmm
