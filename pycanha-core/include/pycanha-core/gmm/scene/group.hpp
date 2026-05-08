#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::gmm {

class GeometryModel;

class Group {
  public:
    explicit Group(CoordinateTransformation transform = {});

    [[nodiscard]] const CoordinateTransformation& transform() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> child_item_indices()
        const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> child_group_indices()
        const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> child_cut_group_indices()
        const noexcept;

  private:
    friend class GeometryModel;

    CoordinateTransformation _transform;
    std::vector<std::uint32_t> _items;
    std::vector<std::uint32_t> _groups;
    std::vector<std::uint32_t> _cut_groups;
};

}  // namespace pycanha::gmm
