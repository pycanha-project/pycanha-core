#pragma once

#include <span>
#include <vector>

#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/scene/group.hpp"

namespace pycanha::gmm {

class CutGroup : public Group {
  public:
    using Group::Group;

    void add_cutter(Primitive primitive);
    [[nodiscard]] std::span<const Primitive> cutters() const noexcept;

  private:
    std::vector<Primitive> _cutters;
};

}  // namespace pycanha::gmm
