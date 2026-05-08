#include "pycanha-core/gmm/scene/cut_group.hpp"

#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"

namespace pycanha::gmm {
namespace {

[[nodiscard]] bool is_valid_cutter(const Primitive& primitive) noexcept {
    return std::holds_alternative<Sphere>(primitive) ||
           std::holds_alternative<Cylinder>(primitive) ||
           std::holds_alternative<Cone>(primitive) ||
           std::holds_alternative<Cube>(primitive);
}

}  // namespace

void CutGroup::add_cutter(Primitive primitive) {
    if (!is_valid_cutter(primitive)) {
        throw std::logic_error(
            "CutGroup cutters must be closed solid primitives");
    }

    _cutters.push_back(std::move(primitive));
}

std::span<const Primitive> CutGroup::cutters() const noexcept {
    return _cutters;
}

}  // namespace pycanha::gmm
