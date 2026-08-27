#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"

#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangular_prism.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/gmm/scene/resolve.hpp"

namespace pycanha::gmm {

bool is_closed_solid(const Primitive& primitive) noexcept {
    return std::holds_alternative<Sphere>(primitive) ||
           std::holds_alternative<Cylinder>(primitive) ||
           std::holds_alternative<Cone>(primitive) ||
           std::holds_alternative<Cube>(primitive) ||
           std::holds_alternative<TriangularPrism>(primitive);
}

namespace {

void validate_cutter(const std::shared_ptr<GeometryItem>& cutter) {
    if (cutter == nullptr) {
        throw std::invalid_argument("GeometryGroupCutted: null cutter");
    }
    if (cutter->owning_model() != nullptr) {
        throw std::invalid_argument(
            "GeometryGroupCutted: cutter is already registered with a model");
    }
    if (!is_closed_solid(cutter->primitive())) {
        throw std::invalid_argument(
            "GeometryGroupCutted: cutter primitive must be a closed solid "
            "(Sphere, Cylinder, Cone, Cube, TriangularPrism)");
    }
}

}  // namespace

GeometryGroupCutted::GeometryGroupCutted(
    std::string name, std::vector<std::shared_ptr<Geometry>> targets,
    std::vector<std::shared_ptr<GeometryItem>> cutters,
    CoordinateTransformation transform)
    : Geometry(std::move(name), std::move(transform)) {
    for (auto& target : targets) {
        if (target == nullptr) {
            throw std::invalid_argument("GeometryGroupCutted: null target");
        }
        if (target->owning_model() != nullptr) {
            throw std::invalid_argument(
                "GeometryGroupCutted: target is already registered");
        }
        _targets.push_back(target);
        _all_children.push_back(std::move(target));
    }
    for (auto& cutter : cutters) {
        cut_with(std::move(cutter));
    }
}

void GeometryGroupCutted::cut_with(std::shared_ptr<GeometryItem> cutter) {
    validate_cutter(cutter);
    if (std::ranges::find(_cutters, cutter) != _cutters.end()) {
        throw std::invalid_argument("GeometryGroupCutted: duplicate cutter");
    }
    _all_children.push_back(cutter);
    _cutters.push_back(std::move(cutter));
    on_geometry_mutated();
}

std::span<const std::shared_ptr<Geometry>> GeometryGroupCutted::targets()
    const noexcept {
    return _targets;
}

std::span<const std::shared_ptr<GeometryItem>> GeometryGroupCutted::cutters()
    const noexcept {
    return _cutters;
}

std::span<const std::shared_ptr<Geometry>> GeometryGroupCutted::children()
    const noexcept {
    return _all_children;
}

const TriMeshD& GeometryGroupCutted::mesh() const {
    // Resolved with this group as the resolution root, so its own cutters and
    // any nested ones apply to the whole target subtree at once. Targets may
    // be any Geometry; only the items inside them produce faces.
    _walk_result = detail::resolve_subtree(*this);
    return _walk_result;
}

void GeometryGroupCutted::create_mesh() {
    for (const auto& target : _targets) {
        target->create_mesh();
    }
}

}  // namespace pycanha::gmm
