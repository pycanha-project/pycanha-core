#include "pycanha-core/gmm/scene/geometry_group.hpp"

#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/resolve.hpp"

namespace pycanha::gmm {

GeometryGroup::GeometryGroup(std::string name,
                             std::vector<std::shared_ptr<Geometry>> children,
                             CoordinateTransformation transform)
    : Geometry(std::move(name), std::move(transform)) {
    for (auto& child : children) {
        add(std::move(child));
    }
}

void GeometryGroup::add(std::shared_ptr<Geometry> child) {
    if (child == nullptr) {
        throw std::invalid_argument("GeometryGroup::add: null child");
    }
    if (child->owning_model() != nullptr) {
        throw std::invalid_argument(
            "GeometryGroup::add: child is already registered with a model");
    }
    if (std::ranges::find(_children, child) != _children.end()) {
        throw std::invalid_argument("GeometryGroup::add: duplicate child");
    }
    _children.push_back(std::move(child));
    // Invalidates the owning model's mesh if this group is registered (no-op
    // for a standalone group, e.g. during construction).
    on_geometry_mutated();
}

bool GeometryGroup::remove_child(const std::shared_ptr<Geometry>& child) {
    const auto iterator = std::ranges::find(_children, child);
    if (iterator == _children.end()) {
        return false;
    }
    _children.erase(iterator);
    return true;
}

std::span<const std::shared_ptr<Geometry>> GeometryGroup::children()
    const noexcept {
    return _children;
}

const TriMeshD& GeometryGroup::mesh() const {
    // Not a concatenation of the children's own meshes: the subtree is
    // resolved with this group as the resolution root, so a cut group anywhere
    // below it cuts its whole target subtree in one operation.
    _walk_result = detail::resolve_subtree(*this);
    return _walk_result;
}

void GeometryGroup::create_mesh() {
    for (const auto& child : _children) {
        child->create_mesh();
    }
}

}  // namespace pycanha::gmm
