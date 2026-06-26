#include "pycanha-core/gmm/scene/geometry_group.hpp"

#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/scene_mesh_detail.hpp"

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
    if (child->id() != GeometryId{0}) {
        throw std::invalid_argument(
            "GeometryGroup::add: child is already registered with a model");
    }
    if (std::ranges::find(_children, child) != _children.end()) {
        throw std::invalid_argument("GeometryGroup::add: duplicate child");
    }
    _children.push_back(std::move(child));
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
    _walk_result = TriMeshD{};
    // Concatenate every child mesh into _walk_result, threading the running
    // face-id offset through concatenate_offset.
    pycanha::MeshIndex offset = 0;
    for (const auto& child : _children) {
        detail::concatenate_offset(_walk_result, child->mesh(), offset);
    }
    detail::apply_transform_in_place(_walk_result, _transform);
    return _walk_result;
}

void GeometryGroup::create_mesh() {
    for (const auto& child : _children) {
        child->create_mesh();
    }
}

}  // namespace pycanha::gmm
