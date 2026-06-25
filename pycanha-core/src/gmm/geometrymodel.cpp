#include "pycanha-core/gmm/geometrymodel.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha::gmm {

namespace {
[[nodiscard]] std::uint64_t raw(GeometryId id) noexcept {
    return static_cast<std::uint64_t>(id);
}
}  // namespace

GeometryModel::GeometryModel(std::string name)
    : _name(std::move(name)), _root(std::make_shared<GeometryGroup>("")) {
    _root->_owning_model = this;
}

const std::string& GeometryModel::name() const noexcept { return _name; }

std::string GeometryModel::canonicalize(const std::string& name) {
    std::string canonical = name;
    std::ranges::transform(canonical, canonical.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return canonical;
}

std::shared_ptr<Geometry> GeometryModel::find(const std::string& name) const {
    const auto id_it = _name_to_id.find(canonicalize(name));
    if (id_it == _name_to_id.end()) {
        return nullptr;
    }
    const auto node_it = _by_id.find(raw(id_it->second));
    return node_it == _by_id.end() ? nullptr : node_it->second;
}

void GeometryModel::collect_subtree(
    const std::shared_ptr<Geometry>& object,
    std::vector<std::shared_ptr<Geometry>>& out) const {
    out.push_back(object);
    for (const auto& child : object->children()) {
        collect_subtree(child, out);
    }
}

void GeometryModel::add(std::shared_ptr<Geometry> object,
                        const std::string& parent_name) {
    if (object == nullptr) {
        throw std::invalid_argument("GeometryModel::add: null object");
    }

    std::shared_ptr<GeometryGroup> parent = _root;
    if (!parent_name.empty()) {
        parent = std::dynamic_pointer_cast<GeometryGroup>(find(parent_name));
        if (parent == nullptr) {
            throw std::invalid_argument("GeometryModel::add: parent '" +
                                        parent_name +
                                        "' is not a registered group");
        }
    }

    std::vector<std::shared_ptr<Geometry>> nodes;
    collect_subtree(object, nodes);

    for (const auto& node : nodes) {
        if (node->id() != GeometryId{0}) {
            throw std::invalid_argument(
                "GeometryModel::add: a node is already registered");
        }
    }

    std::unordered_set<std::string> batch_names;
    for (const auto& node : nodes) {
        if (node->name().empty()) {
            continue;
        }
        const std::string canonical = canonicalize(node->name());
        if (_name_to_id.contains(canonical) ||
            !batch_names.insert(canonical).second) {
            throw std::invalid_argument("GeometryModel::add: name '" +
                                        node->name() + "' already exists");
        }
    }

    parent->_children.push_back(object);

    // Recursive registration with parent-group tracking.
    struct Frame {
        std::shared_ptr<Geometry> node;
        std::shared_ptr<GeometryGroup> parent_group;
    };
    std::vector<Frame> stack{{.node = object, .parent_group = parent}};
    while (!stack.empty()) {
        const Frame frame = stack.back();
        stack.pop_back();

        const GeometryId id = next_geometry_id();
        frame.node->_id = id;
        frame.node->_owning_model = this;
        if (frame.node->_name.empty()) {
            frame.node->_name = "geometry_" + std::to_string(raw(id));
        }
        _name_to_id.emplace(canonicalize(frame.node->_name), id);
        _id_to_name.emplace(raw(id), frame.node->_name);
        _by_id.emplace(raw(id), frame.node);
        _parent_of.emplace(raw(id), frame.parent_group);

        const auto as_group =
            std::dynamic_pointer_cast<GeometryGroup>(frame.node);
        for (const auto& child : frame.node->children()) {
            stack.push_back({.node = child, .parent_group = as_group});
        }
    }

    mark_structural_change();
}

bool GeometryModel::contains(const std::string& name) const noexcept {
    return _name_to_id.contains(canonicalize(name));
}

bool GeometryModel::contains(
    const std::shared_ptr<Geometry>& object) const noexcept {
    if (object == nullptr || object->id() == GeometryId{0}) {
        return false;
    }
    const auto it = _by_id.find(raw(object->id()));
    return it != _by_id.end() && it->second == object;
}

std::shared_ptr<Geometry> GeometryModel::get(const std::string& name) const {
    return find(name);
}

std::shared_ptr<GeometryItem> GeometryModel::get_item(
    const std::string& name) const {
    return std::dynamic_pointer_cast<GeometryItem>(find(name));
}

std::shared_ptr<GeometryGroup> GeometryModel::get_group(
    const std::string& name) const {
    return std::dynamic_pointer_cast<GeometryGroup>(find(name));
}

std::shared_ptr<GeometryGroupCutted> GeometryModel::get_cut_group(
    const std::string& name) const {
    return std::dynamic_pointer_cast<GeometryGroupCutted>(find(name));
}

void GeometryModel::unregister_node(const std::shared_ptr<Geometry>& node) {
    const std::uint64_t key = raw(node->id());
    _name_to_id.erase(canonicalize(node->name()));
    _id_to_name.erase(key);
    _by_id.erase(key);
    _parent_of.erase(key);
    node->_id = GeometryId{0};
    node->_owning_model = nullptr;
}

void GeometryModel::remove(const std::string& name) {
    const auto node = find(name);
    if (node == nullptr) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "Geometry '{}' doesn't exist",
                           name);
        return;
    }
    remove(node);
}

void GeometryModel::remove(const std::shared_ptr<Geometry>& object) {
    if (!contains(object)) {
        return;
    }

    const auto parent_it = _parent_of.find(raw(object->id()));
    if (parent_it != _parent_of.end() && parent_it->second != nullptr) {
        parent_it->second->remove_child(object);
    } else {
        _root->remove_child(object);
    }

    std::vector<std::shared_ptr<Geometry>> nodes;
    collect_subtree(object, nodes);
    for (const auto& node : nodes) {
        if (node->id() != GeometryId{0}) {
            unregister_node(node);
        }
    }

    mark_structural_change();
}

void GeometryModel::rename(const std::string& current_name,
                           std::string new_name) {
    const auto node = find(current_name);
    if (node == nullptr) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "Geometry '{}' doesn't exist",
                           current_name);
        return;
    }

    const std::string current_canonical = canonicalize(current_name);
    const std::string new_canonical = canonicalize(new_name);
    if (new_canonical != current_canonical &&
        _name_to_id.contains(new_canonical)) {
        throw std::invalid_argument("Geometry '" + new_name +
                                    "' already exists");
    }

    const GeometryId id = node->id();
    _name_to_id.erase(current_canonical);
    _name_to_id.emplace(new_canonical, id);
    _id_to_name[raw(id)] = new_name;
    node->_name = std::move(new_name);
    mark_structural_change();
}

void GeometryModel::reparent(const std::string& name,
                             const std::string& new_parent_name) {
    const auto node = find(name);
    if (node == nullptr) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "Geometry '{}' doesn't exist",
                           name);
        return;
    }

    std::shared_ptr<GeometryGroup> new_parent = _root;
    if (!new_parent_name.empty()) {
        new_parent =
            std::dynamic_pointer_cast<GeometryGroup>(find(new_parent_name));
        if (new_parent == nullptr) {
            throw std::invalid_argument("Parent '" + new_parent_name +
                                        "' is not a group");
        }
    }

    // Cycle check: new_parent must not be `node` or a descendant of it.
    for (std::shared_ptr<Geometry> ancestor = new_parent;
         ancestor != nullptr;) {
        if (ancestor == node) {
            throw std::logic_error("Reparent would create a cycle");
        }
        const auto it = _parent_of.find(raw(ancestor->id()));
        ancestor = (it == _parent_of.end()) ? nullptr : it->second;
    }

    const auto parent_it = _parent_of.find(raw(node->id()));
    if (parent_it != _parent_of.end() && parent_it->second != nullptr) {
        parent_it->second->remove_child(node);
    } else {
        _root->remove_child(node);
    }
    new_parent->_children.push_back(node);
    _parent_of[raw(node->id())] = new_parent;
    mark_structural_change();
}

std::span<const std::shared_ptr<Geometry>> GeometryModel::children()
    const noexcept {
    return _root->children();
}

std::vector<std::shared_ptr<Geometry>> GeometryModel::children_recursive()
    const {
    std::vector<std::shared_ptr<Geometry>> out;
    for (const auto& child : _root->children()) {
        collect_subtree(child, out);
    }
    return out;
}

void GeometryModel::set_default_mesh_options(
    MeshOptions mesh_options) noexcept {
    _default_mesh_options = mesh_options;
    _mesh_dirty = true;
    _faces_of_node_dirty = true;
}

const MeshOptions& GeometryModel::default_mesh_options() const noexcept {
    return _default_mesh_options;
}

std::uint64_t GeometryModel::get_structure_version() const noexcept {
    return _structure_version;
}

void GeometryModel::mark_structural_change() noexcept {
    ++_structure_version;
    _mesh_dirty = true;
    _faces_of_node_dirty = true;
}

void GeometryModel::rebuild_mesh() const {
    _cached_mesh = _root->mesh().cast<float>();
    _mesh_dirty = false;
    _faces_of_node_dirty = true;
}

const TriMeshF& GeometryModel::mesh() const {
    if (_mesh_dirty) {
        rebuild_mesh();
    }
    return _cached_mesh;
}

void GeometryModel::create_mesh() {
    _root->create_mesh();
    rebuild_mesh();
}

void GeometryModel::invalidate_mesh() noexcept {
    _mesh_dirty = true;
    _faces_of_node_dirty = true;
}

const TriMeshD& GeometryModel::root_group_mesh() const { return _root->mesh(); }

const std::shared_ptr<GeometryGroup>& GeometryModel::root_group()
    const noexcept {
    return _root;
}

void GeometryModel::rebuild_faces_of_node() const {
    if (_mesh_dirty) {
        rebuild_mesh();
    }
    _cached_faces_of_node.clear();
    const auto& node_numbers = _cached_mesh.node_numbers;
    for (Eigen::Index slot = 0; slot < node_numbers.rows(); ++slot) {
        const NodeNum node_num = node_numbers(slot);
        if (node_num != 0) {
            _cached_faces_of_node[node_num].push_back(
                static_cast<FaceId>(static_cast<pycanha::MeshIndex>(slot)));
        }
    }
    _faces_of_node_dirty = false;
}

std::span<const FaceId> GeometryModel::faces_of_node(NodeNum node_num) const {
    if (_faces_of_node_dirty) {
        rebuild_faces_of_node();
    }
    const auto it = _cached_faces_of_node.find(node_num);
    if (it == _cached_faces_of_node.end()) {
        return {};
    }
    return it->second;
}

}  // namespace pycanha::gmm
