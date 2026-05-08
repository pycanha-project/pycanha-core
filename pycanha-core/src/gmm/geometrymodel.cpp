#include "pycanha-core/gmm/geometrymodel.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/scene/cut_group.hpp"
#include "pycanha-core/gmm/scene/item.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha::gmm {
namespace {

[[nodiscard]] std::string canonicalize_name(std::string_view name) {
    std::string canonical(name.begin(), name.end());
    std::transform(
        canonical.begin(), canonical.end(), canonical.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return canonical;
}

template <typename T>
void erase_index(std::vector<T>& values, T target) {
    values.erase(std::remove(values.begin(), values.end(), target),
                 values.end());
}

template <typename ActiveFlags>
[[nodiscard]] bool is_active_index(const ActiveFlags& flags,
                                   std::uint32_t index) noexcept {
    return index < flags.size() && (flags[index] != 0U);
}

template <typename GroupLike>
void push_active_children(std::vector<GeometryId>& stack,
                          const GroupLike& group,
                          const std::vector<unsigned char>& item_active,
                          const std::vector<unsigned char>& group_active,
                          const std::vector<unsigned char>& cut_group_active) {
    for (const std::uint32_t child : group.child_item_indices()) {
        if (is_active_index(item_active, child)) {
            stack.push_back(make_geometry_id(Kind::Item, child));
        }
    }
    for (const std::uint32_t child : group.child_group_indices()) {
        if (is_active_index(group_active, child)) {
            stack.push_back(make_geometry_id(Kind::Group, child));
        }
    }
    for (const std::uint32_t child : group.child_cut_group_indices()) {
        if (is_active_index(cut_group_active, child)) {
            stack.push_back(make_geometry_id(Kind::CutGroup, child));
        }
    }
}

void erase_name_mapping(
    std::unordered_map<std::string, GeometryId>& name_to_id,
    std::unordered_map<std::uint64_t, std::string>& id_to_name, GeometryId id) {
    const auto name_iterator = id_to_name.find(to_raw(id));
    if (name_iterator == id_to_name.end()) {
        return;
    }

    name_to_id.erase(canonicalize_name(name_iterator->second));
    id_to_name.erase(name_iterator);
}

}  // namespace

GeometryModel::GeometryModel(std::string name)
    : _name(std::move(name)),
      _groups{Group{}},
      _group_active{1U},
      _group_parents{ParentRef{Kind::Group, 0U, false}} {}

std::uint64_t GeometryModel::raw_geometry_id(GeometryId id) noexcept {
    return to_raw(id);
}

std::uint64_t GeometryModel::raw_face_id(FaceId id) noexcept {
    return to_raw(id);
}

const std::string& GeometryModel::name() const noexcept { return _name; }

GeometryModel::ParentRef GeometryModel::resolve_parent(
    const std::string& parent_name) const {
    if (parent_name.empty()) {
        return ParentRef{Kind::Group, _root_group_index, true};
    }

    const auto id = id_optional(parent_name);
    if (!id.has_value()) {
        throw std::invalid_argument("Unknown parent group '" + parent_name +
                                    "'");
    }

    const Kind kind = kind_of(*id);
    if ((kind != Kind::Group) && (kind != Kind::CutGroup)) {
        throw std::invalid_argument("Parent '" + parent_name +
                                    "' is not a group");
    }

    return ParentRef{kind, index_of(*id), true};
}

bool GeometryModel::is_active(GeometryId id) const noexcept {
    const std::uint32_t index = index_of(id);
    switch (kind_of(id)) {
        case Kind::Item:
            return index < _item_active.size() && (_item_active[index] != 0U);
        case Kind::Group:
            return index < _group_active.size() && (_group_active[index] != 0U);
        case Kind::CutGroup:
            return index < _cut_group_active.size() &&
                   (_cut_group_active[index] != 0U);
    }

    return false;
}

bool GeometryModel::is_active_group(Kind kind,
                                    std::uint32_t index) const noexcept {
    if (kind == Kind::Group) {
        return index < _group_active.size() && (_group_active[index] != 0U);
    }

    return index < _cut_group_active.size() && (_cut_group_active[index] != 0U);
}

const Group& GeometryModel::group_ref(Kind kind, std::uint32_t index) const {
    if (kind == Kind::Group) {
        return _groups.at(index);
    }

    return _cut_groups.at(index);
}

Group& GeometryModel::group_ref(Kind kind, std::uint32_t index) {
    if (kind == Kind::Group) {
        return _groups.at(index);
    }

    return _cut_groups.at(index);
}

GeometryModel::ParentRef GeometryModel::parent_of(
    GeometryId id) const noexcept {
    const std::uint32_t index = index_of(id);
    switch (kind_of(id)) {
        case Kind::Item:
            return index < _item_parents.size() ? _item_parents[index]
                                                : ParentRef{};
        case Kind::Group:
            return index < _group_parents.size() ? _group_parents[index]
                                                 : ParentRef{};
        case Kind::CutGroup:
            return index < _cut_group_parents.size() ? _cut_group_parents[index]
                                                     : ParentRef{};
    }

    return {};
}

void GeometryModel::attach_child(ParentRef parent, GeometryId child_id) {
    PYCANHA_ASSERT(parent.has_value, "Cannot attach child to an empty parent");

    Group& parent_group = group_ref(parent.kind, parent.index);
    switch (kind_of(child_id)) {
        case Kind::Item:
            parent_group._items.push_back(index_of(child_id));
            break;
        case Kind::Group:
            parent_group._groups.push_back(index_of(child_id));
            break;
        case Kind::CutGroup:
            parent_group._cut_groups.push_back(index_of(child_id));
            break;
    }
}

void GeometryModel::detach_child(ParentRef parent, GeometryId child_id) {
    if (!parent.has_value) {
        return;
    }

    Group& parent_group = group_ref(parent.kind, parent.index);
    switch (kind_of(child_id)) {
        case Kind::Item:
            erase_index(parent_group._items, index_of(child_id));
            break;
        case Kind::Group:
            erase_index(parent_group._groups, index_of(child_id));
            break;
        case Kind::CutGroup:
            erase_index(parent_group._cut_groups, index_of(child_id));
            break;
    }
}

void GeometryModel::set_parent(GeometryId child_id, ParentRef parent) noexcept {
    const std::uint32_t index = index_of(child_id);
    switch (kind_of(child_id)) {
        case Kind::Item:
            _item_parents[index] = parent;
            break;
        case Kind::Group:
            _group_parents[index] = parent;
            break;
        case Kind::CutGroup:
            _cut_group_parents[index] = parent;
            break;
    }
}

void GeometryModel::mark_structural_change() noexcept {
    ++_structure_version;
    _unified_mesh_dirty = true;
    _faces_of_node_dirty = true;
}

GeometryId GeometryModel::add_item(std::string name, Item item,
                                   const std::string& parent_name) {
    const auto canonical_name = canonicalize_name(name);
    if (_name_to_id.contains(canonical_name)) {
        throw std::invalid_argument("Geometry '" + name + "' already exists");
    }

    const ParentRef parent = resolve_parent(parent_name);
    const auto index = static_cast<std::uint32_t>(_items.size());
    const GeometryId id = make_geometry_id(Kind::Item, index);
    _items.push_back(std::move(item));
    _item_active.push_back(1U);
    _item_parents.push_back(parent);
    _name_to_id.emplace(canonical_name, id);
    _id_to_name.emplace(raw_geometry_id(id), std::move(name));
    attach_child(parent, id);
    mark_structural_change();
    return id;
}

GeometryId GeometryModel::add_group(std::string name, Group group,
                                    const std::string& parent_name) {
    const auto canonical_name = canonicalize_name(name);
    if (_name_to_id.contains(canonical_name)) {
        throw std::invalid_argument("Geometry '" + name + "' already exists");
    }

    const ParentRef parent = resolve_parent(parent_name);
    const auto index = static_cast<std::uint32_t>(_groups.size());
    const GeometryId id = make_geometry_id(Kind::Group, index);
    _groups.push_back(std::move(group));
    _group_active.push_back(1U);
    _group_parents.push_back(parent);
    _name_to_id.emplace(canonical_name, id);
    _id_to_name.emplace(raw_geometry_id(id), std::move(name));
    attach_child(parent, id);
    mark_structural_change();
    return id;
}

GeometryId GeometryModel::add_cut_group(std::string name, CutGroup group,
                                        const std::string& parent_name) {
    const auto canonical_name = canonicalize_name(name);
    if (_name_to_id.contains(canonical_name)) {
        throw std::invalid_argument("Geometry '" + name + "' already exists");
    }

    const ParentRef parent = resolve_parent(parent_name);
    const auto index = static_cast<std::uint32_t>(_cut_groups.size());
    const GeometryId id = make_geometry_id(Kind::CutGroup, index);
    _cut_groups.push_back(std::move(group));
    _cut_group_active.push_back(1U);
    _cut_group_parents.push_back(parent);
    _name_to_id.emplace(canonical_name, id);
    _id_to_name.emplace(raw_geometry_id(id), std::move(name));
    attach_child(parent, id);
    mark_structural_change();
    return id;
}

void GeometryModel::remove_subtree(GeometryId id) {
    std::vector<GeometryId> stack{id};
    while (!stack.empty()) {
        const GeometryId current = stack.back();
        stack.pop_back();
        const std::uint32_t index = index_of(current);

        switch (kind_of(current)) {
            case Kind::Item: {
                if (is_active_index(_item_active, index)) {
                    _item_active[index] = 0U;
                }
                break;
            }
            case Kind::Group: {
                if (!is_active_index(_group_active, index)) {
                    break;
                }
                push_active_children(stack, _groups[index], _item_active,
                                     _group_active, _cut_group_active);
                _group_active[index] = 0U;
                break;
            }
            case Kind::CutGroup: {
                if (!is_active_index(_cut_group_active, index)) {
                    break;
                }
                push_active_children(stack, _cut_groups[index], _item_active,
                                     _group_active, _cut_group_active);
                _cut_group_active[index] = 0U;
                break;
            }
        }

        erase_name_mapping(_name_to_id, _id_to_name, current);
    }
}

void GeometryModel::remove(const std::string& name) {
    const auto id = id_optional(name);
    if (!id.has_value()) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "Geometry '{}' doesn't exist",
                           name);
        return;
    }

    detach_child(parent_of(*id), *id);
    remove_subtree(*id);
    mark_structural_change();
}

void GeometryModel::rename(const std::string& current_name,
                           std::string new_name) {
    const auto id = id_optional(current_name);
    if (!id.has_value()) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "Geometry '{}' doesn't exist",
                           current_name);
        return;
    }

    const auto current_canonical = canonicalize_name(current_name);
    const auto new_canonical = canonicalize_name(new_name);
    if ((new_canonical != current_canonical) &&
        _name_to_id.contains(new_canonical)) {
        throw std::invalid_argument("Geometry '" + new_name +
                                    "' already exists");
    }

    _name_to_id.erase(current_canonical);
    _name_to_id.emplace(new_canonical, *id);
    _id_to_name[raw_geometry_id(*id)] = std::move(new_name);
    mark_structural_change();
}

void GeometryModel::reparent(const std::string& name,
                             const std::string& new_parent_name) {
    const auto id = id_optional(name);
    if (!id.has_value()) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "Geometry '{}' doesn't exist",
                           name);
        return;
    }

    const ParentRef new_parent = resolve_parent(new_parent_name);
    const Kind object_kind = kind_of(*id);
    if ((object_kind == Kind::Group) || (object_kind == Kind::CutGroup)) {
        GeometryId ancestor =
            make_geometry_id(new_parent.kind, new_parent.index);
        while (is_active_group(kind_of(ancestor), index_of(ancestor))) {
            if (ancestor == *id) {
                throw std::logic_error("Reparent would create a cycle");
            }

            const ParentRef ancestor_parent = parent_of(ancestor);
            if (!ancestor_parent.has_value) {
                break;
            }

            ancestor =
                make_geometry_id(ancestor_parent.kind, ancestor_parent.index);
        }
    }

    const ParentRef current_parent = parent_of(*id);
    detach_child(current_parent, *id);
    attach_child(new_parent, *id);
    set_parent(*id, new_parent);
    mark_structural_change();
}

bool GeometryModel::contains(const std::string& name) const noexcept {
    return _name_to_id.contains(canonicalize_name(name));
}

std::optional<GeometryId> GeometryModel::id_optional(
    const std::string& name) const noexcept {
    const auto iterator = _name_to_id.find(canonicalize_name(name));
    if (iterator == _name_to_id.end()) {
        return std::nullopt;
    }

    return iterator->second;
}

std::optional<std::string> GeometryModel::name_of(GeometryId id) const {
    if (!is_active(id)) {
        return std::nullopt;
    }

    const auto iterator = _id_to_name.find(raw_geometry_id(id));
    if (iterator == _id_to_name.end()) {
        return std::nullopt;
    }

    return iterator->second;
}

const Item* GeometryModel::item_optional(
    const std::string& name) const noexcept {
    const auto id = id_optional(name);
    if (!id.has_value() || (kind_of(*id) != Kind::Item) || !is_active(*id)) {
        return nullptr;
    }

    return &_items[index_of(*id)];
}

const Group* GeometryModel::group_optional(
    const std::string& name) const noexcept {
    const auto id = id_optional(name);
    if (!id.has_value() || (kind_of(*id) != Kind::Group) || !is_active(*id)) {
        return nullptr;
    }

    return &_groups[index_of(*id)];
}

const CutGroup* GeometryModel::cut_group_optional(
    const std::string& name) const noexcept {
    const auto id = id_optional(name);
    if (!id.has_value() || (kind_of(*id) != Kind::CutGroup) ||
        !is_active(*id)) {
        return nullptr;
    }

    return &_cut_groups[index_of(*id)];
}

Item* GeometryModel::item_optional(const std::string& name) noexcept {
    const auto id = id_optional(name);
    if (!id.has_value() || (kind_of(*id) != Kind::Item) || !is_active(*id)) {
        return nullptr;
    }

    return &_items[index_of(*id)];
}

Group* GeometryModel::group_optional(const std::string& name) noexcept {
    const auto id = id_optional(name);
    if (!id.has_value() || (kind_of(*id) != Kind::Group) || !is_active(*id)) {
        return nullptr;
    }

    return &_groups[index_of(*id)];
}

CutGroup* GeometryModel::cut_group_optional(const std::string& name) noexcept {
    const auto id = id_optional(name);
    if (!id.has_value() || (kind_of(*id) != Kind::CutGroup) ||
        !is_active(*id)) {
        return nullptr;
    }

    return &_cut_groups[index_of(*id)];
}

void GeometryModel::set_default_mesh_options(
    MeshOptions mesh_options) noexcept {
    _default_mesh_options = mesh_options;
    _unified_mesh_dirty = true;
}

const MeshOptions& GeometryModel::default_mesh_options() const noexcept {
    return _default_mesh_options;
}

std::uint64_t GeometryModel::get_structure_version() const noexcept {
    return _structure_version;
}

void GeometryModel::invalidate_unified_mesh() noexcept {
    _unified_mesh_dirty = true;
}

void GeometryModel::assign_face_to_node(FaceId face_id, NodeNum node_num) {
    _face_to_node[raw_face_id(face_id)] = node_num;
    _faces_of_node_dirty = true;
}

std::optional<NodeNum> GeometryModel::face_to_node(
    FaceId face_id) const noexcept {
    const auto iterator = _face_to_node.find(raw_face_id(face_id));
    if (iterator == _face_to_node.end()) {
        return std::nullopt;
    }

    return iterator->second;
}

std::span<const FaceId> GeometryModel::faces_of_node(NodeNum node_num) const {
    if (_faces_of_node_dirty) {
        _cached_faces_of_node.clear();
        for (const auto& [raw_face_id_value, node] : _face_to_node) {
            _cached_faces_of_node[node].push_back(
                static_cast<FaceId>(raw_face_id_value));
        }
        _faces_of_node_dirty = false;
    }

    const auto iterator = _cached_faces_of_node.find(node_num);
    if (iterator == _cached_faces_of_node.end()) {
        return {};
    }

    return iterator->second;
}

std::uint64_t GeometryModel::structure_version() const noexcept {
    return get_structure_version();
}

}  // namespace pycanha::gmm
