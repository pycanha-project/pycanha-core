#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/unified_trimesh.hpp"
#include "pycanha-core/gmm/scene/cut_group.hpp"
#include "pycanha-core/gmm/scene/group.hpp"
#include "pycanha-core/gmm/scene/item.hpp"

namespace pycanha::gmm {

class GeometryModel {
  public:
    explicit GeometryModel(std::string name);
    ~GeometryModel() = default;

    GeometryModel(const GeometryModel&) = delete;
    GeometryModel& operator=(const GeometryModel&) = delete;
    GeometryModel(GeometryModel&&) noexcept = default;
    GeometryModel& operator=(GeometryModel&&) noexcept = default;

    [[nodiscard]] const std::string& name() const noexcept;

    GeometryId add_item(std::string name, Item item,
                        const std::string& parent_name = "");
    GeometryId add_group(std::string name, Group group,
                         const std::string& parent_name = "");
    GeometryId add_cut_group(std::string name, CutGroup group,
                             const std::string& parent_name = "");
    void remove(const std::string& name);
    void rename(const std::string& current_name, std::string new_name);
    void reparent(const std::string& name, const std::string& new_parent_name);

    [[nodiscard]] bool contains(const std::string& name) const noexcept;
    [[nodiscard]] std::optional<GeometryId> id_optional(
        const std::string& name) const noexcept;
    [[nodiscard]] std::optional<std::string> name_of(GeometryId id) const;

    [[nodiscard]] const Item* item_optional(
        const std::string& name) const noexcept;
    [[nodiscard]] const Group* group_optional(
        const std::string& name) const noexcept;
    [[nodiscard]] const CutGroup* cut_group_optional(
        const std::string& name) const noexcept;
    [[nodiscard]] Item* item_optional(const std::string& name) noexcept;
    [[nodiscard]] Group* group_optional(const std::string& name) noexcept;
    [[nodiscard]] CutGroup* cut_group_optional(
        const std::string& name) noexcept;

    void set_default_mesh_options(MeshOptions mesh_options) noexcept;
    [[nodiscard]] const MeshOptions& default_mesh_options() const noexcept;

    [[nodiscard]] std::uint64_t get_structure_version() const noexcept;

    // Content mutations through mutable item/group accessors do not
    // automatically invalidate the cached unified mesh.
    [[nodiscard]] const UnifiedTriMesh& unified_mesh() const;
    void invalidate_unified_mesh() noexcept;

    void assign_face_to_node(FaceId face_id, NodeNum node_num);
    [[nodiscard]] std::optional<NodeNum> face_to_node(
        FaceId face_id) const noexcept;
    [[nodiscard]] std::span<const FaceId> faces_of_node(NodeNum node_num) const;

    [[deprecated(
        "Phase 0 placeholder; full GeometryModel API lands in later rewrite "
        "phases")]] [[nodiscard]] std::uint64_t
    structure_version() const noexcept;

  private:
    struct ParentRef {
        Kind kind = Kind::Group;
        std::uint32_t index = 0;
        bool has_value = false;
    };

    [[nodiscard]] static std::uint64_t raw_geometry_id(GeometryId id) noexcept;
    [[nodiscard]] static std::uint64_t raw_face_id(FaceId id) noexcept;

    [[nodiscard]] ParentRef resolve_parent(
        const std::string& parent_name) const;
    [[nodiscard]] bool is_active(GeometryId id) const noexcept;
    [[nodiscard]] bool is_active_group(Kind kind,
                                       std::uint32_t index) const noexcept;
    [[nodiscard]] const Group& group_ref(Kind kind, std::uint32_t index) const;
    [[nodiscard]] Group& group_ref(Kind kind, std::uint32_t index);
    [[nodiscard]] ParentRef parent_of(GeometryId id) const noexcept;
    void attach_child(ParentRef parent, GeometryId child_id);
    void detach_child(ParentRef parent, GeometryId child_id);
    void set_parent(GeometryId child_id, ParentRef parent) noexcept;
    void mark_structural_change() noexcept;
    void remove_subtree(GeometryId id);

    std::string _name;
    std::vector<Item> _items;
    std::vector<unsigned char> _item_active;
    std::vector<ParentRef> _item_parents;

    std::vector<Group> _groups;
    std::vector<unsigned char> _group_active;
    std::vector<ParentRef> _group_parents;

    std::vector<CutGroup> _cut_groups;
    std::vector<unsigned char> _cut_group_active;
    std::vector<ParentRef> _cut_group_parents;

    std::unordered_map<std::string, GeometryId> _name_to_id;
    std::unordered_map<std::uint64_t, std::string> _id_to_name;
    std::uint32_t _root_group_index = 0;

    std::uint64_t _structure_version = 0;
    MeshOptions _default_mesh_options{};

    mutable bool _unified_mesh_dirty = true;
    mutable UnifiedTriMesh _cached_unified_mesh;
    std::unordered_map<std::uint64_t, NodeNum> _face_to_node;
    mutable bool _faces_of_node_dirty = true;
    mutable std::unordered_map<NodeNum, std::vector<FaceId>>
        _cached_faces_of_node;
};

}  // namespace pycanha::gmm
