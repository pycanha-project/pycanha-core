#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/scene_part.hpp"

namespace pycanha::gmm {

// Object-centric scene container. HAS-A root GeometryGroup; group-style methods
// forward to it. Adds name uniqueness, lookup, mesh-option defaulting, and the
// reverse node -> face_ids index.
class GeometryModel {
  public:
    explicit GeometryModel(std::string name);
    ~GeometryModel() = default;

    GeometryModel(const GeometryModel&) = delete;
    GeometryModel& operator=(const GeometryModel&) = delete;
    GeometryModel(GeometryModel&&) noexcept = default;
    GeometryModel& operator=(GeometryModel&&) noexcept = default;

    [[nodiscard]] const std::string& name() const noexcept;

    // Registers `object` (and its whole unregistered subtree) under
    // `parent_name` (root when empty). Assigns fresh GeometryIds, sets the
    // owning model, and indexes names. Throws std::invalid_argument on a null
    // object, an already-registered node, a name clash, or an unknown / non-
    // group parent.
    void add(const std::shared_ptr<Geometry>& object,
             const std::string& parent_name = "");

    [[nodiscard]] bool contains(const std::string& name) const noexcept;
    [[nodiscard]] bool contains(
        const std::shared_ptr<Geometry>& object) const noexcept;

    [[nodiscard]] std::shared_ptr<Geometry> get(const std::string& name) const;
    [[nodiscard]] std::shared_ptr<GeometryItem> get_item(
        const std::string& name) const;
    [[nodiscard]] std::shared_ptr<GeometryGroup> get_group(
        const std::string& name) const;
    [[nodiscard]] std::shared_ptr<GeometryGroupCutted> get_cut_group(
        const std::string& name) const;

    void remove(const std::string& name);
    void remove(const std::shared_ptr<Geometry>& object);
    void rename(const std::string& current_name, std::string new_name);
    void reparent(const std::string& name, const std::string& new_parent_name);

    // Iteration (forwards to the root group).
    [[nodiscard]] std::span<const std::shared_ptr<Geometry>> children()
        const noexcept;
    [[nodiscard]] std::vector<std::shared_ptr<Geometry>> children_recursive()
        const;

    void set_default_mesh_options(MeshOptions mesh_options) noexcept;
    [[nodiscard]] const MeshOptions& default_mesh_options() const noexcept;

    [[nodiscard]] std::uint64_t get_structure_version() const noexcept;

    // Invalidation hook called by registered geometry after a content edit
    // (Option B). Bumps the structure version and marks the mesh / reverse
    // lookup dirty without rebuilding; the next mesh() read rebuilds lazily.
    void notify_content_changed() noexcept;

    // Mesh access. mesh() returns the float32 world mesh (built from the root
    // group's float64 mesh via cast). create_mesh() forces a full rebuild.
    [[nodiscard]] const TriMeshF& mesh() const;
    void create_mesh();
    void invalidate_mesh() noexcept;

    // Float64 root mesh (advanced).
    [[nodiscard]] const TriMeshD& root_group_mesh() const;
    [[nodiscard]] const std::shared_ptr<GeometryGroup>& root_group()
        const noexcept;

    // Reverse node -> face_ids lookup, rebuilt from node_numbers at mesh build.
    [[nodiscard]] std::span<const FaceId> faces_of_node(NodeNum node_num) const;

    // Splits the model mesh into rigid parts for the raytracer: one part per
    // named geometry in `split` (group, cut group or item — its whole subtree
    // becomes the part, expressed in that geometry's local frame with
    // `transform` = its world placement) plus one remainder part in the world
    // frame (identity transform, omitted when empty). Face ids stay GLOBAL:
    // the parts' face_id sets partition the unified mesh()'s. part_id is the
    // emission order (remainder first).
    //
    // Built from the same resolution walk as mesh(), so a cut group inside a
    // part is cut by exactly the cutters it would be in the model. A cutter
    // reaching across a part boundary is therefore baked into the part it cuts
    // — legitimate for a fixed configuration, wrong for an articulated one, so
    // it is logged.
    //
    // Throws std::invalid_argument on an unknown split name, a duplicate, or a
    // name that contributes no geometry.
    [[nodiscard]] std::vector<radiative::ScenePart> mesh_parts(
        std::span<const std::string> split = {}) const;

    // Builds the per-face material/activity tables from the per-side
    // ThermalMesh data (side1/side2 optical material + activity), indexed by
    // global face (even/odd = side 1/2). Materials are deduplicated by
    // object identity; faces of items without an optical material get -1
    // (logged warning, one per item). Faces absent from the mesh (gaps after
    // cuts) are -1 and inactive.
    [[nodiscard]] radiative::MaterialTable material_table() const;

  private:
    [[nodiscard]] static std::string canonicalize(const std::string& name);
    [[nodiscard]] std::shared_ptr<Geometry> find(const std::string& name) const;
    static void collect_subtree(const std::shared_ptr<Geometry>& object,
                                std::vector<std::shared_ptr<Geometry>>& out);
    void register_node(const std::shared_ptr<Geometry>& node);
    void unregister_node(const std::shared_ptr<Geometry>& node);
    void mark_structural_change() noexcept;
    void rebuild_mesh() const;
    void rebuild_faces_of_node() const;

    std::string _name;
    std::shared_ptr<GeometryGroup> _root;

    std::unordered_map<std::string, GeometryId> _name_to_id;
    std::unordered_map<std::uint64_t, std::string> _id_to_name;
    std::unordered_map<std::uint64_t, std::shared_ptr<Geometry>> _by_id;
    std::unordered_map<std::uint64_t, std::shared_ptr<GeometryGroup>>
        _parent_of;

    std::uint64_t _structure_version = 0;
    // Per-model counter seeding auto-generated "geo_<k>" names. Not atomic
    // (model mutation is single-threaded). Persisted natively; reseeded from
    // existing geo_<N> names when loading a foreign format.
    std::uint64_t _auto_name_counter = 0;
    MeshOptions _default_mesh_options{};

    mutable bool _mesh_dirty = true;
    mutable TriMeshF _cached_mesh;
    mutable bool _faces_of_node_dirty = true;
    mutable std::unordered_map<NodeNum, std::vector<FaceId>>
        _cached_faces_of_node;
};

}  // namespace pycanha::gmm
