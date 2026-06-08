#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/ops/boundary_edges.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/mesh/unified_trimesh.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/cut_group.hpp"
#include "pycanha-core/gmm/scene/group.hpp"
#include "pycanha-core/gmm/scene/item.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::CutGroup;
using pycanha::gmm::Cylinder;
using pycanha::gmm::FaceId;
using pycanha::gmm::GeometryId;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::Group;
using pycanha::gmm::Item;
using pycanha::gmm::Kind;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriMesh;
using pycanha::gmm::UnifiedTriMesh;
namespace mesh_ops = pycanha::gmm::mesh::ops;

struct SceneIds {
    GeometryId panel_id;
    GeometryId tube_id;
};

[[nodiscard]] Item make_panel_item() {
    return Item(Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
                ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 1.0}});
}

[[nodiscard]] Item make_tube_item() {
    return Item(Cylinder({0.0, 0.0, -0.75}, {0.0, 0.0, 0.75},
                         {0.35, 0.0, -0.75}, 0.35, 0.0, 2.0 * std::numbers::pi),
                ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}},
                CoordinateTransformation::from_translation({0.0, 2.0, 0.0}));
}

// Local replacement for the removed ThermalMesh::face_id(i, j, Side): even
// id = side 1 (front), odd = side 2 (back). Side parity is now an internal
// convention.
[[nodiscard]] FaceId tm_face_id(const ThermalMesh& thermal_mesh, std::size_t i,
                                std::size_t j, unsigned side) {
    const std::size_t num_dir2_cells = thermal_mesh.get_dir2_mesh().size() - 1U;
    const std::size_t linear_index = i * num_dir2_cells + j;
    return static_cast<FaceId>(2U * static_cast<std::uint64_t>(linear_index) +
                               (side == 2U ? 1U : 0U));
}

[[nodiscard]] std::unordered_set<std::uint64_t> face_id_set(
    const ThermalMesh& thermal_mesh) {
    std::unordered_set<std::uint64_t> ids;
    for (std::size_t i = 0; i < thermal_mesh.get_dir1_mesh().size() - 1U; ++i) {
        for (std::size_t j = 0; j < thermal_mesh.get_dir2_mesh().size() - 1U;
             ++j) {
            ids.insert(pycanha::gmm::to_raw(tm_face_id(thermal_mesh, i, j, 1U)));
            ids.insert(pycanha::gmm::to_raw(tm_face_id(thermal_mesh, i, j, 2U)));
        }
    }
    return ids;
}

[[nodiscard]] double area_for_geometry(const UnifiedTriMesh& mesh,
                                       GeometryId geometry_id) {
    const auto areas = mesh_ops::compute_areas(
        TriMesh{mesh.vertices, mesh.triangles, mesh.face_ids});
    double area_sum = 0.0;
    const std::uint64_t raw_geometry_id = pycanha::gmm::to_raw(geometry_id);
    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        if (mesh.geometry_ids[tri_idx] == raw_geometry_id) {
            area_sum += areas[tri_idx];
        }
    }
    return area_sum;
}

[[nodiscard]] TriMesh subset_for_geometry(const UnifiedTriMesh& mesh,
                                          GeometryId geometry_id) {
    const std::uint64_t raw_geometry_id = pycanha::gmm::to_raw(geometry_id);
    std::vector<Eigen::Index> triangle_indices;
    triangle_indices.reserve(static_cast<std::size_t>(mesh.triangles.rows()));
    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        if (mesh.geometry_ids[tri_idx] == raw_geometry_id) {
            triangle_indices.push_back(tri_idx);
        }
    }

    TriMesh subset;
    subset.vertices = mesh.vertices;
    subset.triangles.resize(static_cast<Eigen::Index>(triangle_indices.size()),
                            3);
    subset.face_ids.resize(static_cast<Eigen::Index>(triangle_indices.size()));

    for (Eigen::Index row = 0;
         row < static_cast<Eigen::Index>(triangle_indices.size()); ++row) {
        const Eigen::Index tri_idx =
            triangle_indices[static_cast<std::size_t>(row)];
        subset.triangles.row(row) = mesh.triangles.row(tri_idx);
        subset.face_ids[row] = mesh.face_ids[tri_idx];
    }

    return subset;
}

[[nodiscard]] std::string geometry_name_or_empty(const GeometryModel& model,
                                                 GeometryId geometry_id) {
    const auto name = model.name_of(geometry_id);
    return name.value_or(std::string{});
}

void require_face_id_matches_geometry(
    const std::string& name, std::uint64_t face_id,
    const std::unordered_set<std::uint64_t>& panel_face_ids,
    const std::unordered_set<std::uint64_t>& tube_face_ids,
    const char* failure_prefix) {
    if (name == "panel") {
        REQUIRE(panel_face_ids.contains(face_id));
        return;
    }
    if (name == "tube") {
        REQUIRE(tube_face_ids.contains(face_id));
        return;
    }
    FAIL(std::string(failure_prefix) + name);
}

void require_triangle_metadata_for_triangle(
    const GeometryModel& model, const UnifiedTriMesh& mesh,
    const std::unordered_set<std::uint64_t>& panel_face_ids,
    const std::unordered_set<std::uint64_t>& tube_face_ids,
    const char* failure_prefix, Eigen::Index tri_idx) {
    const auto geometry_id =
        static_cast<GeometryId>(mesh.geometry_ids[tri_idx]);
    REQUIRE(pycanha::gmm::kind_of(geometry_id) == Kind::Item);

    const std::string name = geometry_name_or_empty(model, geometry_id);
    REQUIRE_FALSE(name.empty());
    require_face_id_matches_geometry(name, mesh.face_ids[tri_idx],
                                     panel_face_ids, tube_face_ids,
                                     failure_prefix);
}

void require_triangle_metadata(
    const GeometryModel& model, const UnifiedTriMesh& mesh,
    const std::unordered_set<std::uint64_t>& panel_face_ids,
    const std::unordered_set<std::uint64_t>& tube_face_ids,
    const char* failure_prefix) {
    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        require_triangle_metadata_for_triangle(model, mesh, panel_face_ids,
                                               tube_face_ids, failure_prefix,
                                               tri_idx);
    }
}

[[nodiscard]] SceneIds add_scene_items(GeometryModel& model) {
    model.add_group(
        "rig",
        Group(CoordinateTransformation::from_translation({5.0, 0.0, 0.0})));

    return {model.add_item("panel", make_panel_item(), "rig"),
            model.add_item("tube", make_tube_item(), "rig")};
}

void require_initial_mesh(const UnifiedTriMesh& mesh) {
    REQUIRE(mesh.vertices.rows() > 0);
    REQUIRE(mesh.triangles.rows() > 0);
    REQUIRE(mesh.face_ids.rows() == mesh.triangles.rows());
    REQUIRE(mesh.geometry_ids.rows() == mesh.triangles.rows());
}

[[nodiscard]] double mutate_panel_without_invalidating(
    GeometryModel& model, GeometryId panel_id, const UnifiedTriMesh& mesh) {
    const std::uint64_t structure_version_before_content_change =
        model.get_structure_version();
    const Eigen::MatrixX3d cached_vertices = mesh.vertices;

    Item* panel = model.item_optional("panel");
    REQUIRE(panel != nullptr);
    panel->set_transform(
        CoordinateTransformation::from_translation({0.0, 0.0, 0.1}));

    REQUIRE(model.get_structure_version() ==
            structure_version_before_content_change);
    REQUIRE(model.unified_mesh().vertices.isApprox(cached_vertices));

    return area_for_geometry(model.unified_mesh(), panel_id);
}

void reparent_panel_into_cut_group(GeometryModel& model) {
    CutGroup trim(CoordinateTransformation::from_translation({5.0, 0.0, 0.0}));
    trim.add_cutter(Cylinder({1.0, 0.5, -1.0}, {1.0, 0.5, 1.0},
                             {1.35, 0.5, -1.0}, 0.35, 0.0,
                             2.0 * std::numbers::pi));

    const std::uint64_t structure_version_before_cut =
        model.get_structure_version();
    model.add_cut_group("trim", std::move(trim));
    model.reparent("panel", "trim");
    REQUIRE(model.get_structure_version() == structure_version_before_cut + 2U);
}

void assign_face_node_mapping(GeometryModel& model, FaceId face_x,
                              FaceId face_y, FaceId face_z) {
    model.assign_face_to_node(face_x, 42);
    model.assign_face_to_node(face_y, 42);
    model.assign_face_to_node(face_z, 7);
}

void require_forward_face_node_mapping(const GeometryModel& model,
                                       FaceId face_x, FaceId face_y,
                                       FaceId face_z) {
    REQUIRE(model.face_to_node(face_x) ==
            std::optional<pycanha::NodeNum>{42});
    REQUIRE(model.face_to_node(face_y) ==
            std::optional<pycanha::NodeNum>{42});
    REQUIRE(model.face_to_node(face_z) ==
            std::optional<pycanha::NodeNum>{7});
}

void require_faces_for_node_42(std::span<const FaceId> faces_for_42,
                               FaceId face_x, FaceId face_y, FaceId face_z) {
    REQUIRE(faces_for_42.size() == 2U);
    REQUIRE(std::array{faces_for_42[0], faces_for_42[1]} !=
            std::array{face_z, face_z});
    REQUIRE(std::find(faces_for_42.begin(), faces_for_42.end(), face_x) !=
            faces_for_42.end());
    REQUIRE(std::find(faces_for_42.begin(), faces_for_42.end(), face_y) !=
            faces_for_42.end());
}

void require_faces_for_node_7(std::span<const FaceId> faces_for_7,
                              FaceId face_z) {
    REQUIRE(faces_for_7.size() == 1U);
    REQUIRE(faces_for_7[0] == face_z);
}

void require_reverse_face_node_mapping(const GeometryModel& model,
                                       FaceId face_x, FaceId face_y,
                                       FaceId face_z) {
    const auto faces_for_42 = model.faces_of_node(42);
    const auto faces_for_7 = model.faces_of_node(7);
    const auto faces_for_99 = model.faces_of_node(99);

    require_faces_for_node_42(faces_for_42, face_x, face_y, face_z);
    require_faces_for_node_7(faces_for_7, face_z);
    REQUIRE(faces_for_99.empty());
}

void require_face_node_mapping(GeometryModel& model,
                               const ThermalMesh& panel_mesh,
                               const ThermalMesh& tube_mesh) {
    const FaceId face_x = tm_face_id(panel_mesh, 0U, 0U, 1U);
    const FaceId face_y = tm_face_id(panel_mesh, 1U, 0U, 1U);
    const FaceId face_z = tm_face_id(tube_mesh, 1U, 1U, 2U);

    assign_face_node_mapping(model, face_x, face_y, face_z);
    require_forward_face_node_mapping(model, face_x, face_y, face_z);
    require_reverse_face_node_mapping(model, face_x, face_y, face_z);
}

}  // namespace

TEST_CASE("GeometryModel supports the public scene-building workflow",
          "[api][geometrymodel]") {
    GeometryModel model("scene");
    const ThermalMesh panel_mesh{{0.0, 0.5, 1.0}, {0.0, 1.0}};
    const ThermalMesh tube_mesh{{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}};
    const auto panel_face_ids = face_id_set(panel_mesh);
    const auto tube_face_ids = face_id_set(tube_mesh);

    const SceneIds scene_ids = add_scene_items(model);

    const auto& initial_mesh = model.unified_mesh();
    require_initial_mesh(initial_mesh);
    require_triangle_metadata(model, initial_mesh, panel_face_ids,
                              tube_face_ids,
                              "Unexpected geometry in unified mesh: ");

    const double panel_area_before_cut = mutate_panel_without_invalidating(
        model, scene_ids.panel_id, initial_mesh);

    reparent_panel_into_cut_group(model);

    const auto& cut_mesh = model.unified_mesh();
    require_initial_mesh(cut_mesh);
    REQUIRE(area_for_geometry(cut_mesh, scene_ids.panel_id) <
            panel_area_before_cut);

    const TriMesh cut_panel_mesh =
        subset_for_geometry(cut_mesh, scene_ids.panel_id);
    REQUIRE_FALSE(mesh_ops::boundary_edge_loops(cut_panel_mesh).empty());
    require_triangle_metadata(model, cut_mesh, panel_face_ids, tube_face_ids,
                              "Unexpected geometry in cut unified mesh: ");

    require_face_node_mapping(model, panel_mesh, tube_mesh);

    REQUIRE(model.name_of(scene_ids.panel_id) ==
            std::optional<std::string>{"panel"});
    REQUIRE(model.name_of(scene_ids.tube_id) ==
            std::optional<std::string>{"tube"});
}
