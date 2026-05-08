#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <numbers>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/group.hpp"
#include "pycanha-core/gmm/scene/item.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::Group;
using pycanha::gmm::Item;
using pycanha::gmm::Kind;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
namespace mesh_ops = pycanha::gmm::mesh::ops;

[[nodiscard]] Item make_item(double width, double height,
                             CoordinateTransformation transform = {}) {
    return Item(
        Rectangle({0.0, 0.0, 0.0}, {width, 0.0, 0.0}, {0.0, height, 0.0}),
        ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 1.0}}, transform);
}

}  // namespace

TEST_CASE("GeometryModel unified mesh is empty for an empty model",
          "[gmm][geometrymodel][mesh]") {
    const GeometryModel model("scene");

    const auto& mesh = model.unified_mesh();
    REQUIRE(mesh.vertices.rows() == 0);
    REQUIRE(mesh.triangles.rows() == 0);
    REQUIRE(mesh.face_ids.rows() == 0);
    REQUIRE(mesh.geometry_ids.rows() == 0);
}

TEST_CASE("GeometryModel unified mesh appends root items with geometry ids",
          "[gmm][geometrymodel][mesh]") {
    GeometryModel model("scene");
    model.add_item("panel", make_item(2.0, 1.0));

    const auto& mesh = model.unified_mesh();
    const auto areas = mesh_ops::compute_areas(mesh);
    const double area_sum = areas.sum();

    REQUIRE(mesh.vertices.rows() > 0);
    REQUIRE(mesh.triangles.rows() > 0);
    REQUIRE(area_sum == Catch::Approx(2.0));
    REQUIRE(std::all_of(mesh.geometry_ids.begin(), mesh.geometry_ids.end(),
                        [](std::uint64_t value) {
                            return value == pycanha::gmm::to_raw(
                                                pycanha::gmm::make_geometry_id(
                                                    Kind::Item, 0U));
                        }));
    REQUIRE(std::is_sorted(mesh.face_ids.begin(), mesh.face_ids.end()));
}

TEST_CASE("GeometryModel unified mesh composes group and item transforms",
          "[gmm][geometrymodel][mesh]") {
    GeometryModel model("scene");
    model.add_group(
        "rig",
        Group(CoordinateTransformation::from_translation({10.0, 0.0, 0.0})));
    model.add_item(
        "panel-a",
        make_item(1.0, 1.0,
                  CoordinateTransformation::from_translation({0.0, 2.0, 0.0})),
        "rig");
    model.add_item(
        "panel-b",
        make_item(1.0, 1.0,
                  CoordinateTransformation::from_euler(
                      {0.0, 0.0, 3.0}, {0.0, 0.0, std::numbers::pi / 2.0})),
        "rig");

    const auto& mesh = model.unified_mesh();
    REQUIRE(mesh.vertices.rows() >= 8);
    REQUIRE(mesh.triangles.rows() >= 4);
    REQUIRE(mesh.triangles.maxCoeff() < mesh.vertices.rows());

    bool found_panel_a_vertex = false;
    bool found_panel_b_vertex = false;
    for (Eigen::Index vertex_idx = 0; vertex_idx < mesh.vertices.rows();
         ++vertex_idx) {
        const auto vertex = mesh.vertices.row(vertex_idx);
        found_panel_a_vertex =
            found_panel_a_vertex ||
            vertex.isApprox(Eigen::RowVector3d(10.0, 2.0, 0.0));
        found_panel_b_vertex =
            found_panel_b_vertex ||
            vertex.isApprox(Eigen::RowVector3d(10.0, 0.0, 3.0));
    }

    REQUIRE(found_panel_a_vertex);
    REQUIRE(found_panel_b_vertex);
}

TEST_CASE("GeometryModel unified mesh stays cached across content mutation",
          "[gmm][geometrymodel][mesh]") {
    GeometryModel model("scene");
    model.add_item("panel", make_item(2.0, 1.0));

    const auto before = model.unified_mesh().vertices;
    auto* item = model.item_optional("panel");
    REQUIRE(item != nullptr);
    item->set_primitive(
        Rectangle({0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {0.0, 2.0, 0.0}));

    const auto cached = model.unified_mesh().vertices;
    REQUIRE(cached.isApprox(before));

    model.invalidate_unified_mesh();
    const auto rebuilt = model.unified_mesh().vertices;
    REQUIRE_FALSE(rebuilt.isApprox(before));
}

TEST_CASE("GeometryModel unified mesh rebuilds after structural mutation",
          "[gmm][geometrymodel][mesh]") {
    GeometryModel model("scene");
    model.add_group("rig", Group{});
    model.add_item("panel-a", make_item(1.0, 1.0), "rig");

    const auto triangles_before = model.unified_mesh().triangles.rows();
    const auto version_before = model.get_structure_version();

    model.add_item("panel-b", make_item(1.0, 1.0), "rig");
    REQUIRE(model.get_structure_version() == version_before + 1U);
    REQUIRE(model.unified_mesh().triangles.rows() > triangles_before);

    const auto version_before_remove = model.get_structure_version();
    model.remove("panel-b");
    REQUIRE(model.get_structure_version() == version_before_remove + 1U);
    REQUIRE(model.unified_mesh().triangles.rows() == triangles_before);
}
