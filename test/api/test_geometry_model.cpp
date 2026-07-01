#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <numbers>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Geometry;
using pycanha::gmm::GeometryGroup;
using pycanha::gmm::GeometryGroupCutted;
using pycanha::gmm::GeometryId;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriMeshF;
namespace mesh_ops = pycanha::gmm::mesh::ops;

[[nodiscard]] std::shared_ptr<GeometryItem> make_panel() {
    return std::make_shared<GeometryItem>(
        "panel", Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 1.0}});
}

[[nodiscard]] std::shared_ptr<GeometryItem> make_tube() {
    return std::make_shared<GeometryItem>(
        "tube",
        Cylinder({0.0, 0.0, -0.75}, {0.0, 0.0, 0.75}, {0.35, 0.0, -0.75}, 0.35,
                 0.0, 2.0 * std::numbers::pi),
        ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}},
        CoordinateTransformation::from_translation({0.0, 2.0, 0.0}));
}

// Area of the triangles owned by the primitive with the given geometry id.
[[nodiscard]] double area_of_geometry(const TriMeshF& mesh, GeometryId id) {
    const auto areas = mesh_ops::compute_areas(mesh);
    double sum = 0.0;
    for (const auto& range : mesh.primitives) {
        if (range.geometry_id != id) {
            continue;
        }
        for (Eigen::Index tri = 0; tri < mesh.face_ids.rows(); ++tri) {
            const auto face_id = mesh.face_ids(tri);
            if (face_id >= range.first_face_id &&
                face_id <= range.last_face_id) {
                sum += areas[tri];
            }
        }
    }
    return sum;
}

}  // namespace

TEST_CASE("GeometryModel builds a world mesh with primitive provenance",
          "[api][geometrymodel]") {
    GeometryModel model("scene");
    auto rig = std::make_shared<GeometryGroup>(
        "rig", std::vector<std::shared_ptr<Geometry>>{},
        CoordinateTransformation::from_translation({5.0, 0.0, 0.0}));
    model.add(rig);
    model.add(make_panel(), "rig");
    model.add(make_tube(), "rig");

    model.create_mesh();
    const TriMeshF& mesh = model.mesh();

    REQUIRE(mesh.vertices.rows() > 0);
    REQUIRE(mesh.triangles.rows() > 0);
    REQUIRE(mesh.face_ids.rows() == mesh.triangles.rows());
    REQUIRE(mesh.node_numbers.rows() == static_cast<Eigen::Index>(mesh.nf()));

    // Two primitives (panel, tube), each owning a disjoint face_id range.
    REQUIRE(mesh.primitives.size() == 2U);
    const GeometryId panel_id = model.get_item("panel")->id();
    const GeometryId tube_id = model.get_item("tube")->id();
    REQUIRE(area_of_geometry(mesh, panel_id) > 0.0);
    REQUIRE(area_of_geometry(mesh, tube_id) > 0.0);

    REQUIRE(model.get("panel")->name() == "panel");
    REQUIRE(model.get("tube")->name() == "tube");
}

TEST_CASE("GeometryGroupCutted reduces a target's area in the world mesh",
          "[api][geometrymodel]") {
    // Reference (uncut) panel area.
    const double full_area =
        mesh_ops::compute_areas(make_panel()->mesh()).sum();

    GeometryModel model("scene");
    auto cutter = std::make_shared<GeometryItem>(
        "cutter",
        Cylinder({1.0, 0.5, -1.0}, {1.0, 0.5, 1.0}, {1.35, 0.5, -1.0}, 0.35,
                 0.0, 2.0 * std::numbers::pi),
        ThermalMesh{});
    auto trim = std::make_shared<GeometryGroupCutted>(
        "trim", std::vector<std::shared_ptr<Geometry>>{make_panel()},
        std::vector<std::shared_ptr<GeometryItem>>{cutter});
    model.add(trim);

    model.create_mesh();
    const TriMeshF& mesh = model.mesh();
    REQUIRE(mesh.triangles.rows() > 0);

    const GeometryId panel_id = model.get_item("panel")->id();
    const double cut_area = area_of_geometry(mesh, panel_id);
    REQUIRE(cut_area > 0.0);
    REQUIRE(cut_area < full_area);
}
