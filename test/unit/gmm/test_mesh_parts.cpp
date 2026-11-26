#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/cutting/cutter.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/scene_part.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cube;
using pycanha::gmm::CutSense;
using pycanha::gmm::Geometry;
using pycanha::gmm::GeometryGroup;
using pycanha::gmm::GeometryGroupCutted;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriMeshF;
using pycanha::radiative::PartKind;
using pycanha::radiative::ScenePart;

[[nodiscard]] std::shared_ptr<GeometryItem> make_panel(
    const std::string& name, CoordinateTransformation transform = {}) {
    // Two dir1 face pairs -> two face pairs (4 faces) per panel.
    ThermalMesh thermal_mesh{{0.0, 0.5, 1.0}, {0.0, 1.0}};
    return std::make_shared<GeometryItem>(
        name, Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        std::move(thermal_mesh), std::move(transform));
}

// Per-triangle face ids of a mesh, sorted (multiset semantics).
[[nodiscard]] std::vector<pycanha::MeshIndex> sorted_face_ids(
    const TriMeshF& mesh) {
    std::vector<pycanha::MeshIndex> ids;
    ids.reserve(static_cast<std::size_t>(mesh.face_ids.rows()));
    for (pycanha::Index row = 0; row < mesh.face_ids.rows(); ++row) {
        ids.push_back(mesh.face_ids(row));
    }
    std::ranges::sort(ids);
    return ids;
}

}  // namespace

TEST_CASE("mesh_parts: single part when no split", "[gmm][parts]") {
    GeometryModel model("scene");
    model.add(make_panel("body"));
    model.add(make_panel(
        "lid", CoordinateTransformation::from_translation({0.0, 0.0, 1.0})));

    const auto parts = model.mesh_parts();
    const TriMeshF& full = model.mesh();

    REQUIRE(parts.size() == 1);
    REQUIRE(parts[0].kind == PartKind::Spacecraft);
    REQUIRE(parts[0].part_id == 0);
    REQUIRE(parts[0].transform.is_identity());
    REQUIRE(parts[0].mesh.vertices.rows() == full.vertices.rows());
    REQUIRE(parts[0].mesh.triangles.rows() == full.triangles.rows());
    REQUIRE(parts[0].mesh.vertices.cwiseEqual(full.vertices).all());
    REQUIRE(parts[0].mesh.triangles.cwiseEqual(full.triangles).all());
    REQUIRE(parts[0].mesh.face_ids.cwiseEqual(full.face_ids).all());
}

TEST_CASE("mesh_parts: split group into its own part", "[gmm][parts]") {
    GeometryModel model("scene");
    model.add(make_panel("body"));
    const CoordinateTransformation wing_tf =
        CoordinateTransformation::from_translation({5.0, 0.0, 0.0});
    model.add(std::make_shared<GeometryGroup>(
        "wing", std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{},
        wing_tf));
    model.add(make_panel("wing_panel"), "wing");

    const std::vector<std::string> split{"wing"};
    const auto parts = model.mesh_parts(split);
    const TriMeshF& full = model.mesh();

    REQUIRE(parts.size() == 2);
    REQUIRE(parts[0].kind == PartKind::Spacecraft);
    REQUIRE(parts[1].kind == PartKind::Articulated);
    REQUIRE(parts[1].part_id == 1);

    // Face ids stay global: the parts' id sets are disjoint and their union
    // is exactly the full model's.
    auto combined = sorted_face_ids(parts[0].mesh);
    const auto wing_ids = sorted_face_ids(parts[1].mesh);
    combined.insert(combined.end(), wing_ids.begin(), wing_ids.end());
    std::ranges::sort(combined);
    REQUIRE(combined == sorted_face_ids(full));

    // The wing part is in its local frame (untranslated vertices) with the
    // translation carried by part.transform (== wing's world transform).
    REQUIRE(parts[1].mesh.vertices.col(0).maxCoeff() <= 1.0F);
    REQUIRE(parts[1]
                .transform.apply({0.0, 0.0, 0.0})
                .isApprox(pycanha::Point3D{5.0, 0.0, 0.0}));
    // The remainder part is in the world frame already.
    REQUIRE(parts[0].transform.is_identity());
}

TEST_CASE("mesh_parts: nested articulated group", "[gmm][parts]") {
    GeometryModel model("scene");
    const CoordinateTransformation parent_tf =
        CoordinateTransformation::from_translation({0.0, 3.0, 0.0});
    const CoordinateTransformation wing_tf =
        CoordinateTransformation::from_translation({5.0, 0.0, 0.0});
    model.add(std::make_shared<GeometryGroup>(
        "assembly", std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{},
        parent_tf));
    model.add(
        std::make_shared<GeometryGroup>(
            "wing", std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{},
            wing_tf),
        "assembly");
    model.add(make_panel("wing_panel"), "wing");

    const std::vector<std::string> split{"wing"};
    const auto parts = model.mesh_parts(split);

    REQUIRE(parts.size() == 1);  // empty remainder is omitted
    REQUIRE(parts[0].kind == PartKind::Articulated);
    REQUIRE(parts[0].part_id == 0);
    // Part transform composes parent x wing; vertices stay wing-local.
    REQUIRE(parts[0]
                .transform.apply({0.0, 0.0, 0.0})
                .isApprox(pycanha::Point3D{5.0, 3.0, 0.0}));
    REQUIRE(parts[0].mesh.vertices.col(0).maxCoeff() <= 1.0F);
    REQUIRE(parts[0].mesh.vertices.col(1).maxCoeff() <= 1.0F);
}

TEST_CASE("mesh_parts: unknown split name throws", "[gmm][parts]") {
    GeometryModel model("scene");
    model.add(make_panel("body"));

    const std::vector<std::string> unknown{"no_such_group"};
    REQUIRE_THROWS_AS(model.mesh_parts(unknown), std::invalid_argument);

    const std::vector<std::string> duplicated{"body", "body"};
    REQUIRE_THROWS_AS(model.mesh_parts(duplicated), std::invalid_argument);
}

TEST_CASE("mesh_parts: item inside a split group not duplicated",
          "[gmm][parts]") {
    GeometryModel model("scene");
    model.add(make_panel("body"));
    model.add(std::make_shared<GeometryGroup>("wing"));
    model.add(make_panel("wing_panel"), "wing");

    const std::vector<std::string> split{"wing"};
    const auto parts = model.mesh_parts(split);
    const TriMeshF& full = model.mesh();

    const auto total_triangles =
        std::accumulate(parts.begin(), parts.end(), pycanha::Index{0},
                        [](pycanha::Index acc, const ScenePart& part) {
                            return acc + part.mesh.triangles.rows();
                        });
    REQUIRE(total_triangles == full.triangles.rows());
}

TEST_CASE("mesh_parts: item as split target", "[gmm][parts]") {
    GeometryModel model("scene");
    model.add(make_panel("body"));
    model.add(make_panel(
        "probe", CoordinateTransformation::from_translation({2.0, 0.0, 0.0})));

    const std::vector<std::string> split{"probe"};
    const auto parts = model.mesh_parts(split);

    REQUIRE(parts.size() == 2);
    REQUIRE(parts[1].kind == PartKind::Articulated);
    // The item's own transform moves to the part transform; the part mesh is
    // expressed in the item frame.
    REQUIRE(parts[1]
                .transform.apply({0.0, 0.0, 0.0})
                .isApprox(pycanha::Point3D{2.0, 0.0, 0.0}));
    REQUIRE(parts[1].mesh.vertices.col(0).maxCoeff() <= 1.0F + 1e-6F);
}

// The cut group sits in the body and its target holds the split wing, so the
// cutter reaches across the part boundary. Each part must come out trimmed
// exactly as the model is.
TEST_CASE("mesh_parts: a trim reaches across a part boundary", "[gmm][parts]") {
    GeometryModel model("scene");
    const auto wing = std::make_shared<GeometryGroup>(
        "wing", std::vector<std::shared_ptr<Geometry>>{make_panel("panel")},
        CoordinateTransformation::from_translation({5.0, 0.0, 0.0}));
    const auto body = make_panel("body");
    const auto all = std::make_shared<GeometryGroup>(
        "all", std::vector<std::shared_ptr<Geometry>>{body, wing});
    // Keeps x in [0.75, 5.25]: the last quarter of the body panel and the
    // first quarter of the wing's.
    const auto cutter = std::make_shared<GeometryItem>(
        "trim", Cube({3.0, 0.5, 0.0}, {4.5, 2.0, 2.0}), ThermalMesh{});
    model.add(std::make_shared<GeometryGroupCutted>(
        "cut", std::vector<std::shared_ptr<Geometry>>{all},
        std::vector<std::shared_ptr<GeometryItem>>{cutter},
        CoordinateTransformation{},
        std::vector<CutSense>{CutSense::KeepInside}));

    const std::vector<std::string> split{"wing"};
    const auto parts = model.mesh_parts(split);
    REQUIRE(parts.size() == 2);

    auto combined = sorted_face_ids(parts[0].mesh);
    const auto wing_ids = sorted_face_ids(parts[1].mesh);
    combined.insert(combined.end(), wing_ids.begin(), wing_ids.end());
    std::ranges::sort(combined);
    REQUIRE(combined == sorted_face_ids(model.mesh()));

    const auto area_of = [](const TriMeshF& mesh) {
        return pycanha::gmm::mesh::ops::compute_areas(mesh).sum();
    };
    REQUIRE(area_of(parts[0].mesh) == Catch::Approx(0.25).epsilon(1e-5));
    REQUIRE(area_of(parts[1].mesh) == Catch::Approx(0.25).epsilon(1e-5));
    // In the wing's own frame the surviving strip starts at its edge.
    REQUIRE(parts[0].mesh.vertices.col(0).minCoeff() ==
            Catch::Approx(0.75).epsilon(1e-5));
    REQUIRE(parts[1].mesh.vertices.col(0).minCoeff() ==
            Catch::Approx(0.0).margin(1e-5));
    REQUIRE(parts[1].mesh.vertices.col(0).maxCoeff() ==
            Catch::Approx(0.25).epsilon(1e-5));
}
