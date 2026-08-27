#include <Eigen/Dense>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/ops/validate.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::gmm::Cube;
using pycanha::gmm::Geometry;
using pycanha::gmm::GeometryGroup;
using pycanha::gmm::GeometryGroupCutted;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::NO_NODE;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriMeshD;
namespace mesh_ops = pycanha::gmm::mesh::ops;

// A 2 x 1 m plate split into two subdivisions along x, so it owns two face
// pairs (four faces): side 1 gets `first_node` and `first_node + 1`, side 2
// the same numbers offset by 100.
[[nodiscard]] std::shared_ptr<GeometryItem> make_plate(
    const char* name, double x_origin, std::int32_t first_node) {
    ThermalMesh thermal_mesh{{0.0, 0.5, 1.0}, {0.0, 1.0}};
    thermal_mesh.set_node1_start(first_node);
    thermal_mesh.set_node1_step(1);
    thermal_mesh.set_node2_start(first_node + 100);
    thermal_mesh.set_node2_step(1);
    return std::make_shared<GeometryItem>(
        name,
        Rectangle({x_origin, 0.0, 0.0}, {x_origin + 2.0, 0.0, 0.0},
                  {x_origin, 1.0, 0.0}),
        thermal_mesh);
}

[[nodiscard]] Eigen::Index first_face_of(const TriMeshD& mesh,
                                         const Geometry& node) {
    const auto range = std::ranges::find(
        mesh.primitives, node.id(), &TriMeshD::PrimitiveRange::geometry_id);
    if (range == mesh.primitives.end()) {
        FAIL("geometry not present in the mesh");
        return -1;
    }
    return static_cast<Eigen::Index>(range->first_face_id);
}

}  // namespace

// A cut removes triangles; it must never remove FACES. Deriving the face count
// from the surviving triangles (max(face_ids) + 2) shrinks an item whose last
// pair was cut away, which slides every later item in the model down by two
// faces -- and with it every node number, thermo-optical property and activity
// flag, silently, from the first trailing cut onwards.
TEST_CASE("A trailing cut-away pair does not renumber later items",
          "[gmm][scene][faces]") {
    auto plate_a = make_plate("a", 0.0, 100);
    auto plate_b = make_plate("b", 4.0, 200);

    // Covers the whole second subdivision of plate a (x in [1, 2]) and nothing
    // of plate b, so a's trailing pair loses every triangle it had.
    auto cutter = std::make_shared<GeometryItem>(
        "cutter", Cube({1.5, 0.5, 0.0}, {1.2, 2.0, 2.0}), ThermalMesh{});

    auto cut_a = std::make_shared<GeometryGroupCutted>(
        "cut_a", std::vector<std::shared_ptr<Geometry>>{plate_a},
        std::vector<std::shared_ptr<GeometryItem>>{cutter});

    GeometryGroup root("root", {cut_a, plate_b});
    const TriMeshD& mesh = root.mesh();

    // Two plates of two pairs each: eight faces, whatever the cut removed.
    REQUIRE(mesh.nf() == 8U);
    REQUIRE(mesh_ops::has_consistent_face_ids(mesh));
    REQUIRE(first_face_of(mesh, *plate_a) == 0);
    REQUIRE(first_face_of(mesh, *plate_b) == 4);

    // a's trailing pair survives as ids 2/3 with no area -- the reliable test
    // for a pair a cut destroyed, and the one the radiative scene builder uses
    // to drop dead emitters.
    const auto face_areas = mesh_ops::compute_face_areas(mesh);
    REQUIRE(face_areas.size() == 8);
    REQUIRE(face_areas[0] > 0.0);
    REQUIRE(face_areas[2] == 0.0);
    REQUIRE(face_areas[3] == 0.0);
    REQUIRE(face_areas[4] > 0.0);

    // b's own node numbers are still attached to b's own geometry.
    REQUIRE(mesh.node_numbers[4] == 200);
    REQUIRE(mesh.node_numbers[5] == 300);
    REQUIRE(mesh.node_numbers[6] == 201);
    REQUIRE(mesh.node_numbers[7] == 301);
}

// An interior hole was always handled; keep it that way, and keep it distinct
// from the trailing case above.
TEST_CASE("An interior cut leaves its pair in place, marked dead",
          "[gmm][scene][faces]") {
    auto plate = make_plate("plate", 0.0, 100);
    // Covers the first subdivision (x in [0, 1]) -- an interior gap, since a
    // later pair survives.
    auto cutter = std::make_shared<GeometryItem>(
        "cutter", Cube({0.5, 0.5, 0.0}, {1.2, 2.0, 2.0}), ThermalMesh{});

    auto cut = std::make_shared<GeometryGroupCutted>(
        "cut", std::vector<std::shared_ptr<Geometry>>{plate},
        std::vector<std::shared_ptr<GeometryItem>>{cutter});

    const TriMeshD& mesh = cut->mesh();

    REQUIRE(mesh.nf() == 4U);
    const auto face_areas = mesh_ops::compute_face_areas(mesh);
    REQUIRE(face_areas[0] == 0.0);
    REQUIRE(face_areas[1] == 0.0);
    REQUIRE(face_areas[2] > 0.0);
    REQUIRE(mesh.node_numbers[0] == NO_NODE);
    REQUIRE(mesh.node_numbers[2] == 101);
}

// Parity is load-bearing: face_id % 2 IS the side. Every item must therefore
// start on an even id, which is only true while a cut result's face count
// stays even and its triangles all carry even ids.
TEST_CASE("Every item's faces start even and stay even",
          "[gmm][scene][faces]") {
    auto plate_a = make_plate("a", 0.0, 100);
    auto plate_b = make_plate("b", 4.0, 200);
    auto cutter = std::make_shared<GeometryItem>(
        "cutter", Cube({1.5, 0.5, 0.0}, {0.6, 2.0, 2.0}), ThermalMesh{});

    auto cut_a = std::make_shared<GeometryGroupCutted>(
        "cut_a", std::vector<std::shared_ptr<Geometry>>{plate_a},
        std::vector<std::shared_ptr<GeometryItem>>{cutter});

    GeometryGroup root("root", {cut_a, plate_b});
    const TriMeshD& mesh = root.mesh();

    REQUIRE(mesh.nf() % 2U == 0U);
    for (const auto& range : mesh.primitives) {
        REQUIRE(range.first_face_id % 2U == 0U);
    }
    for (Eigen::Index tri_idx = 0; tri_idx < mesh.face_ids.rows(); ++tri_idx) {
        REQUIRE(mesh.face_ids(tri_idx) % 2U == 0U);
    }
}
