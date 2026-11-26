// Cutters that remove their outside (trims), resolved through the scene tree.

#include <Eigen/Dense>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/cutting/cutter.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/ops/validate.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cube;
using pycanha::gmm::CutSense;
using pycanha::gmm::Geometry;
using pycanha::gmm::GeometryGroup;
using pycanha::gmm::GeometryGroupCutted;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::NO_NODE;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriMeshD;
using Cutters = std::vector<std::shared_ptr<GeometryItem>>;
using Senses = std::vector<CutSense>;
using Targets = std::vector<std::shared_ptr<Geometry>>;
namespace mesh_ops = pycanha::gmm::mesh::ops;

// A 4 m x 1 m plate along +x in the z = 0 plane.
[[nodiscard]] std::shared_ptr<GeometryItem> make_plate(
    ThermalMesh thermal_mesh = {}) {
    return std::make_shared<GeometryItem>(
        "plate", Rectangle({0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        std::move(thermal_mesh));
}

// A box reaching beyond the plate in y and z, of the given extent in x.
[[nodiscard]] std::shared_ptr<GeometryItem> make_box(
    const char* name, double x_min, double x_max,
    CoordinateTransformation transform = {}) {
    return std::make_shared<GeometryItem>(
        name,
        Cube({0.5 * (x_min + x_max), 0.5, 0.0}, {x_max - x_min, 2.0, 2.0}),
        ThermalMesh{}, std::move(transform));
}

[[nodiscard]] std::shared_ptr<GeometryGroupCutted> make_cut(
    const char* name, Targets targets, Cutters cutters,
    const Senses& cut_senses = {}, CoordinateTransformation transform = {}) {
    return std::make_shared<GeometryGroupCutted>(
        name, std::move(targets), std::move(cutters), std::move(transform),
        cut_senses);
}

[[nodiscard]] Senses senses_of(const GeometryGroupCutted& cut) {
    const auto senses = cut.cut_senses();
    return {senses.begin(), senses.end()};
}

// The face areas as a whole, so that one assertion compares them all.
[[nodiscard]] bool has_face_areas(const TriMeshD& mesh,
                                  const Eigen::VectorXd& expected) {
    return mesh_ops::compute_face_areas(mesh).isApprox(expected, 1e-6);
}

[[nodiscard]] double area_of(const Geometry& node) {
    return mesh_ops::compute_areas(node.mesh()).sum();
}

// Extent of a node's mesh along x.
[[nodiscard]] std::array<double, 2> x_range(const Geometry& node) {
    return {node.mesh().vertices.col(0).minCoeff(),
            node.mesh().vertices.col(0).maxCoeff()};
}

void require_same_mesh(const TriMeshD& lhs, const TriMeshD& rhs) {
    REQUIRE(lhs.nf() == rhs.nf());
    REQUIRE(lhs.vertices.rows() == rhs.vertices.rows());
    REQUIRE(lhs.triangles.rows() == rhs.triangles.rows());
    REQUIRE(lhs.vertices.cwiseEqual(rhs.vertices).all());
    REQUIRE(lhs.triangles.cwiseEqual(rhs.triangles).all());
    REQUIRE(lhs.face_ids.cwiseEqual(rhs.face_ids).all());
}

}  // namespace

TEST_CASE("A cutter removes its inside by default", "[gmm][scene][resolve]") {
    auto cut = make_cut("cut", {make_plate()}, {make_box("c1", 0.0, 1.0)});
    cut->cut_with(make_box("c2", 1.0, 2.0));
    REQUIRE(senses_of(*cut) ==
            Senses{CutSense::KeepOutside, CutSense::KeepOutside});
}

TEST_CASE("A cut group takes one cut sense per cutter",
          "[gmm][scene][resolve]") {
    const auto build = [](const Senses& senses) {
        return make_cut("cut", {make_plate()},
                        {make_box("c1", 0.0, 1.0), make_box("c2", 1.0, 2.0)},
                        senses);
    };
    const Senses senses{CutSense::KeepOutside, CutSense::KeepInside};
    REQUIRE(senses_of(*build(senses)) == senses);
    REQUIRE_THROWS_AS(build({CutSense::KeepInside}), std::invalid_argument);
}

TEST_CASE("A refused cutter adds no cut sense", "[gmm][scene][resolve]") {
    const auto hole = make_box("hole", 0.0, 1.0);
    auto cut = make_cut("cut", {make_plate()}, {hole});
    REQUIRE_THROWS_AS(cut->cut_with(hole, CutSense::KeepInside),
                      std::invalid_argument);
    REQUIRE(cut->cutters().size() == 1U);
    REQUIRE(senses_of(*cut) == Senses{CutSense::KeepOutside});
}

TEST_CASE("The two senses of a cutter split the target",
          "[gmm][scene][resolve]") {
    const auto left =
        make_cut("left", {make_plate()}, {make_box("trim", 0.0, 1.5)},
                 {CutSense::KeepInside});
    const auto removed =
        make_cut("removed", {make_plate()}, {make_box("hole", 0.0, 1.5)});

    CHECK(area_of(*left) == Catch::Approx(1.5).margin(1e-6));
    CHECK(area_of(*removed) == Catch::Approx(2.5).margin(1e-6));
    CHECK(x_range(*left)[0] == Catch::Approx(0.0).margin(1e-9));
    CHECK(x_range(*left)[1] == Catch::Approx(1.5).margin(1e-9));
    REQUIRE(mesh_ops::has_consistent_face_ids(left->mesh()));
}

// Naming the default must not change anything, down to the last vertex.
TEST_CASE("Cutters that remove their inside resolve as they always did",
          "[gmm][scene][resolve]") {
    const auto implicit =
        make_cut("cut", {make_plate()},
                 {make_box("c1", 0.0, 1.0), make_box("c2", 2.0, 3.0)});
    const auto named =
        make_cut("cut", {make_plate()},
                 {make_box("c1", 0.0, 1.0), make_box("c2", 2.0, 3.0)},
                 {CutSense::KeepOutside, CutSense::KeepOutside});
    require_same_mesh(implicit->mesh(), named->mesh());
    REQUIRE(area_of(*named) == Catch::Approx(2.0).margin(1e-6));
}

TEST_CASE("A trim added later is the same as one given to the constructor",
          "[gmm][scene][resolve]") {
    const auto built =
        make_cut("cut", {make_plate()},
                 {make_box("hole", 0.0, 1.0), make_box("trim", 0.5, 3.0)},
                 {CutSense::KeepOutside, CutSense::KeepInside});
    auto added = make_cut("cut", {make_plate()}, {make_box("hole", 0.0, 1.0)});
    // The trim also takes the area the hole does not reach.
    REQUIRE(area_of(*added) == Catch::Approx(3.0).margin(1e-6));
    added->cut_with(make_box("trim", 0.5, 3.0), CutSense::KeepInside);
    require_same_mesh(built->mesh(), added->mesh());
    REQUIRE(area_of(*added) == Catch::Approx(2.0).margin(1e-6));
    CHECK(x_range(*added)[0] == Catch::Approx(1.0).margin(1e-9));
    CHECK(x_range(*added)[1] == Catch::Approx(3.0).margin(1e-9));
}

// One cutter of each kind, on one group or spread over two in either nesting
// order: always the same shape.
TEST_CASE("Nested cut groups combine trims and holes",
          "[gmm][scene][resolve]") {
    const auto flat =
        make_cut("flat", {make_plate()},
                 {make_box("hole", 0.0, 1.0), make_box("trim", 0.5, 3.0)},
                 {CutSense::KeepOutside, CutSense::KeepInside});

    const auto hole_inside =
        make_cut("inner", {make_plate()}, {make_box("hole", 0.0, 1.0)});
    const auto trim_outside =
        make_cut("outer", {hole_inside}, {make_box("trim", 0.5, 3.0)},
                 {CutSense::KeepInside});
    const auto trim_inside =
        make_cut("inner", {make_plate()}, {make_box("trim", 0.5, 3.0)},
                 {CutSense::KeepInside});
    const auto hole_outside =
        make_cut("outer", {trim_inside}, {make_box("hole", 0.0, 1.0)});

    REQUIRE(area_of(*flat) == Catch::Approx(2.0).margin(1e-6));
    require_same_mesh(flat->mesh(), trim_outside->mesh());
    require_same_mesh(flat->mesh(), hole_outside->mesh());
    // Resolved at the inner group, the outer cutter is out of scope.
    REQUIRE(area_of(*hole_inside) == Catch::Approx(3.0).margin(1e-6));
    REQUIRE(area_of(*trim_inside) == Catch::Approx(2.5).margin(1e-6));
}

TEST_CASE("Two trims leave what is inside both", "[gmm][scene][resolve]") {
    const auto cut =
        make_cut("cut", {make_plate()},
                 {make_box("t1", 0.0, 2.0), make_box("t2", 1.25, 4.0)},
                 {CutSense::KeepInside, CutSense::KeepInside});
    CHECK(area_of(*cut) == Catch::Approx(0.75).margin(1e-6));
    CHECK(x_range(*cut)[0] == Catch::Approx(1.25).margin(1e-9));
    CHECK(x_range(*cut)[1] == Catch::Approx(2.0).margin(1e-9));
}

TEST_CASE("A trim cuts every item of a target group", "[gmm][scene][resolve]") {
    const auto panel = [](const char* name, double x_origin) {
        return std::make_shared<GeometryItem>(
            name,
            Rectangle({x_origin, 0.0, 0.0}, {x_origin + 2.0, 0.0, 0.0},
                      {x_origin, 1.0, 0.0}),
            ThermalMesh{});
    };
    const auto left = panel("left", 0.0);
    const auto right = panel("right", 2.0);
    const auto away = panel("away", 10.0);
    const auto all =
        std::make_shared<GeometryGroup>("all", Targets{left, right, away});
    // Straddles the seam: half a square metre of each near panel is left, and
    // nothing of the far one.
    const auto cut = make_cut("cut", {all}, {make_box("trim", 1.5, 2.5)},
                              {CutSense::KeepInside});

    const TriMeshD& mesh = cut->mesh();
    REQUIRE(mesh_ops::has_consistent_face_ids(mesh));
    // Every item still owns its faces, in tree order.
    REQUIRE(mesh.nf() == 6U);
    REQUIRE(mesh.primitives.size() == 3U);
    REQUIRE(mesh.primitives[0].geometry_id == left->id());
    REQUIRE(mesh.primitives[2].geometry_id == away->id());
    REQUIRE(
        has_face_areas(mesh, Eigen::VectorXd{{0.5, 0.5, 0.5, 0.5, 0.0, 0.0}}));
}

TEST_CASE("A trim leaves face ids and node numbers where they were",
          "[gmm][scene][faces]") {
    ThermalMesh thermal_mesh{{0.0, 0.25, 0.5, 0.75, 1.0}, {0.0, 1.0}};
    thermal_mesh.set_node1_start(100);
    thermal_mesh.set_node1_step(1);
    thermal_mesh.set_node2_start(200);
    thermal_mesh.set_node2_step(1);
    // Leaves the second and the third of the four subdivisions.
    const auto cut =
        make_cut("cut", {make_plate(std::move(thermal_mesh))},
                 {make_box("trim", 1.0, 3.0)}, {CutSense::KeepInside});

    const TriMeshD& mesh = cut->mesh();
    REQUIRE(mesh.nf() == 8U);
    REQUIRE(mesh_ops::has_consistent_face_ids(mesh));
    REQUIRE(has_face_areas(
        mesh, Eigen::VectorXd{{0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0}}));
    const std::vector<pycanha::NodeNum> nodes(mesh.node_numbers.begin(),
                                              mesh.node_numbers.end());
    REQUIRE(nodes == std::vector<pycanha::NodeNum>{NO_NODE, NO_NODE, 101, 201,
                                                   102, 202, NO_NODE, NO_NODE});
}

TEST_CASE("A trim follows the transform of its cutter and of its ancestors",
          "[gmm][scene][resolve]") {
    // The box spans x in [0, 1] in its own frame and sits 2 m further on.
    const auto cut = make_cut(
        "cut", {make_plate()},
        {make_box("trim", 0.0, 1.0,
                  CoordinateTransformation::from_translation({2.0, 0.0, 0.0}))},
        {CutSense::KeepInside},
        CoordinateTransformation::from_translation({0.0, 0.0, 5.0}));

    CHECK(area_of(*cut) == Catch::Approx(1.0).margin(1e-6));
    CHECK(x_range(*cut)[0] == Catch::Approx(2.0).margin(1e-9));
    CHECK(x_range(*cut)[1] == Catch::Approx(3.0).margin(1e-9));
    // A cut mesh is the outer surface of a thin shell about the plate.
    CHECK(cut->mesh().vertices.col(2).minCoeff() ==
          Catch::Approx(5.0).margin(1e-3));

    // A quarter turn about z takes +x onto +y: the survivor goes along.
    const GeometryGroup rig(
        "rig", {cut},
        CoordinateTransformation::from_rotation(Eigen::AngleAxisd(
            0.5 * std::numbers::pi, pycanha::Vector3D::UnitZ())));
    CHECK(area_of(rig) == Catch::Approx(1.0).margin(1e-6));
    CHECK(rig.mesh().vertices.col(1).minCoeff() ==
          Catch::Approx(2.0).margin(1e-9));
    CHECK(rig.mesh().vertices.col(1).maxCoeff() ==
          Catch::Approx(3.0).margin(1e-9));
}
