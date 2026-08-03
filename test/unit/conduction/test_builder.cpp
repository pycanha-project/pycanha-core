#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/conduction/builder.hpp"
#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"

namespace {

using pycanha::ThermalModel;
using pycanha::conduction::DiagnosticCode;
using pycanha::conduction::TmmBuildOptions;
using pycanha::conduction::TmmBuildReport;
using pycanha::gmm::ActiveSide;
using pycanha::gmm::BulkMaterial;
using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cube;
using pycanha::gmm::Cylinder;
using pycanha::gmm::GeometryGroupCutted;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::Triangle;

// conductivity 1 W/(m K) so a conductance reads as the shape factor, plus a
// unit thickness so the capacitance is easy to predict.
[[nodiscard]] std::shared_ptr<BulkMaterial> unit_bulk() {
    return std::make_shared<BulkMaterial>("unit", 1.0, 1.0, 1.0);
}

[[nodiscard]] ThermalMesh side1_shell(std::vector<double> dir1,
                                      std::vector<double> dir2,
                                      pycanha::NodeNum start, int step) {
    ThermalMesh mesh(std::move(dir1), std::move(dir2));
    mesh.set_side1_material(unit_bulk());
    mesh.set_side1_thick(1.0);
    mesh.set_node1_start(start);
    mesh.set_node1_step(step);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    return mesh;
}

[[nodiscard]] std::shared_ptr<GeometryItem> make_plate(
    const std::string& name, ThermalMesh mesh,
    CoordinateTransformation transform = {}) {
    return std::make_shared<GeometryItem>(
        name, Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        std::move(mesh), std::move(transform));
}

[[nodiscard]] bool has_code(const TmmBuildReport& report, DiagnosticCode code) {
    return std::ranges::any_of(report.diagnostics, [code](const auto& entry) {
        return entry.code == code;
    });
}

[[nodiscard]] pycanha::Index num_conductors(ThermalModel& model) {
    return model.tmm()
        .conductive_couplings()
        .matrices()
        .get_num_total_couplings();
}

}  // namespace

TEST_CASE("builder: a split plate produces one node pair and one conductor",
          "[conduction][builder]") {
    ThermalModel model("split_plate");
    model.gmm().add(
        make_plate("plate", side1_shell({0.0, 0.5, 1.0}, {0.0, 1.0}, 100, 1)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.items_processed == 1U);
    REQUIRE(report.nodes_created == 2U);
    REQUIRE(report.conductors_created == 1U);
    REQUIRE(report.cell_links_computed == 1U);
    REQUIRE(num_conductors(model) == 1);
    // 1 m shared edge over a 0.5 m distance between the two references.
    REQUIRE(model.tmm().conductive_couplings().get_coupling_value(100, 101) ==
            Catch::Approx(2.0));
}

TEST_CASE("builder: parallel paths between the same node pair are summed",
          "[conduction][builder]") {
    ThermalModel model("parallel_paths");
    // A full-revolution cylinder cut into two half-shells: they meet along two
    // separate seams, so the same node pair gets two conductors in parallel.
    ThermalMesh mesh = side1_shell({0.0, 0.5, 1.0}, {0.0, 1.0}, 1, 1);
    model.gmm().add(std::make_shared<GeometryItem>(
        "tube",
        Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {1.0, 0.0, 0.0}, 1.0, 0.0,
                 2.0 * std::numbers::pi),
        std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.nodes_created == 2U);
    // The interior seam and the wrap, reduced onto the single node pair.
    REQUIRE(report.cell_links_computed == 2U);
    REQUIRE(report.conductors_created == 1U);
    // Each seam carries height / (radius * PI), and the two add.
    REQUIRE(model.tmm().conductive_couplings().get_coupling_value(1, 2) ==
            Catch::Approx(2.0 * (2.0 / (1.0 * std::numbers::pi))));
}

TEST_CASE("builder: a self-coupling is dropped", "[conduction][builder]") {
    ThermalModel model("one_node");
    // step 0: every cell of the plate is the same node, so no in-plane
    // conductor survives.
    model.gmm().add(make_plate(
        "plate", side1_shell({0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}, 9, 0)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.nodes_created == 1U);
    REQUIRE(report.cell_links_computed == 4U);
    REQUIRE(report.conductors_created == 0U);
    REQUIRE(num_conductors(model) == 0);
}

TEST_CASE("builder: nodes shared across items merge their contributions",
          "[conduction][builder]") {
    ThermalModel model("shared_node");
    model.gmm().add(
        make_plate("plate_a", side1_shell({0.0, 1.0}, {0.0, 1.0}, 42, 0)));
    model.gmm().add(make_plate(
        "plate_b", side1_shell({0.0, 1.0}, {0.0, 1.0}, 42, 0),
        CoordinateTransformation::from_translation({5.0, 0.0, 0.0})));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.items_processed == 2U);
    // One node fed by both plates: areas and capacitances add...
    REQUIRE(report.nodes_created == 1U);
    REQUIRE(model.tmm().nodes().get_a(42) == Catch::Approx(2.0));
    REQUIRE(model.tmm().nodes().get_C(42) == Catch::Approx(2.0));
    // ...but nothing connects two different geometries.
    REQUIRE(report.conductors_created == 0U);
    // The coordinate is the area-weighted centroid of both plates.
    REQUIRE(model.tmm().nodes().get_fx(42) == Catch::Approx(3.0));
}

TEST_CASE("builder: through-thickness conductors join the two sides",
          "[conduction][builder]") {
    ThermalModel model("through_thickness");
    ThermalMesh mesh;
    mesh.set_side1_material(std::make_shared<BulkMaterial>("a", 1.0, 2.0, 1.0));
    mesh.set_side2_material(std::make_shared<BulkMaterial>("b", 1.0, 5.0, 1.0));
    mesh.set_side1_thick(0.1);
    mesh.set_side2_thick(0.2);
    mesh.set_node1_start(1);
    mesh.set_node2_start(2);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.nodes_created == 2U);
    REQUIRE(report.conductors_created == 1U);
    // A / (t1/k1 + t2/k2) = 1 / (0.05 + 0.04).
    REQUIRE(model.tmm().conductive_couplings().get_coupling_value(1, 2) ==
            Catch::Approx(1.0 / 0.09));

    ThermalModel disabled("through_thickness_off");
    ThermalMesh other;
    other.set_side1_material(
        std::make_shared<BulkMaterial>("a", 1.0, 2.0, 1.0));
    other.set_side2_material(
        std::make_shared<BulkMaterial>("b", 1.0, 5.0, 1.0));
    other.set_side1_thick(0.1);
    other.set_side2_thick(0.2);
    other.set_node1_start(1);
    other.set_node2_start(2);
    disabled.gmm().add(make_plate("plate", std::move(other)));

    TmmBuildOptions options;
    options.through_thickness_conductors = false;
    REQUIRE(disabled.build_tmm_from_gmm(options).conductors_created == 0U);
}

TEST_CASE("builder: min_conductance drops weak conductors",
          "[conduction][builder]") {
    ThermalModel model("min_conductance");
    model.gmm().add(
        make_plate("plate", side1_shell({0.0, 0.5, 1.0}, {0.0, 1.0}, 1, 1)));

    TmmBuildOptions options;
    options.min_conductance = 10.0;  // the plate's conductor is 2 W/K
    const TmmBuildReport report = model.build_tmm_from_gmm(options);
    REQUIRE(report.nodes_created == 2U);
    REQUIRE(report.conductors_created == 0U);
}

TEST_CASE("builder: intra-primitive conductors can be switched off",
          "[conduction][builder]") {
    ThermalModel model("no_intra");
    model.gmm().add(
        make_plate("plate", side1_shell({0.0, 0.5, 1.0}, {0.0, 1.0}, 1, 1)));

    TmmBuildOptions options;
    options.intra_primitive_conductors = false;
    const TmmBuildReport report = model.build_tmm_from_gmm(options);
    REQUIRE(report.nodes_created == 2U);
    REQUIRE(report.cell_links_computed == 0U);
    REQUIRE(report.conductors_created == 0U);
}

TEST_CASE("builder: a non-empty tmm is refused", "[conduction][builder]") {
    ThermalModel model("non_empty");
    model.gmm().add(
        make_plate("plate", side1_shell({0.0, 1.0}, {0.0, 1.0}, 1, 0)));
    model.tmm().add_node(1);

    REQUIRE_THROWS_AS(model.build_tmm_from_gmm(), std::invalid_argument);
}

TEST_CASE("builder: cut geometry is skipped with one diagnostic per group",
          "[conduction][builder]") {
    ThermalModel model("cut_group");
    auto target =
        make_plate("target", side1_shell({0.0, 1.0}, {0.0, 1.0}, 1, 0));
    auto cutter = std::make_shared<GeometryItem>(
        "cutter",
        Cylinder({0.5, 0.5, -1.0}, {0.5, 0.5, 1.0}, {0.8, 0.5, -1.0}, 0.3, 0.0,
                 2.0 * std::numbers::pi),
        ThermalMesh{});
    model.gmm().add(std::make_shared<GeometryGroupCutted>(
        "cut", std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{target},
        std::vector<std::shared_ptr<GeometryItem>>{cutter}));
    model.gmm().add(make_plate(
        "intact", side1_shell({0.0, 1.0}, {0.0, 1.0}, 2, 0),
        CoordinateTransformation::from_translation({0.0, 0.0, 3.0})));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(has_code(report, DiagnosticCode::CutGeometrySkipped));
    REQUIRE(std::ranges::count_if(report.diagnostics, [](const auto& entry) {
                return entry.code == DiagnosticCode::CutGeometrySkipped;
            }) == 1);
    // Only the intact plate contributes.
    REQUIRE(report.items_processed == 1U);
    REQUIRE(report.nodes_created == 1U);
    REQUIRE(model.tmm().nodes().is_node(2));
    REQUIRE_FALSE(model.tmm().nodes().is_node(1));
}

TEST_CASE("builder: a cube never reaches the builder",
          "[conduction][builder]") {
    ThermalModel model("cube_only");
    ThermalMesh mesh;
    mesh.set_node1_start(1);
    model.gmm().add(std::make_shared<GeometryItem>(
        "block", Cube({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}), std::move(mesh)));

    // A Cube is cutter-only, so it is the world mesh that refuses it: a Cube
    // may only appear as a cutter inside a cut group, which the builder skips
    // as cut geometry anyway.
    REQUIRE_THROWS(model.build_tmm_from_gmm());
}

TEST_CASE("builder: a triangle reports its discrete fallback",
          "[conduction][builder]") {
    ThermalModel model("triangle");
    ThermalMesh mesh = side1_shell({0.0, 0.5, 1.0}, {0.0, 1.0}, 1, 1);
    model.gmm().add(std::make_shared<GeometryItem>(
        "wedge", Triangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(has_code(report, DiagnosticCode::TriangleApproximated));
    REQUIRE(report.nodes_created == 2U);
    REQUIRE(report.conductors_created == 1U);
}

TEST_CASE("builder: a model with no node numbers builds nothing",
          "[conduction][builder]") {
    ThermalModel model("no_nodes");
    ThermalMesh mesh;  // node starts default to NO_NODE
    mesh.set_side1_material(unit_bulk());
    mesh.set_side1_thick(1.0);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(has_code(report, DiagnosticCode::NoNodeNumbers));
    REQUIRE(report.nodes_created == 0U);
    REQUIRE(report.conductors_created == 0U);
    REQUIRE(model.tmm().nodes().get_num_nodes() == 0);
}

TEST_CASE("builder: an empty geometry model builds an empty tmm",
          "[conduction][builder]") {
    ThermalModel model("empty");
    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.items_processed == 0U);
    REQUIRE(report.nodes_created == 0U);
    REQUIRE(report.conductors_created == 0U);
    REQUIRE(report.diagnostics.empty());
}
