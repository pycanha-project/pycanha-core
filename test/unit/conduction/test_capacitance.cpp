#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <string>
#include <utility>

#include "pycanha-core/conduction/builder.hpp"
#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
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
using pycanha::gmm::GeometryItem;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;

// density 2 kg/m3, conductivity 10 W/(m K), specific heat 3 J/(kg K).
[[nodiscard]] std::shared_ptr<BulkMaterial> make_bulk(const std::string& name) {
    return std::make_shared<BulkMaterial>(name, 2.0, 10.0, 3.0);
}

// A 1 m x 1 m plate in the z = 0 plane, one face pair.
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

}  // namespace

TEST_CASE("capacitance: a single-surfaced node sums one side only",
          "[conduction][builder]") {
    ThermalModel model("single_surfaced");
    ThermalMesh mesh;
    mesh.set_side1_material(make_bulk("alu"));
    mesh.set_side1_thick(0.05);
    mesh.set_node1_start(10);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.nodes_created == 1U);
    // rho * cp * t * A = 2 * 3 * 0.05 * 1.
    REQUIRE(model.tmm().nodes().get_C(10) == Catch::Approx(0.3));
    REQUIRE(model.tmm().nodes().get_a(10) == Catch::Approx(1.0));
    REQUIRE(model.tmm().nodes().get_type(10) == 'D');
}

TEST_CASE("capacitance: a dual-surfaced node picks up both thicknesses",
          "[conduction][builder]") {
    ThermalModel model("dual_surfaced");
    ThermalMesh mesh;
    const auto bulk = make_bulk("alu");
    mesh.set_side1_material(bulk);
    mesh.set_side2_material(bulk);
    mesh.set_side1_thick(0.05);
    mesh.set_side2_thick(0.02);
    // Both sides map to the same node, which is what makes it dual-surfaced.
    mesh.set_node1_start(7);
    mesh.set_node2_start(7);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.nodes_created == 1U);
    // The two sides add: rho * cp * (t1 + t2) * A.
    REQUIRE(model.tmm().nodes().get_C(7) == Catch::Approx(2.0 * 3.0 * 0.07));
    // Each active side counts its own face area.
    REQUIRE(model.tmm().nodes().get_a(7) == Catch::Approx(2.0));
    // The two sides are the same node, so no through-thickness conductor.
    REQUIRE(report.conductors_created == 0U);
}

TEST_CASE("capacitance: mismatched bulks are summed and reported",
          "[conduction][builder]") {
    ThermalModel model("mixed_bulk");
    ThermalMesh mesh;
    mesh.set_side1_material(
        std::make_shared<BulkMaterial>("a", 2.0, 10.0, 3.0));
    mesh.set_side2_material(
        std::make_shared<BulkMaterial>("b", 4.0, 10.0, 5.0));
    mesh.set_side1_thick(0.1);
    mesh.set_side2_thick(0.1);
    mesh.set_node1_start(1);
    mesh.set_node2_start(1);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(has_code(report, DiagnosticCode::MixedBulkOnNode));
    REQUIRE(model.tmm().nodes().get_C(1) ==
            Catch::Approx((2.0 * 3.0 * 0.1) + (4.0 * 5.0 * 0.1)));
}

TEST_CASE("capacitance: a radiative-only side keeps its node and its mass",
          "[conduction][builder]") {
    ThermalModel model("radiative_only_side");
    ThermalMesh mesh;
    const auto bulk = make_bulk("alu");
    mesh.set_side1_material(bulk);
    mesh.set_side2_material(bulk);
    mesh.set_side1_thick(0.05);
    mesh.set_side2_thick(0.05);
    mesh.set_node1_start(1);
    mesh.set_node2_start(2);
    // Side 2 radiates without conducting: it is part of the model, so it gets
    // its node and its half of the shell's mass. Only the conductors go.
    mesh.set_conductive_active_side(ActiveSide::Side1);
    mesh.set_radiative_active_side(ActiveSide::Both);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.nodes_created == 2U);
    REQUIRE(has_code(report, DiagnosticCode::InactiveSideSkipped));
    REQUIRE(model.tmm().nodes().is_node(1));
    REQUIRE(model.tmm().nodes().is_node(2));
    REQUIRE(model.tmm().nodes().get_C(2) == Catch::Approx(2.0 * 3.0 * 0.05));
    REQUIRE(model.tmm().nodes().get_a(2) == Catch::Approx(1.0));
    // Side 2 is out of the through-thickness series, so the two nodes are not
    // linked.
    REQUIRE(report.conductors_created == 0U);
}

TEST_CASE("capacitance: a side active in neither physics contributes nothing",
          "[conduction][builder]") {
    ThermalModel model("dead_side");
    ThermalMesh mesh;
    const auto bulk = make_bulk("alu");
    mesh.set_side1_material(bulk);
    mesh.set_side2_material(bulk);
    mesh.set_side1_thick(0.05);
    mesh.set_side2_thick(0.05);
    mesh.set_node1_start(1);
    mesh.set_node2_start(2);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    mesh.set_radiative_active_side(ActiveSide::Side1);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.nodes_created == 1U);
    REQUIRE(has_code(report, DiagnosticCode::InactiveSideSkipped));
    REQUIRE(model.tmm().nodes().is_node(1));
    REQUIRE_FALSE(model.tmm().nodes().is_node(2));
}

TEST_CASE("capacitance: a dual-surfaced node drops its inactive side's mass",
          "[conduction][builder]") {
    ThermalModel model("dual_surfaced_one_side_dead");
    ThermalMesh mesh;
    const auto bulk = make_bulk("alu");
    mesh.set_side1_material(bulk);
    mesh.set_side2_material(bulk);
    mesh.set_side1_thick(0.05);
    mesh.set_side2_thick(0.02);
    // One node fed by both sides, but side 2 takes part in neither physics.
    mesh.set_node1_start(7);
    mesh.set_node2_start(7);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    mesh.set_radiative_active_side(ActiveSide::Side1);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.nodes_created == 1U);
    // Side 1 alone: t2 is not there to add.
    REQUIRE(model.tmm().nodes().get_C(7) == Catch::Approx(2.0 * 3.0 * 0.05));
    REQUIRE(model.tmm().nodes().get_a(7) == Catch::Approx(1.0));
}

TEST_CASE("capacitance: a conductive-only side still builds its node",
          "[conduction][builder]") {
    ThermalModel model("conductive_only");
    ThermalMesh mesh;
    mesh.set_side1_material(make_bulk("alu"));
    mesh.set_side1_thick(0.05);
    mesh.set_node1_start(5);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    mesh.set_radiative_active_side(ActiveSide::None);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    REQUIRE(model.build_tmm_from_gmm().nodes_created == 1U);
    REQUIRE(model.tmm().nodes().get_C(5) == Catch::Approx(0.3));
}

TEST_CASE("capacitance: missing bulk keeps the node without capacitance",
          "[conduction][builder]") {
    ThermalModel model("no_bulk");
    ThermalMesh mesh;
    mesh.set_node1_start(3);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.nodes_created == 1U);
    REQUIRE(report.conductors_created == 0U);
    REQUIRE(has_code(report, DiagnosticCode::MissingBulk));
    REQUIRE(model.tmm().nodes().get_C(3) == Catch::Approx(0.0));
    REQUIRE(model.tmm().nodes().get_a(3) == Catch::Approx(1.0));
}

TEST_CASE("capacitance: zero thickness keeps the node without capacitance",
          "[conduction][builder]") {
    ThermalModel model("no_thickness");
    ThermalMesh mesh;
    mesh.set_side1_material(make_bulk("alu"));
    mesh.set_node1_start(3);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.nodes_created == 1U);
    REQUIRE(has_code(report, DiagnosticCode::ZeroThickness));
    REQUIRE(model.tmm().nodes().get_C(3) == Catch::Approx(0.0));
}

TEST_CASE("capacitance: node coordinates are the area-weighted centroid",
          "[conduction][builder]") {
    ThermalModel model("centroid");
    ThermalMesh mesh({0.0, 0.5, 1.0}, {0.0, 1.0});
    mesh.set_side1_material(make_bulk("alu"));
    mesh.set_side1_thick(0.05);
    // Both half-plates share one node, so its coordinate is the centroid of
    // the whole plate.
    mesh.set_node1_start(4);
    mesh.set_node1_step(0);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    // Placed 10 m up so the world transform is visible in the answer.
    model.gmm().add(make_plate(
        "plate", std::move(mesh),
        CoordinateTransformation::from_translation({0.0, 0.0, 10.0})));

    REQUIRE(model.build_tmm_from_gmm().nodes_created == 1U);
    REQUIRE(model.tmm().nodes().get_fx(4) == Catch::Approx(0.5));
    REQUIRE(model.tmm().nodes().get_fy(4) == Catch::Approx(0.5));
    REQUIRE(model.tmm().nodes().get_fz(4) == Catch::Approx(10.0));
}

TEST_CASE("capacitance: the initial temperature is an option",
          "[conduction][builder]") {
    ThermalModel model("initial_temperature");
    ThermalMesh mesh;
    mesh.set_side1_material(make_bulk("alu"));
    mesh.set_side1_thick(0.05);
    mesh.set_node1_start(1);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    model.gmm().add(make_plate("plate", std::move(mesh)));

    TmmBuildOptions options;
    options.initial_temperature = 293.15;
    REQUIRE(model.build_tmm_from_gmm(options).nodes_created == 1U);
    REQUIRE(model.tmm().nodes().get_T(1) == Catch::Approx(293.15));
}
