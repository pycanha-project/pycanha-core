#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/conduction/builder.hpp"
#include "pycanha-core/conduction/links.hpp"
#include "pycanha-core/conduction/network_part.hpp"
#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"

// Catch2's assertion macros expand to branches, which the complexity check
// counts against every test case.
// NOLINTBEGIN(readability-function-cognitive-complexity)

namespace {

using pycanha::NodeNum;
using pycanha::ThermalMathematicalModel;
using pycanha::ThermalModel;
using pycanha::conduction::DiagnosticCode;
using pycanha::conduction::NetworkPart;
using pycanha::conduction::TmmBuildOptions;
using pycanha::conduction::TmmBuildReport;
using pycanha::gmm::ActiveSide;
using pycanha::gmm::BulkMaterial;
using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cube;
using pycanha::gmm::Cylinder;
using pycanha::gmm::GeometryGroup;
using pycanha::gmm::GeometryGroupCutted;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Sphere;
using pycanha::gmm::ThermalMesh;

using PairValues = std::map<std::pair<NodeNum, NodeNum>, double>;

constexpr double pi = std::numbers::pi;

[[nodiscard]] std::shared_ptr<BulkMaterial> aluminium() {
    static const auto material =
        std::make_shared<BulkMaterial>("alu", 2700.0, 160.0, 900.0);
    return material;
}

[[nodiscard]] std::vector<double> even_cuts(int count) {
    std::vector<double> cuts;
    for (int cut = 0; cut <= count; ++cut) {
        cuts.push_back(static_cast<double>(cut) / count);
    }
    return cuts;
}

// Both sides on one node per face pair, conducting.
[[nodiscard]] ThermalMesh shared_shell(int dir1, int dir2, NodeNum start) {
    ThermalMesh mesh(even_cuts(dir1), even_cuts(dir2));
    mesh.set_side1_material(aluminium());
    mesh.set_side2_material(aluminium());
    mesh.set_side1_thick(0.001);
    mesh.set_side2_thick(0.001);
    mesh.set_node1_start(start);
    mesh.set_node1_step(1);
    mesh.set_node2_start(start);
    mesh.set_node2_step(1);
    return mesh;
}

[[nodiscard]] std::shared_ptr<GeometryItem> unit_plate(
    const std::string& name, ThermalMesh mesh,
    CoordinateTransformation transform = {}) {
    return std::make_shared<GeometryItem>(
        name, Rectangle({0, 0, 0}, {1, 0, 0}, {0, 1, 0}), std::move(mesh),
        std::move(transform));
}

[[nodiscard]] PairValues couplings(ThermalMathematicalModel& tmm) {
    PairValues pairs;
    const auto arrays = tmm.conductive_couplings().to_arrays();
    for (std::size_t entry = 0; entry < arrays.values.size(); ++entry) {
        pairs[{std::min(arrays.node_1[entry], arrays.node_2[entry]),
               std::max(arrays.node_1[entry], arrays.node_2[entry])}] =
            arrays.values[entry];
    }
    return pairs;
}

[[nodiscard]] bool has_code(const TmmBuildReport& report, DiagnosticCode code) {
    return std::ranges::any_of(report.diagnostics, [code](const auto& entry) {
        return entry.code == code;
    });
}

// The old builder's in-plane sums, written out independently: every link of
// intra_primitive_links, in order, added onto its node pair.
[[nodiscard]] PairValues reference_links(const GeometryItem& item) {
    PairValues pairs;
    const ThermalMesh& mesh = item.thermal_mesh();
    const auto dir1 =
        static_cast<pycanha::MeshIndex>(mesh.get_dir1_mesh().size() - 1U);
    for (const auto& link : pycanha::conduction::intra_primitive_links(
             item.primitive(), mesh, TmmBuildOptions{})) {
        const auto node_of = [&](pycanha::MeshIndex face_pair) {
            return mesh.node_of(face_pair % dir1, face_pair / dir1, link.side);
        };
        const NodeNum first = node_of(link.face_pair_a);
        const NodeNum second = node_of(link.face_pair_b);
        if (first != second) {
            pairs[{std::min(first, second), std::max(first, second)}] +=
                link.conductance;
        }
    }
    return pairs;
}

}  // namespace

TEST_CASE("network part: in-plane conductances are the parametric sums",
          "[conduction][network_part]") {
    // Shared node numbers: no through-thickness conductor, so every coupling
    // is an in-plane sum, and must be bit-identical to the reference.
    ThermalModel model("sphere");
    auto sphere = std::make_shared<GeometryItem>(
        "sphere",
        Sphere({0, 0, 0}, {0, 0, 1}, {1, 0, 0}, 0.7, -0.7, 0.7, 0.0, 2.0 * pi),
        shared_shell(12, 8, 1),
        CoordinateTransformation::from_euler({1, 2, 3}, {0.3, 0.2, 0.1}));
    model.gmm().add(sphere);
    static_cast<void>(model.build_tmm_from_gmm());
    REQUIRE(couplings(model.tmm()) == reference_links(*sphere));

    // Capacities are exact: they add up to rho * c * t over the whole surface.
    const std::vector<double> capacities =
        model.tmm().nodes().get_values(pycanha::NodeAttribute::C);
    const double capacity =
        std::accumulate(capacities.begin(), capacities.end(), 0.0);
    const double surface = 4.0 * pi * 0.7 * 0.7;
    REQUIRE(std::abs(capacity - (2700.0 * 900.0 * 0.002 * surface)) <=
            1e-12 * capacity);
}

TEST_CASE("network part: one item's part is what the builder commits",
          "[conduction][network_part]") {
    const auto transform =
        CoordinateTransformation::from_euler({0.5, -1.0, 2.0}, {0.4, 0.1, 0.9});
    ThermalMesh mesh = shared_shell(6, 4, 100);
    mesh.set_node2_start(500);  // separate sides: through-thickness too

    ThermalModel built("built");
    built.gmm().add(unit_plate("plate", mesh, transform));
    const TmmBuildReport report = built.build_tmm_from_gmm();

    // The part of the same item, placed by hand, committed on its own.
    const auto item = unit_plate("plate", mesh);
    const NetworkPart part =
        pycanha::conduction::build_network_part(*item, transform);
    REQUIRE(std::ranges::is_sorted(part.node_numbers));
    REQUIRE(part.node_numbers.size() == 48U);
    ThermalMathematicalModel committed("committed");
    const std::vector<NetworkPart> parts{part};
    const TmmBuildReport direct =
        pycanha::conduction::commit_network_parts(committed, parts);
    REQUIRE(direct.nodes_created == report.nodes_created);
    REQUIRE(couplings(committed) == couplings(built.tmm()));
    for (const NodeNum node : part.node_numbers) {
        REQUIRE(committed.nodes().get_C(node) ==
                built.tmm().nodes().get_C(node));
        REQUIRE(committed.nodes().get_fz(node) ==
                built.tmm().nodes().get_fz(node));
    }

    // Moving the item changes the positions, nothing else.
    const NetworkPart still = pycanha::conduction::build_network_part(
        *item, CoordinateTransformation{});
    REQUIRE(still.thermal_capacity == part.thermal_capacity);
    REQUIRE(still.conductance == part.conductance);
    REQUIRE(still.coupling_index_1 == part.coupling_index_1);
    const pycanha::Point3D moved = transform.apply(
        {still.position_x[3], still.position_y[3], still.position_z[3]});
    REQUIRE(std::abs(moved.x() - part.position_x[3]) <= 1e-14);
    REQUIRE(std::abs(moved.z() - part.position_z[3]) <= 1e-14);
}

TEST_CASE("network part: parts sharing nodes are merged by summing",
          "[conduction][network_part]") {
    // Two plates numbered over the same range, one of them nested and moved:
    // the shared nodes sum capacity and conductance, and their position is
    // the weighted mean of both.
    ThermalModel model("shared");
    model.gmm().add(unit_plate("a", shared_shell(3, 3, 1)));
    auto group = std::make_shared<GeometryGroup>(
        "group",
        std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{
            unit_plate("b", shared_shell(3, 3, 1),
                       CoordinateTransformation::from_translation({0, 0, 1}))},
        CoordinateTransformation::from_translation({0, 0, 1}));
    model.gmm().add(group);
    const TmmBuildReport report = model.build_tmm_from_gmm();
    REQUIRE(report.items_processed == 2U);
    REQUIRE(report.nodes_created == 9U);

    ThermalModel single("single");
    single.gmm().add(unit_plate("a", shared_shell(3, 3, 1)));
    static_cast<void>(single.build_tmm_from_gmm());
    for (const auto& [pair, value] : couplings(single.tmm())) {
        REQUIRE(couplings(model.tmm()).at(pair) == value + value);
    }
    REQUIRE(model.tmm().nodes().get_C(5) ==
            Catch::Approx(2.0 * single.tmm().nodes().get_C(5)));
    REQUIRE(model.tmm().nodes().get_fz(5) == Catch::Approx(1.0));
}

TEST_CASE("network part: many items give the same network in any order",
          "[conduction][network_part]") {
    // Built in parallel over the items; committing the parts built one by one
    // gives the same model.
    ThermalModel model("many");
    std::vector<std::shared_ptr<GeometryItem>> items;
    for (int plate = 0; plate < 24; ++plate) {
        items.push_back(
            unit_plate("plate_" + std::to_string(plate),
                       shared_shell(5, 4, 1 + (plate * 100)),
                       CoordinateTransformation::from_translation(
                           {static_cast<double>(plate), 0.0, 0.0})));
        model.gmm().add(items.back());
    }
    static_cast<void>(model.build_tmm_from_gmm());

    std::vector<NetworkPart> parts(items.size());
    std::ranges::transform(std::views::reverse(items), parts.begin(),
                           [](const auto& item) {
                               return pycanha::conduction::build_network_part(
                                   *item, item->transform());
                           });
    ThermalMathematicalModel serial("serial");
    static_cast<void>(pycanha::conduction::commit_network_parts(serial, parts));
    REQUIRE(couplings(serial) == couplings(model.tmm()));
    REQUIRE(serial.nodes().node_numbers() ==
            model.tmm().nodes().node_numbers());
    REQUIRE(serial.nodes().C_vector == model.tmm().nodes().C_vector);
}

TEST_CASE("cut items: whole face pairs cut away leave the rest untouched",
          "[conduction][network_part][cut]") {
    // A 4 x 4 plate and a box through its central 2 x 2 face pairs.
    const auto make_model = [](bool with_hole) {
        auto model = std::make_unique<ThermalModel>("hole");
        auto plate = unit_plate("plate", shared_shell(4, 4, 1));
        if (!with_hole) {
            model->gmm().add(plate);
            return model;
        }
        auto box = std::make_shared<GeometryItem>(
            "box", Cube({0.5, 0.5, 0.0}, {0.5, 0.5, 1.0}), ThermalMesh{});
        model->gmm().add(std::make_shared<GeometryGroupCutted>(
            "cut", std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{plate},
            std::vector<std::shared_ptr<GeometryItem>>{box}));
        return model;
    };
    auto whole = make_model(false);
    auto holed = make_model(true);
    static_cast<void>(whole->build_tmm_from_gmm());
    const TmmBuildReport report = holed->build_tmm_from_gmm();

    REQUIRE(report.face_pairs_removed == 4U);
    REQUIRE(report.face_pairs_cut == 0U);
    const std::vector<NodeNum> removed{6, 7, 10, 11};
    for (const NodeNum node : whole->tmm().nodes().node_numbers()) {
        const bool gone = std::ranges::find(removed, node) != removed.end();
        REQUIRE(holed->tmm().nodes().is_node(node) == !gone);
        if (!gone) {
            REQUIRE(
                holed->tmm().nodes().get_C(node) ==
                Catch::Approx(whole->tmm().nodes().get_C(node)).epsilon(1e-12));
        }
    }
    for (const auto& [pair, value] : couplings(whole->tmm())) {
        const bool touches =
            std::ranges::find(removed, pair.first) != removed.end() ||
            std::ranges::find(removed, pair.second) != removed.end();
        REQUIRE(couplings(holed->tmm()).contains(pair) == !touches);
    }
}

TEST_CASE("cut items: a hole through face pairs scales and isolates them",
          "[conduction][network_part][cut]") {
    // A cylinder of radius 0.3 through the middle of a 5 x 5 plate.
    const auto build = [](int chained) {
        auto model = std::make_unique<ThermalModel>("round_hole");
        std::shared_ptr<pycanha::gmm::Geometry> target =
            unit_plate("plate", shared_shell(5, 5, 1));
        auto cutter = std::make_shared<GeometryItem>(
            "drill",
            Cylinder({0.5, 0.5, -1.0}, {0.5, 0.5, 1.0}, {0.8, 0.5, -1.0}, 0.3,
                     0.0, 2.0 * pi),
            ThermalMesh{});
        target = std::make_shared<GeometryGroupCutted>(
            "cut", std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{target},
            std::vector<std::shared_ptr<GeometryItem>>{cutter});
        if (chained != 0) {
            // A second cut of the cut group: a corner box.
            auto corner = std::make_shared<GeometryItem>(
                "corner", Cube({0.0, 0.0, 0.0}, {0.2, 0.2, 1.0}),
                ThermalMesh{});
            target = std::make_shared<GeometryGroupCutted>(
                "cut_again",
                std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{target},
                std::vector<std::shared_ptr<GeometryItem>>{corner});
        }
        model->gmm().add(target);
        return model;
    };

    for (const int chained : {0, 1}) {
        INFO("chained " << chained);
        auto model = build(chained);
        const TmmBuildReport report = model->build_tmm_from_gmm();
        REQUIRE(has_code(report, DiagnosticCode::CutFacePairs));
        REQUIRE(report.face_pairs_cut > 0U);
        REQUIRE(report.links_removed > 0U);
        // Every face pair of a single-sided numbering keeps its node, so the
        // cut ones are isolated in-plane until their correction exists.
        REQUIRE(has_code(report, DiagnosticCode::UncoupledNodes));

        const double face_capacity = 2700.0 * 900.0 * 0.002 * 0.04;
        double total = 0.0;
        for (const NodeNum node : model->tmm().nodes().node_numbers()) {
            const double capacity = model->tmm().nodes().get_C(node);
            REQUIRE(capacity <= face_capacity * (1.0 + 1e-12));
            total += capacity;
        }
        // The removed area is the triangulated hole (and corner).
        const double hole = (pi * 0.3 * 0.3) + (chained != 0 ? 0.01 : 0.0);
        REQUIRE(total > 2700.0 * 900.0 * 0.002 * (1.0 - hole));
        REQUIRE(total < 2700.0 * 900.0 * 0.002 * (1.0 - (0.95 * hole)));
        // An uncut corner face pair is exact.
        REQUIRE(model->tmm().nodes().get_C(chained != 0 ? 5 : 1) ==
                Catch::Approx(face_capacity).epsilon(1e-12));

        // The node area is the cut triangulation's, both faces of each face
        // pair counted, each 1 mm thick.
        static_cast<void>(pycanha::conduction::assign_node_areas(*model));
        const std::vector<double> areas =
            model->tmm().nodes().get_values(pycanha::NodeAttribute::A);
        const double area = std::accumulate(areas.begin(), areas.end(), 0.0);
        REQUIRE(area * 2700.0 * 900.0 * 0.001 == Catch::Approx(total));
    }
}

TEST_CASE("node areas are the triangulated ones, on demand",
          "[conduction][network_part]") {
    ThermalModel model("areas");
    ThermalMesh mesh = shared_shell(8, 3, 1);
    mesh.set_radiative_active_side(ActiveSide::Side1);
    mesh.set_conductive_active_side(ActiveSide::Side1);  // side 2 inactive
    model.gmm().add(std::make_shared<GeometryItem>(
        "tube", Cylinder({0, 0, 0}, {0, 0, 1}, {1, 0, 0}, 0.5, 0.0, 2.0 * pi),
        std::move(mesh)));
    static_cast<void>(model.build_tmm_from_gmm());
    REQUIRE(model.tmm().nodes().get_a(1) == 0.0);

    const pycanha::BulkReport report =
        pycanha::conduction::assign_node_areas(model);
    REQUIRE(report.accepted == 24U);
    REQUIRE(report.rejected == 0U);
    const std::vector<double> areas =
        model.tmm().nodes().get_values(pycanha::NodeAttribute::A);
    const double area = std::accumulate(areas.begin(), areas.end(), 0.0);
    // One side only, and below the exact area by the tessellation error.
    const double exact = 2.0 * pi * 0.5;
    REQUIRE(area < exact);
    REQUIRE(area > 0.99 * exact);
}

// NOLINTEND(readability-function-cognitive-complexity)
