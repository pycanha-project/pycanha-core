// NOLINTBEGIN(readability-function-cognitive-complexity)
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/materials/optical_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/gebhart.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/retracer.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace {

namespace rad = pycanha::radiative;
using pycanha::gmm::ActiveSide;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::OpticalMaterial;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::solvers::Settings;
using pycanha::solvers::steady::SteadyState;
using pycanha::solvers::transient::CrankNicolson;
using Model = std::shared_ptr<pycanha::ThermalMathematicalModel>;

constexpr int space_node = 1000;
constexpr double space_temperature = 3.0;

// The emissivity of the paint: 0.2 at 200 K to 0.9 at 400 K, held outside.
constexpr double eps_low = 0.2;
constexpr double eps_high = 0.9;
constexpr double t_low = 200.0;
constexpr double t_high = 400.0;

double paint_emissivity(double temperature) {
    const double clamped = std::clamp(temperature, t_low, t_high);
    return eps_low +
           ((eps_high - eps_low) * (clamped - t_low) / (t_high - t_low));
}

void add_paint_table(pycanha::ThermalMathematicalModel& model) {
    model.parameters().add_table(
        "eps_paint",
        pycanha::LookupTable1D(Eigen::Vector2d(t_low, t_high),
                               Eigen::Vector2d(eps_low, eps_high)));
}

std::shared_ptr<OpticalMaterial> paint() {
    auto material = std::make_shared<OpticalMaterial>("paint", 0.5, 0.3);
    material->set_emissivity_ir_table("eps_paint");
    return material;
}

// Side 1 radiates with `material` and belongs to `node`; side 2 is inactive
// and has no node.
ThermalMesh one_sided(std::shared_ptr<OpticalMaterial> material,
                      pycanha::NodeNum node) {
    ThermalMesh mesh;
    mesh.set_side1_optical(std::move(material));
    mesh.set_node1_start(node);
    mesh.set_radiative_active_side(ActiveSide::Side1);
    return mesh;
}

Model make_tmm(std::span<const std::pair<int, double>> nodes) {
    auto model = std::make_shared<pycanha::ThermalMathematicalModel>("tmm");
    for (const auto& [number, temperature] : nodes) {
        pycanha::Node node(number);
        node.set_T(temperature);
        node.set_C(500.0);
        model->add_node(node);
    }
    pycanha::Node space(space_node);
    space.set_type(pycanha::BOUNDARY_NODE);
    space.set_T(space_temperature);
    model->add_node(space);
    return model;
}

// A unit plate facing +z, painted on side 1, node 1.
std::unique_ptr<GeometryModel> make_lone_plate() {
    auto gmm = std::make_unique<GeometryModel>("plate");
    gmm->add(std::make_shared<GeometryItem>(
        "plate", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        one_sided(paint(), 1)));
    return gmm;
}

rad::TraceSettings few_rays() {
    rad::TraceSettings trace;
    trace.rays_per_face = 2'000;
    trace.seed = 1;
    return trace;
}

Settings converging(int max_iterations, double tolerance) {
    Settings settings;
    settings.max_iterations = max_iterations;
    settings.tolerance = tolerance;
    return settings;
}

// The largest relative difference between the retraced couplings of a
// closed unit cube and the Gebhart method on the same faces. Side 1 faces
// inward and every wall is one node at its own temperature: floor 1,
// ceiling 2, x = 0 3, x = 1 4, y = 0 5, y = 1 6. Each wall is split into
// `divisions` x `divisions` faces, because the Gebhart method takes the
// radiosity of a face as uniform, which a whole wall is not (6 % off on the
// opposite walls). The reference takes its view factors from the first-hit
// kernel, with its own seed.
double box_gebhart_deviation(rad::Device& device, int divisions,
                             std::uint64_t rays_per_face) {
    const std::array<Rectangle, 6> walls{
        Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        Rectangle({0.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 1.0}),
        Rectangle({0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}),
        Rectangle({1.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}),
        Rectangle({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}),
        Rectangle({0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0})};
    std::vector<double> grid;
    for (int point = 0; point <= divisions; ++point) {
        grid.push_back(static_cast<double>(point) / divisions);
    }
    auto gmm = std::make_unique<GeometryModel>("box");
    const auto material = paint();
    for (std::size_t wall = 0; wall < walls.size(); ++wall) {
        ThermalMesh mesh(grid, grid);
        mesh.set_side1_optical(material);
        mesh.set_node1_start(static_cast<pycanha::NodeNum>(wall + 1));
        mesh.set_radiative_active_side(ActiveSide::Side1);
        gmm->add(std::make_shared<GeometryItem>(
            "wall_" + std::to_string(wall), walls.at(wall), std::move(mesh)));
    }
    const std::array<std::pair<int, double>, 6> nodes{{{1, 250.0},
                                                       {2, 280.0},
                                                       {3, 310.0},
                                                       {4, 340.0},
                                                       {5, 370.0},
                                                       {6, 300.0}}};
    auto model = make_tmm(nodes);
    add_paint_table(*model);

    rad::TraceSettings trace;
    trace.rays_per_face = rays_per_face;
    trace.seed = 7;
    // Only the trace at the current temperatures is compared.
    rad::RadiativeRetracer retracer(device, *gmm, model, space_node, trace,
                                    rad::AccumConfig{},
                                    std::set<rad::PatternState>{});
    retracer.initialize();
    // Closed: only rays slipping through the edges reach space.
    CHECK(retracer.statistics().space_total < 1e-3);

    rad::RadiativeScene scene(device, gmm->mesh_parts(), gmm->material_table());
    rad::VfAccumulator view_factors(scene);
    trace.seed = 11;
    scene.accumulate_vf(view_factors, trace);
    const auto& node_numbers = gmm->mesh().node_numbers;
    const std::vector<pycanha::NodeNum> face_nodes(node_numbers.begin(),
                                                   node_numbers.end());
    Eigen::VectorXd emissivity =
        Eigen::VectorXd::Ones(static_cast<Eigen::Index>(face_nodes.size()));
    for (std::size_t face = 0; face < face_nodes.size(); ++face) {
        if (face_nodes[face] != pycanha::gmm::NO_NODE) {
            const double temperature =
                nodes.at(static_cast<std::size_t>(face_nodes[face] - 1)).second;
            emissivity(static_cast<Eigen::Index>(face)) =
                paint_emissivity(temperature);
        }
    }
    const rad::SparseMatrix gebhart = rad::gebhart_node_factors(
        view_factors.result().vf, emissivity, face_nodes, scene.face_areas());

    double deviation = 0.0;
    for (int first = 0; first < 6; ++first) {
        for (int second = first + 1; second < 6; ++second) {
            const double expected = gebhart.coeff(first, second);
            const double traced =
                model->radiative_couplings().get_coupling_value(first + 1,
                                                                second + 1);
            deviation =
                std::max(deviation, std::abs(traced - expected) / expected);
        }
    }
    return deviation;
}

}  // namespace

TEST_CASE("A lone plate follows its emissivity table",
          "[radiative][gpu][retracer]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    rad::Device device = rad::Device::create();
    const auto gmm = make_lone_plate();
    const std::array<std::pair<int, double>, 1> nodes{{{1, 300.0}}};
    auto model = make_tmm(nodes);
    add_paint_table(*model);
    constexpr double dissipation = 200.0;
    model->nodes().set_qi(1, dissipation);
    // A coupling the geometry does not produce, to a node without geometry:
    // the retracer owns it too.
    pycanha::Node radiator(50);
    radiator.set_type(pycanha::BOUNDARY_NODE);
    radiator.set_T(300.0);
    model->add_node(radiator);
    model->add_radiative_coupling(1, 50, 0.1);

    auto retracer = std::make_shared<rad::RadiativeRetracer>(
        device, *gmm, model, space_node, few_rays());
    retracer->initialize();
    REQUIRE(retracer->initialized());
    REQUIRE(retracer->statistics().faces_without_node == 0);
    // No other face: everything the plate emits leaves to space.
    REQUIRE(model->radiative_couplings().get_coupling_value(1, space_node) ==
            Catch::Approx(paint_emissivity(300.0)).epsilon(1e-6));
    REQUIRE(model->radiative_couplings().get_coupling_value(1, 50) == 0.0);

    SECTION("steady state") {
        SteadyState solver(model);
        solver.set_settings(converging(100, 1e-6));
        solver.set_retracer(retracer);
        solver.initialize();
        solver.solve();
        REQUIRE(solver.converged());
        REQUIRE(solver.report().retraces > 0);
        REQUIRE(solver.report().overflow_couplings == 0);

        const double temperature = model->nodes().get_T(1);
        const double emitted =
            pycanha::STF_BOLTZ * paint_emissivity(temperature) *
            (std::pow(temperature, 4) - std::pow(space_temperature, 4));
        REQUIRE(emitted == Catch::Approx(dissipation).epsilon(1e-5));
        REQUIRE(
            model->radiative_couplings().get_coupling_value(1, space_node) ==
            Catch::Approx(paint_emissivity(temperature)).epsilon(1e-5));
    }

    SECTION("transient, against the same coupling as a formula") {
        auto reference = make_tmm(nodes);
        add_paint_table(*reference);
        reference->nodes().set_qi(1, dissipation);
        reference->add_radiative_coupling(1, space_node, 0.5);
        reference->formulas().add_formula("GR(1,1000)", "eps_paint(T1)");

        const auto run =
            [](const Model& tmm,
               const std::shared_ptr<rad::RadiativeRetracer>& with) {
                CrankNicolson solver(tmm);
                Settings settings = converging(50, 1e-6);
                settings.output_name = "plate_run";
                solver.set_settings(settings);
                solver.set_simulation_time(0.0, 2000.0, 50.0, 50.0);
                solver.set_retracer(with);
                solver.initialize();
                solver.solve();
                return solver.report();
            };
        const auto traced = run(model, retracer);
        const auto formula = run(reference, nullptr);
        REQUIRE(traced.converged());
        REQUIRE(formula.converged());
        REQUIRE(traced.retraces > 0);
        REQUIRE(std::abs(model->nodes().get_T(1) -
                         reference->nodes().get_T(1)) < 1e-4);
    }

    SECTION("a positive retrace_dT traces less") {
        const auto transient_retraces =
            [&](rad::RadiativeRetracer::RetraceThreshold threshold) {
                const std::array<std::pair<int, double>, 1> plate{{{1, 300.0}}};
                auto tmm = make_tmm(plate);
                add_paint_table(*tmm);
                tmm->nodes().set_qi(1, dissipation);
                auto tracer = std::make_shared<rad::RadiativeRetracer>(
                    device, *gmm, tmm, space_node, few_rays(),
                    rad::AccumConfig{}, std::set<rad::PatternState>{},
                    std::move(threshold));
                tracer->initialize();
                CrankNicolson solver(tmm);
                Settings settings = converging(50, 1e-3);
                settings.output_name = "threshold_run";
                solver.set_settings(settings);
                solver.set_simulation_time(0.0, 2000.0, 50.0, 50.0);
                solver.set_retracer(tracer);
                solver.initialize();
                solver.solve();
                REQUIRE(solver.converged());
                return solver.report().retraces;
            };
        const int every_change = transient_retraces(0.0);
        const int uniform = transient_retraces(5.0);
        const int per_node =
            transient_retraces(std::map<pycanha::NodeNum, double>{{1, 5.0}});
        REQUIRE(uniform < every_change);
        REQUIRE(per_node == uniform);
    }

    SECTION("a formula on a coupling the retracer writes wins") {
        model->formulas().add_formula("GR(1,50)", 0.1);
        SteadyState solver(model);
        solver.set_settings(converging(100, 1e-6));
        solver.set_retracer(retracer);
        solver.initialize();
        solver.solve();
        REQUIRE(model->radiative_couplings().get_coupling_value(1, 50) ==
                Catch::Approx(0.1));
    }
}

TEST_CASE("Retraced couplings of a closed box match the Gebhart method",
          "[radiative][gpu][retracer]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    rad::Device device = rad::Device::create();
    // The measured difference is 0.45 %: the remaining non-uniformity of the
    // radiosity on 6 x 6 faces and the noise of 10 000 rays per face. A wrong
    // emissivity on one wall moves its couplings by several percent.
    REQUIRE(box_gebhart_deviation(device, 6, 10'000) < 0.01);
}

TEST_CASE("Couplings outside the pattern overflow and are written back",
          "[radiative][gpu][retracer]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    rad::Device device = rad::Device::create();
    // Plate 1 (z = 0, facing up) is black to IR below 249 K and has an
    // emissivity of 0.8 above 251 K; plate 2 (z = 0.5, facing down) is
    // gray, 0.5. Plate 1 starts at 200 K and is heated.
    auto gmm = std::make_unique<GeometryModel>("plates");
    auto switching = std::make_shared<OpticalMaterial>("switching", 0.0, 0.3);
    switching->set_emissivity_ir_table("eps_switch");
    gmm->add(std::make_shared<GeometryItem>(
        "plate_1", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        one_sided(switching, 1)));
    gmm->add(std::make_shared<GeometryItem>(
        "plate_2", Rectangle({0.0, 0.0, 0.5}, {0.0, 1.0, 0.5}, {1.0, 0.0, 0.5}),
        one_sided(std::make_shared<OpticalMaterial>("gray", 0.5, 0.5), 2)));

    const std::array<std::pair<int, double>, 2> nodes{{{1, 200.0}, {2, 300.0}}};
    const auto run = [&](std::set<rad::PatternState> states) {
        auto model = make_tmm(nodes);
        model->parameters().add_table(
            "eps_switch", pycanha::LookupTable1D(Eigen::Vector2d(249.0, 251.0),
                                                 Eigen::Vector2d(0.0, 0.8)));
        model->nodes().set_C(1, 100.0);
        model->nodes().set_qi(1, 300.0);
        auto retracer = std::make_shared<rad::RadiativeRetracer>(
            device, *gmm, model, space_node, few_rays(), rad::AccumConfig{},
            std::move(states));
        retracer->initialize();
        CrankNicolson solver(model);
        Settings settings = converging(100, 1e-6);
        settings.output_name = "overflow_run";
        solver.set_settings(settings);
        solver.set_simulation_time(0.0, 500.0, 10.0, 10.0);
        solver.set_retracer(retracer);
        solver.initialize();
        solver.solve();
        REQUIRE(solver.converged());
        return std::make_tuple(model, retracer, solver.report(),
                               solver.initialized());
    };

    // Traced only at 200 K, plate 1 couples to nothing.
    const auto [partial, partial_retracer, partial_report,
                partial_initialized] = run({});
    REQUIRE(partial_report.overflow_couplings == 2);
    REQUIRE(partial_report.overflow_share > 0.0);
    // Written back: the solver and the retracer need initialize() again.
    REQUIRE_FALSE(partial_initialized);
    REQUIRE_FALSE(partial_retracer->initialized());
    REQUIRE(partial->radiative_couplings().get_coupling_value(1, 2) > 0.0);
    REQUIRE(partial->radiative_couplings().get_coupling_value(1, space_node) >
            0.0);

    // The HIGH state finds every pair up front.
    const auto [complete, complete_retracer, complete_report,
                complete_initialized] = run({rad::PatternState::HIGH});
    REQUIRE(complete_report.overflow_couplings == 0);
    REQUIRE(complete_initialized);
    // The overflow load is exact at convergence.
    REQUIRE(std::abs(partial->nodes().get_T(1) - complete->nodes().get_T(1)) <
            1e-4);
    REQUIRE(std::abs(partial->nodes().get_T(2) - complete->nodes().get_T(2)) <
            1e-4);
}

TEST_CASE("The retracer checks the model and the tables",
          "[radiative][gpu][retracer]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    rad::Device device = rad::Device::create();
    const auto gmm = make_lone_plate();
    const std::array<std::pair<int, double>, 1> nodes{{{1, 300.0}}};
    auto model = make_tmm(nodes);

    REQUIRE_THROWS_AS(
        std::make_shared<rad::RadiativeRetracer>(
            device, *gmm, model, space_node, few_rays(), rad::AccumConfig{},
            std::set<rad::PatternState>{}, -1.0),
        std::invalid_argument);

    SECTION("a missing table") {
        rad::RadiativeRetracer retracer(device, *gmm, model, space_node);
        REQUIRE_THROWS_AS(retracer.initialize(), std::invalid_argument);
    }
    SECTION("a missing space node") {
        add_paint_table(*model);
        rad::RadiativeRetracer retracer(device, *gmm, model, 77);
        REQUIRE_THROWS_AS(retracer.initialize(), std::invalid_argument);
    }
    SECTION("linear extrapolation") {
        model->parameters().add_table(
            "eps_paint",
            pycanha::LookupTable1D(Eigen::Vector2d(t_low, t_high),
                                   Eigen::Vector2d(eps_low, eps_high),
                                   pycanha::InterpolationMethod::Linear,
                                   pycanha::ExtrapolationMethod::Linear));
        rad::RadiativeRetracer retracer(device, *gmm, model, space_node);
        REQUIRE_THROWS_AS(retracer.initialize(), std::invalid_argument);
    }
    SECTION("IR properties adding up to more than 1") {
        add_paint_table(*model);
        auto shiny = std::make_unique<GeometryModel>("shiny");
        auto material = std::make_shared<OpticalMaterial>(
            "shiny", OpticalMaterial::Properties{0.2, 0.3, 0.0, 0.3, 0.0, 0.0});
        material->set_emissivity_ir_table("eps_paint");
        shiny->add(std::make_shared<GeometryItem>(
            "plate",
            Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
            one_sided(material, 1)));
        rad::RadiativeRetracer retracer(device, *shiny, model, space_node);
        REQUIRE_THROWS_AS(retracer.initialize(), std::invalid_argument);
    }
    SECTION("a threshold on a node that is not in the model") {
        add_paint_table(*model);
        rad::RadiativeRetracer retracer(
            device, *gmm, model, space_node, few_rays(), rad::AccumConfig{},
            std::set<rad::PatternState>{},
            std::map<pycanha::NodeNum, double>{{42, 1.0}});
        REQUIRE_THROWS_AS(retracer.initialize(), std::invalid_argument);
    }
    SECTION("a face without a node is counted") {
        add_paint_table(*model);
        auto orphan = make_lone_plate();
        orphan->add(std::make_shared<GeometryItem>(
            "orphan",
            Rectangle({0.0, 0.0, 2.0}, {1.0, 0.0, 2.0}, {0.0, 1.0, 2.0}),
            one_sided(paint(), pycanha::gmm::NO_NODE)));
        rad::RadiativeRetracer retracer(device, *orphan, model, space_node,
                                        few_rays());
        retracer.initialize();
        REQUIRE(retracer.statistics().faces_without_node == 1);
    }
    SECTION("a solve after a change of the model's structure") {
        add_paint_table(*model);
        auto retracer = std::make_shared<rad::RadiativeRetracer>(
            device, *gmm, model, space_node, few_rays());
        retracer->initialize();
        const pycanha::Node extra(5);
        model->add_node(extra);
        SteadyState solver(model);
        solver.set_retracer(retracer);
        solver.initialize();
        REQUIRE_THROWS_AS(solver.solve(), std::runtime_error);
    }
}

// NOLINTEND(readability-function-cognitive-complexity)
