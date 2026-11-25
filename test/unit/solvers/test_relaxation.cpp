#include <Eigen/Core>
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "solver_test_models.hpp"

namespace {

using pycanha::NodeNum;
using pycanha::ThermalMathematicalModel;
using pycanha::solvers::RelaxationMode;
using pycanha::solvers::ResolvedSettings;
using pycanha::solvers::Settings;
using pycanha::solvers::SettingValue;
using pycanha::solvers::SolverReport;
using pycanha::solvers::SolverType;
using pycanha::solvers::steady::SteadyState;
using pycanha::solvers::transient::CrankNicolson;
using Model = std::shared_ptr<ThermalMathematicalModel>;

// A coating whose emissivity rises from 0.15 at 250 K to 0.75 at 350 K, held
// outside. One node of it with 115 W radiating to a 3 K boundary settles near
// 280 K, where each plain iteration multiplies the temperature error by
// -(T/4) (deps/dT)/eps = -1.27: it never converges.
constexpr double eps_low = 0.15;
constexpr double eps_high = 0.75;
constexpr double t_low = 250.0;
constexpr double t_high = 350.0;
constexpr double dissipation = 115.0;
constexpr double sink_temperature = 3.0;
constexpr NodeNum node = 1;
constexpr NodeNum sink = 2;
// 1 mm of aluminium over 1 m^2 [J/K].
constexpr double plate_capacity = 2430.0;

double emissivity(double temperature) {
    const double clamped = std::clamp(temperature, t_low, t_high);
    return eps_low +
           ((eps_high - eps_low) * (clamped - t_low) / (t_high - t_low));
}

double net_heat(double temperature) {
    return dissipation -
           (pycanha::STF_BOLTZ * emissivity(temperature) *
            (std::pow(temperature, 4) - std::pow(sink_temperature, 4)));
}

// The temperature where the net heat is `target`, by bisection (the net heat
// falls with the temperature).
double temperature_at(double target) {
    double low = 100.0;
    double high = 500.0;
    constexpr int halvings = 200;
    for (int step = 0; step < halvings; ++step) {
        const double middle = 0.5 * (low + high);
        if (net_heat(middle) > target) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return 0.5 * (low + high);
}

double steady_root() { return temperature_at(0.0); }

// One diffusive node of capacity `capacity` at 300 K, its radiative coupling
// to the sink a formula of its temperature through a table.
Model make_coated_node(double capacity) {
    auto model = std::make_shared<ThermalMathematicalModel>("coated");
    const std::vector<NodeNum> diffusive{node};
    const std::vector<double> t_diffusive{300.0};
    const std::vector<double> c_diffusive{capacity};
    model->nodes().add_nodes({.type = 'D',
                              .numbers = diffusive,
                              .temperature = t_diffusive,
                              .capacity = c_diffusive});
    const std::vector<NodeNum> boundary{sink};
    const std::vector<double> t_boundary{sink_temperature};
    model->nodes().add_nodes(
        {.type = 'B', .numbers = boundary, .temperature = t_boundary});
    model->nodes().set_qi(node, dissipation);
    model->parameters().add_table(
        "eps_coating",
        pycanha::LookupTable1D(Eigen::Vector2d(t_low, t_high),
                               Eigen::Vector2d(eps_low, eps_high)));
    model->add_radiative_coupling(node, sink, eps_low);
    model->formulas().add_formula("GR(1,2)", "eps_coating(T1)");
    return model;
}

Settings relaxed(RelaxationMode mode, double factor, int max_iterations,
                 double tolerance) {
    Settings settings;
    settings.relaxation_mode = mode;
    settings.relaxation = factor;
    settings.max_iterations = max_iterations;
    settings.tolerance = tolerance;
    return settings;
}

SolverReport solve_steady(const Model& model, const Settings& settings) {
    SteadyState solver(model);
    solver.set_settings(settings);
    solver.initialize();
    solver.solve();
    return solver.report();
}

// Three time steps of `dt` seconds of a coated node; the report and the final
// temperature.
std::pair<SolverReport, double> coated_transient(double capacity, double dt,
                                                 RelaxationMode mode) {
    constexpr int steps = 3;
    auto model = make_coated_node(capacity);
    CrankNicolson solver(model);
    Settings settings = relaxed(mode, 1.0, 50, 1e-6);
    settings.output_name = "relaxation_run";
    solver.set_settings(settings);
    const double end = dt * steps;
    solver.set_simulation_time(0.0, end, dt, end);
    solver.initialize();
    solver.solve();
    return {solver.report(), model->nodes().get_T(node)};
}

ResolvedSettings resolve_for(const Settings& settings, SolverType type) {
    return pycanha::solvers::resolve(
        settings, {.solver_type = type, .has_radiation = false});
}

bool not_applied(const ResolvedSettings& resolved, const std::string& name) {
    return std::ranges::any_of(resolved.not_applied, [&](const auto& ignored) {
        return ignored.name == name;
    });
}

}  // namespace

TEST_CASE("Relaxation defaults to the adaptive mode starting at 1",
          "[solver][settings][relaxation]") {
    const ResolvedSettings defaults =
        resolve_for(Settings{}, SolverType::STEADY_STATE);
    REQUIRE(defaults.relaxation == 1.0);
    REQUIRE(defaults.relaxation_mode == RelaxationMode::ADAPTIVE);
}

TEST_CASE("Relaxation factors outside (0, 1] are refused",
          "[solver][settings][relaxation]") {
    Settings settings;
    REQUIRE_THROWS_AS(settings.set("relaxation", SettingValue{0.0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(settings.set("relaxation", SettingValue{-0.1}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(settings.set("relaxation", SettingValue{1.1}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(settings.set("relaxation", SettingValue{std::nan("")}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        settings.set("relaxation_mode", SettingValue{std::string{"AITKEN"}}),
        std::invalid_argument);
    REQUIRE_FALSE(settings.relaxation.has_value());
}

TEST_CASE("Relaxation settings by name", "[solver][settings][relaxation]") {
    Settings settings;
    settings.set("relaxation", SettingValue{1.0});
    settings.set("relaxation", SettingValue{0.3});
    settings.set("relaxation_mode", SettingValue{std::string{"ADAPTIVE"}});
    REQUIRE(settings.relaxation == 0.3);
    REQUIRE(settings.relaxation_mode == RelaxationMode::ADAPTIVE);
    REQUIRE(Settings::from_dict(settings.to_dict()) == settings);
}

TEST_CASE("Relaxation applies to the steady state and Crank-Nicolson only",
          "[solver][settings][relaxation]") {
    Settings settings;
    settings.relaxation = 0.5;
    settings.relaxation_mode = RelaxationMode::ADAPTIVE;
    settings.time_step = 1.0;
    // The sensitivities of the Jacobian propagation follow the plain
    // iteration.
    const ResolvedSettings jacobian =
        resolve_for(settings, SolverType::JACOBIAN_PROPAGATION);
    REQUIRE_FALSE(jacobian.applies("relaxation"));
    REQUIRE_FALSE(jacobian.applies("relaxation_mode"));
    REQUIRE(not_applied(jacobian, "relaxation"));
    const ResolvedSettings transient =
        resolve_for(settings, SolverType::CRANK_NICOLSON);
    REQUIRE(transient.applies("relaxation"));
    REQUIRE(transient.relaxation_mode == RelaxationMode::ADAPTIVE);
}

TEST_CASE("The plain steady state of a steep coating oscillates",
          "[solver][steady][relaxation]") {
    auto model = make_coated_node(1.0);
    const SolverReport report =
        solve_steady(model, relaxed(RelaxationMode::FIXED, 1.0, 60, 1e-3));
    REQUIRE_FALSE(report.converged());
    REQUIRE(report.min_relaxation == 1.0);
    REQUIRE_FALSE(report.to_string().contains("relaxation"));
}

TEST_CASE("A fixed relaxation makes the steep steady state converge",
          "[solver][steady][relaxation]") {
    // The error factor becomes 1 + 0.5 (-1.27 - 1) = -0.13.
    auto model = make_coated_node(1.0);
    const SolverReport report =
        solve_steady(model, relaxed(RelaxationMode::FIXED, 0.5, 60, 1e-8));
    REQUIRE(report.converged());
    REQUIRE(model->nodes().get_T(node) ==
            Catch::Approx(steady_root()).margin(1e-6));
    REQUIRE(report.min_relaxation == 0.5);
    REQUIRE(report.last_relaxation == 0.5);
}

TEST_CASE("The adaptive relaxation finds its factor",
          "[solver][steady][relaxation]") {
    auto fixed = make_coated_node(1.0);
    const SolverReport by_fixed =
        solve_steady(fixed, relaxed(RelaxationMode::FIXED, 0.5, 60, 1e-8));
    auto adaptive = make_coated_node(1.0);
    const SolverReport by_adaptive = solve_steady(
        adaptive, relaxed(RelaxationMode::ADAPTIVE, 1.0, 60, 1e-8));
    REQUIRE(by_adaptive.converged());
    REQUIRE(adaptive->nodes().get_T(node) ==
            Catch::Approx(steady_root()).margin(1e-6));
    REQUIRE(by_adaptive.iterations < by_fixed.iterations);
    REQUIRE(by_adaptive.min_relaxation < 1.0);
    REQUIRE(by_adaptive.to_string().contains("relaxation"));
    REQUIRE(std::get<double>(by_adaptive.to_dict().at("min_relaxation")) ==
            by_adaptive.min_relaxation);
}

TEST_CASE("The adaptive relaxation leaves a converging steady state as it is",
          "[solver][steady][relaxation]") {
    constexpr double tolerance = 1e-9;
    auto plain = pycanha::test_models::make_small_radiative_model();
    auto adaptive = pycanha::test_models::make_small_radiative_model();
    const SolverReport by_plain = solve_steady(
        plain, relaxed(RelaxationMode::FIXED, 1.0, 100, tolerance));
    const SolverReport by_adaptive = solve_steady(
        adaptive, relaxed(RelaxationMode::ADAPTIVE, 1.0, 100, tolerance));
    REQUIRE(by_plain.converged());
    REQUIRE(by_adaptive.converged());
    REQUIRE(by_adaptive.iterations == by_plain.iterations);
    for (const NodeNum number : {10, 15, 20, 25}) {
        INFO(number);
        REQUIRE(adaptive->nodes().get_T(number) ==
                Catch::Approx(plain->nodes().get_T(number)).margin(1e-8));
    }
}

TEST_CASE("A fixed relaxation slows a converging steady state",
          "[solver][steady][relaxation]") {
    // The radiation is linearised exactly (Newton), so the plain iteration
    // converges quadratically; a fixed factor of 0.5 makes it linear.
    constexpr double tolerance = 1e-9;
    auto plain = pycanha::test_models::make_small_radiative_model();
    auto damped = pycanha::test_models::make_small_radiative_model();
    const SolverReport by_plain = solve_steady(
        plain, relaxed(RelaxationMode::FIXED, 1.0, 100, tolerance));
    const SolverReport by_damped = solve_steady(
        damped, relaxed(RelaxationMode::FIXED, 0.5, 100, tolerance));
    REQUIRE(by_damped.converged());
    REQUIRE(by_damped.iterations > by_plain.iterations + 2);
    REQUIRE(damped->nodes().get_T(15) ==
            Catch::Approx(plain->nodes().get_T(15)).margin(1e-8));
}

TEST_CASE("Relaxation with reuse_factors", "[solver][steady][relaxation]") {
    if (!pycanha::MKL_ENABLED) {
        SUCCEED("reuse_factors needs MKL");
        return;
    }
    auto model = make_coated_node(1.0);
    Settings settings = relaxed(RelaxationMode::ADAPTIVE, 1.0, 60, 1e-8);
    settings.reuse_factors = true;
    REQUIRE(solve_steady(model, settings).converged());
    REQUIRE(model->nodes().get_T(node) ==
            Catch::Approx(steady_root()).margin(1e-6));
}

// In a time step the capacity damps the error factor to
// -1.27 / (1 + 2 C / (dt G)), G = 4 sigma eps T^3.
TEST_CASE("Relaxation converges the time steps of a node without capacity",
          "[solver][transient][relaxation]") {
    const auto [plain, plain_t] =
        coated_transient(0.0, 10.0, RelaxationMode::FIXED);
    const auto [adaptive, adaptive_t] =
        coated_transient(0.0, 10.0, RelaxationMode::ADAPTIVE);
    REQUIRE(plain.nonconverged_steps > 0);
    REQUIRE(adaptive.converged());
    // Each step converges, but the step is the trapezoidal rule on a node
    // whose capacity is only capacity_delta (1e-7 J/K): the heat at the two
    // ends of the step cancels, f(T^{n+1}) = -f(T^n), so from 300 K the node
    // alternates between 300 K and the temperature where f = -f(300 K)
    // instead of settling.
    const double alternate = temperature_at(-net_heat(300.0));
    REQUIRE(adaptive_t == Catch::Approx(alternate).margin(1e-4));
    REQUIRE(std::abs(alternate - steady_root()) > 1.0);
}

TEST_CASE("Relaxation converges long time steps",
          "[solver][transient][relaxation]") {
    // dt = 3e4 s on 1 mm of aluminium: factor -1.16.
    const auto [plain, plain_t] =
        coated_transient(plate_capacity, 3e4, RelaxationMode::FIXED);
    const auto [adaptive, adaptive_t] =
        coated_transient(plate_capacity, 3e4, RelaxationMode::ADAPTIVE);
    REQUIRE(plain.nonconverged_steps > 0);
    REQUIRE(adaptive.converged());
    REQUIRE(adaptive.min_relaxation < 1.0);
}

TEST_CASE("Relaxation leaves short time steps as they are",
          "[solver][transient][relaxation]") {
    // dt = 10 s: factor -0.004.
    const auto [plain, plain_t] =
        coated_transient(plate_capacity, 10.0, RelaxationMode::FIXED);
    const auto [adaptive, adaptive_t] =
        coated_transient(plate_capacity, 10.0, RelaxationMode::ADAPTIVE);
    REQUIRE(plain.converged());
    REQUIRE(adaptive.converged());
    REQUIRE(adaptive.iterations == plain.iterations);
    REQUIRE(adaptive_t == plain_t);
}
