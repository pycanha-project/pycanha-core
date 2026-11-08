#include <Eigen/Core>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <memory>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "solver_test_models.hpp"

namespace {

using pycanha::NodeNum;
using pycanha::ThermalMathematicalModel;
using pycanha::solvers::CapacityScheme;
using pycanha::solvers::Settings;
using pycanha::solvers::transient::CrankNicolson;

// C(T) = a + b T [J/K], the capacity of every node of these tests.
constexpr double cap_a = 27.5;
constexpr double cap_b = 0.075;
constexpr double conductance = 0.5;

double capacity(double t) { return cap_a + (cap_b * t); }

// The integral of C dT from 0 [J].
double enthalpy(double t) { return (cap_a * t) + (0.5 * cap_b * t * t); }

Settings converged_settings(CapacityScheme scheme) {
    Settings settings;
    settings.capacity_scheme = scheme;
    settings.tolerance = 1e-12;
    settings.max_iterations = 100;
    return settings;
}

// One diffusive node at t0 coupled to a boundary at tb, its capacity a
// formula of its temperature.
std::shared_ptr<ThermalMathematicalModel> make_cooling_node(double t0,
                                                            double tb) {
    auto model = std::make_shared<ThermalMathematicalModel>("cooling");
    const std::vector<NodeNum> diffusive{1};
    const std::vector<double> t_diffusive{t0};
    const std::vector<double> c_diffusive{capacity(t0)};
    model->nodes().add_nodes({.type = 'D',
                              .numbers = diffusive,
                              .temperature = t_diffusive,
                              .capacity = c_diffusive});
    const std::vector<NodeNum> boundary{2};
    const std::vector<double> t_boundary{tb};
    model->nodes().add_nodes(
        {.type = 'B', .numbers = boundary, .temperature = t_boundary});
    model->add_conductive_coupling(1, 2, conductance);
    model->formulas().add_formula("C1", "27.5 + 0.075 * T1");
    return model;
}

// The exact temperature of make_cooling_node() at `time`: C(T) dT/dt =
// G (tb - T) integrates to G t = -(a + b tb) ln((tb - T)/(tb - t0)) -
// b (T - t0), solved for T by bisection.
double exact_cooling(double t0, double tb, double time) {
    const auto elapsed = [&](double t) {
        return ((-(cap_a + (cap_b * tb)) * std::log((tb - t) / (tb - t0))) -
                (cap_b * (t - t0))) /
               conductance;
    };
    double near = t0;
    double far = tb;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (near + far);
        (elapsed(mid) < time ? near : far) = mid;
    }
    return 0.5 * (near + far);
}

double cooling_error(CapacityScheme scheme, double dt) {
    constexpr double t0 = 300.0;
    constexpr double tb = 200.0;
    constexpr double end = 100.0;
    const auto model = make_cooling_node(t0, tb);
    CrankNicolson solver(model);
    solver.set_settings(converged_settings(scheme));
    solver.set_simulation_time(0.0, end, dt, end);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    return model->nodes().get_T(1) - exact_cooling(t0, tb, end);
}

}  // namespace

TEST_CASE("A temperature-dependent capacity is integrated to second order",
          "[solver][transient][capacity]") {
    const auto scheme = GENERATE(CapacityScheme::AVERAGE_RATE,
                                 CapacityScheme::AVERAGE_CAPACITY);
    INFO(pycanha::solvers::to_string(scheme));
    const double coarse = cooling_error(scheme, 2.0);
    const double fine = cooling_error(scheme, 1.0);
    // Integrating d(C T)/dt instead of C dT/dt leaves 13 K here.
    REQUIRE(std::abs(fine) < 1e-3);
    REQUIRE(std::abs(coarse / fine) > 3.5);
    REQUIRE(std::abs(coarse / fine) < 4.5);
}

TEST_CASE("AVERAGE_CAPACITY conserves the energy of a capacity linear in T",
          "[solver][transient][capacity]") {
    // Two nodes that only see each other: the enthalpy sum is constant.
    const auto run = [](CapacityScheme scheme) {
        auto model = std::make_shared<ThermalMathematicalModel>("closed");
        const std::vector<NodeNum> diffusive{1, 2};
        const std::vector<double> t0{350.0, 150.0};
        const std::vector<double> c0{capacity(t0[0]), capacity(t0[1])};
        model->nodes().add_nodes({.type = 'D',
                                  .numbers = diffusive,
                                  .temperature = t0,
                                  .capacity = c0});
        const std::vector<NodeNum> boundary{3};
        const std::vector<double> t_boundary{300.0};
        model->nodes().add_nodes(
            {.type = 'B', .numbers = boundary, .temperature = t_boundary});
        model->add_conductive_coupling(1, 2, conductance);
        model->formulas().add_formula("C1", "27.5 + 0.075 * T1");
        model->formulas().add_formula("C2", "27.5 + 0.075 * T2");
        CrankNicolson solver(model);
        solver.set_settings(converged_settings(scheme));
        solver.set_simulation_time(0.0, 1000.0, 50.0, 1000.0);
        solver.initialize();
        solver.solve();
        REQUIRE(solver.converged());
        const double before = enthalpy(t0[0]) + enthalpy(t0[1]);
        const double after = enthalpy(model->nodes().get_T(1)) +
                             enthalpy(model->nodes().get_T(2));
        return std::abs(after - before) / before;
    };
    REQUIRE(run(CapacityScheme::AVERAGE_CAPACITY) < 1e-10);
    // AVERAGE_RATE is second order but not conservative.
    REQUIRE(run(CapacityScheme::AVERAGE_RATE) > 1e-6);
}

TEST_CASE("Both capacity schemes give the same steps for constant capacities",
          "[solver][transient][capacity]") {
    const auto run = [](CapacityScheme scheme) {
        const auto model =
            pycanha::test_models::make_plate(4, /*radiation=*/true);
        CrankNicolson solver(model);
        Settings settings;
        settings.capacity_scheme = scheme;
        solver.set_settings(settings);
        solver.set_simulation_time(0.0, 100.0, 10.0, 10.0);
        solver.initialize();
        solver.solve();
        return Eigen::MatrixXd(solver.output_model().T().values());
    };
    const Eigen::MatrixXd rate = run(CapacityScheme::AVERAGE_RATE);
    const Eigen::MatrixXd mean = run(CapacityScheme::AVERAGE_CAPACITY);
    REQUIRE((rate - mean).cwiseAbs().maxCoeff() < 1e-9);
}

TEST_CASE("AVERAGE_RATE is the default capacity scheme",
          "[solver][transient][capacity]") {
    const auto model = make_cooling_node(300.0, 200.0);
    CrankNicolson solver(model);
    solver.set_simulation_time(0.0, 10.0, 1.0, 10.0);
    REQUIRE(solver.resolved_settings().capacity_scheme ==
            CapacityScheme::AVERAGE_RATE);
}
