#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstddef>
#include <memory>
#include <vector>

#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/jacobian_propagation.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/dense_matrix_time_series.hpp"
#include "pycanha-core/thermaldata/dense_time_series.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

// A transient whose length is not a whole number of time steps ends at
// end_time, as ESATAN's does: full steps while at least two remain, then the
// rest in one step, or in two equal ones when it is longer than a step.

namespace {

constexpr double capacity = 100.0;
constexpr double start_temperature = 300.0;
constexpr double sink_temperature = 200.0;
constexpr double dissipation = 10.0;
constexpr double conductance = 1.0;
constexpr double time_step = 10.0;

// Node 1 (100 J/K, 10 W) - GL k - node 2 (200 K). With a parameter, GL(1,2)
// is the formula "k".
std::shared_ptr<pycanha::ThermalMathematicalModel> make_model(
    double coupling, bool with_parameter) {
    auto model =
        std::make_shared<pycanha::ThermalMathematicalModel>("end_time");
    pycanha::Node node(1);
    node.set_T(start_temperature);
    node.set_C(capacity);
    node.set_qi(dissipation);
    pycanha::Node sink(2);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(sink_temperature);
    model->add_node(node);
    model->add_node(sink);
    model->add_conductive_coupling(1, 2, coupling);
    if (with_parameter) {
        model->parameters().add_parameter("k", coupling);
        static_cast<void>(
            model->formulas().add_parameter_formula("GL(1,2)", "k"));
        model->formulas().apply_formulas();
        model->formulas().parameters_with_derivatives().add_parameter("k");
    }
    return model;
}

pycanha::solvers::Settings tight_settings(bool fixed_couplings) {
    pycanha::solvers::Settings settings;
    settings.max_iterations = 50;
    settings.tolerance = 1.0e-10;
    settings.fixed_couplings = fixed_couplings;
    return settings;
}

// One Crank-Nicolson step of length dt of the linear model.
double step(double temperature, double dt) {
    const double c_over = 2.0 * capacity / dt;
    return (((c_over - conductance) * temperature) + (2.0 * dissipation) +
            (2.0 * conductance * sink_temperature)) /
           (c_over + conductance);
}

std::vector<double> output_times(const pycanha::DenseTimeSeries& table) {
    std::vector<double> times;
    for (Eigen::Index row = 0; row < table.num_timesteps(); ++row) {
        times.push_back(table.times()(row));
    }
    return times;
}

}  // namespace

TEST_CASE("CrankNicolson ends at end_time between two whole steps",
          "[solver][transient]") {
    const bool fixed = GENERATE(false, true);
    auto model = make_model(conductance, /*with_parameter=*/false);
    pycanha::solvers::transient::CrankNicolson solver(model);
    solver.set_settings(tight_settings(fixed));
    // 25 s in steps of 10 s: 10 s, then the 15 s left in two steps of 7.5 s.
    solver.set_simulation_time(0.0, 25.0, time_step, time_step);
    solver.initialize();
    solver.solve();

    const pycanha::DenseTimeSeries& table = solver.output_model().T();
    const std::vector<double> times = output_times(table);
    REQUIRE(times.size() == 3);
    REQUIRE(times.at(0) == Catch::Approx(0.0));
    REQUIRE(times.at(1) == Catch::Approx(10.0));
    REQUIRE(times.at(2) == 25.0);
    const double at_10 = step(start_temperature, time_step);
    const double at_25 = step(step(at_10, 7.5), 7.5);
    REQUIRE(table.values()(1, 0) == Catch::Approx(at_10).margin(1.0e-6));
    REQUIRE(table.values()(2, 0) == Catch::Approx(at_25).margin(1.0e-6));
    REQUIRE(model->time == 25.0);
}

TEST_CASE("CrankNicolson takes a run shorter than a step in one step",
          "[solver][transient]") {
    auto model = make_model(conductance, /*with_parameter=*/false);
    pycanha::solvers::transient::CrankNicolson solver(model);
    solver.set_settings(tight_settings(/*fixed_couplings=*/false));
    solver.set_simulation_time(0.0, 4.0, time_step, 0.0);
    solver.initialize();
    solver.solve();

    const pycanha::DenseTimeSeries& table = solver.output_model().T();
    const std::vector<double> times = output_times(table);
    REQUIRE(times.size() == 2);
    REQUIRE(times.at(1) == 4.0);
    REQUIRE(table.values()(1, 0) ==
            Catch::Approx(step(start_temperature, 4.0)).margin(1.0e-6));
}

TEST_CASE("JacobianPropagation ends at end_time between two whole steps",
          "[solver][transient][jacobian]") {
    // dT/dk at the end against a central difference of two runs.
    const auto end_temperature = [](double coupling) {
        auto model = make_model(coupling, /*with_parameter=*/false);
        pycanha::solvers::transient::CrankNicolson solver(model);
        solver.set_settings(tight_settings(/*fixed_couplings=*/false));
        solver.set_simulation_time(0.0, 25.0, time_step, time_step);
        solver.initialize();
        solver.solve();
        const pycanha::DenseTimeSeries& table = solver.output_model().T();
        return table.values()(table.num_timesteps() - 1, 0);
    };
    constexpr double h = 1.0e-4;
    const double central =
        (end_temperature(conductance + h) - end_temperature(conductance - h)) /
        (2.0 * h);

    auto model = make_model(conductance, /*with_parameter=*/true);
    pycanha::solvers::sensitivity::JacobianPropagation solver(model);
    solver.set_settings(tight_settings(/*fixed_couplings=*/false));
    solver.set_simulation_time(0.0, 25.0, time_step, time_step);
    solver.initialize();
    solver.solve();
    const pycanha::DenseMatrixTimeSeries& jacobian =
        solver.output_model().jacobian();
    const Eigen::Index last = jacobian.num_timesteps() - 1;
    REQUIRE(jacobian.time_at(last) == 25.0);
    REQUIRE(jacobian.at(last)(0, 0) ==
            Catch::Approx(central).epsilon(1.0e-5).margin(1.0e-7));
    const std::vector<double> times = output_times(solver.output_model().T());
    REQUIRE(times.size() == 3);
}
