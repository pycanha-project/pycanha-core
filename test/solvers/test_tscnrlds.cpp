#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>

#include "pycanha-core/solvers/tscnrlds.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace {

constexpr double init_temp = 273.15;
constexpr int num_nodes = 5;
constexpr double tol_temp = 1e-2;
constexpr int num_time_steps = 10;
constexpr double tol_time = 1e-6;

// Expected times recorded in the solver output table.
constexpr std::array<double, num_time_steps + 1> times = {
    0.0000000,  10000.0000, 20000.0000, 30000.0000, 40000.0000, 50000.0000,
    60000.0000, 70000.0000, 80000.0000, 90000.0000, 100000.0};

// Transient expected temperatures (time steps x nodes)
constexpr std::array<std::array<double, num_nodes>, num_time_steps + 1>
    expected_temps = {{
        {273.14999, 273.14999, 273.14999, 273.14999, 3.14999},
        {259.03552, 283.85105, 258.98241, 262.06791, 3.14999},
        {247.56014, 291.67014, 247.37629, 253.45623, 3.14999},
        {237.98527, 297.25685, 237.62266, 246.62735, 3.14999},
        {229.83503, 301.16946, 229.26392, 241.11244, 3.14999},
        {222.78667, 303.85891, 221.98896, 236.58283, 3.14999},
        {216.61234, 305.67267, 215.57742, 232.80415, 3.14999},
        {211.14591, 306.86934, 209.86801, 229.60718, 3.14999},
        {206.26295, 307.63674, 204.73939, 226.86828, 3.14999},
        {201.86811, 308.10888, 200.09819, 224.49601, 3.14999},
        {197.88691, 308.38019, 195.87117, 222.42185, 3.14999},
    }};

constexpr std::array<int, num_nodes> node_ids = {10, 15, 20, 25, 99};

std::shared_ptr<pycanha::ThermalMathematicalModel> make_model() {
    auto model =
        std::make_shared<pycanha::ThermalMathematicalModel>("test_model");

    auto node_10 = pycanha::Node(10);
    auto node_15 = pycanha::Node(15);
    auto node_20 = pycanha::Node(20);
    auto node_25 = pycanha::Node(25);
    auto env_node = pycanha::Node(99);

    // Set initial temperatures
    node_10.set_T(init_temp);
    node_15.set_T(init_temp);
    node_20.set_T(init_temp);
    node_25.set_T(init_temp);
    env_node.set_T(3.15);

    // Set thermal capacities
    node_10.set_C(2.0e5);
    node_15.set_C(2.0e5);
    node_20.set_C(2.0e5);
    node_25.set_C(2.0e5);

    // Set dissipation
    node_15.set_qi(500.0);

    // Set node types
    env_node.set_type(pycanha::BOUNDARY_NODE);

    // Add nodes to the model
    model->add_node(node_10);
    model->add_node(node_15);
    model->add_node(node_20);
    model->add_node(node_25);
    model->add_node(env_node);

    // Add conductive couplings
    model->add_conductive_coupling(10, 15, 0.1);
    model->add_conductive_coupling(20, 25, 0.1);

    // Add radiative couplings
    model->add_radiative_coupling(10, 99, 1.0);
    model->add_radiative_coupling(15, 25, 0.2);
    model->add_radiative_coupling(15, 99, 0.8);
    model->add_radiative_coupling(20, 99, 1.0);
    model->add_radiative_coupling(25, 99, 0.8);

    return model;
}

bool compare_temps(pycanha::ThermalMathematicalModel& model,
                   bool print_diffs = false) {
    const auto& thermal_data = model.thermal_data;
    if (!thermal_data.has_table("TSCNRLDS_OUTPUT")) {
        if (print_diffs) {
            std::cout << "Thermal data table 'TSCNRLDS_OUTPUT' not found."
                      << '\n';
        }
        return false;
    }

    const auto& output_table = thermal_data.get_table("TSCNRLDS_OUTPUT");
    const auto output_rows = static_cast<std::size_t>(output_table.rows());
    const auto output_cols = static_cast<std::size_t>(output_table.cols());

    bool all_within_tol = true;

    if (output_rows != times.size() || output_cols != num_nodes + 1) {
        if (print_diffs) {
            std::cout << "Unexpected output table shape: " << output_rows << "x"
                      << output_cols << " (expected " << times.size() << "x"
                      << (num_nodes + 1) << ")\n";
        }
        return false;
    }

    std::array<Eigen::Index, num_nodes> node_column_indices{};
    const auto& nodes = model.nodes();
    for (std::size_t i = 0; i < node_ids.size(); ++i) {
        node_column_indices.at(i) =
            nodes.get_idx_from_node_num(node_ids.at(i)) + 1;
    }

    for (std::size_t time_idx = 0; time_idx < times.size(); ++time_idx) {
        const auto row = static_cast<Eigen::Index>(time_idx);
        const double computed_time = output_table(row, 0);
        const double expected_time = times.at(time_idx);

        if (std::fabs(computed_time - expected_time) > tol_time) {
            all_within_tol = false;
            if (print_diffs) {
                std::cout << "Time index " << time_idx
                          << ": Computed time = " << computed_time
                          << " s, Expected time = " << expected_time << " s\n";
            }
        }

        for (std::size_t node_idx = 0; node_idx < node_ids.size(); ++node_idx) {
            const auto column = node_column_indices.at(node_idx);
            const double computed_temp = output_table(row, column);
            const double expected_temp =
                expected_temps.at(time_idx).at(node_idx);

            if (std::fabs(computed_temp - expected_temp) > tol_temp) {
                all_within_tol = false;
            }

            if (print_diffs) {
                std::cout << "t=" << expected_time << " s, Node "
                          << node_ids.at(node_idx)
                          << ": Computed = " << computed_temp
                          << " K, Expected = " << expected_temp << " K, Diff = "
                          << std::fabs(computed_temp - expected_temp) << " K\n";
            }
        }
    }

    return all_within_tol;
}

void reset_model_temps(pycanha::ThermalMathematicalModel& model) {
    model.nodes().set_T(10, init_temp);
    model.nodes().set_T(15, init_temp);
    model.nodes().set_T(20, init_temp);
    model.nodes().set_T(25, init_temp);
}

}  // namespace

TEST_CASE("TSCNRLDS solves a simple model", "[solver][tscnrlds]") {
    auto model = make_model();

    pycanha::TSCNRLDS solver(model);
    solver.MAX_ITERS = 100;
    solver.abstol_temp = 1e-6;
    solver.set_simulation_time(0.0, 100000.0, 1000.0, 10000.0);

    solver.initialize();

    REQUIRE(solver.solver_initialized);

    solver.solve();

    // In case of error, set print_diffs to true to see detailed comparison
    REQUIRE(compare_temps(*model, false));

    // Re-run to verify no errors (like mem-leaks) on multiple initializations
    solver.deinitialize();
    reset_model_temps(*model);
    solver.MAX_ITERS = 100;
    solver.abstol_temp = 1e-6;
    solver.set_simulation_time(0.0, 100000.0, 1000.0, 10000.0);
    solver.initialize();
    solver.solve();
    REQUIRE(compare_temps(*model, false));

    // Create another solver instance to verify multiple solvers work in the
    // same model
    auto solver2 = pycanha::TSCNRLDS(model);
    reset_model_temps(*model);
    solver2.MAX_ITERS = 100;
    solver2.abstol_temp = 1e-6;
    solver2.set_simulation_time(0.0, 100000.0, 1000.0, 10000.0);
    solver2.initialize();
    solver2.solve();
    REQUIRE(compare_temps(*model, false));
}
