#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include "pycanha-core/solvers/sslu.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"



namespace {


constexpr double init_temp = 273.15;
constexpr int num_nodes = 5;
constexpr double tol_temp = 1e-2;

// Steady state expected temperatures
// 
constexpr std::array<double, num_nodes> expected_temps = {
    132.38706, 306.56526, 111.78443, 200.32387, 3.14999
};

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
    model->add_radiative_coupling(20, 99, 1.0);
    model->add_radiative_coupling(15, 25, 0.2);
    model->add_radiative_coupling(15, 99, 0.8);
    model->add_radiative_coupling(25, 99, 0.8);



    return model;

}


bool compare_temps(pycanha::ThermalMathematicalModel& model, bool print_diffs = false) {

    bool all_within_tol = true;
    // Loop over node ids
    for (int i = 0; i < num_nodes; ++i) {
        const auto node_id = node_ids[i];
        const auto node_temp = model.nodes().get_T(node_id);
        const auto expected_temp = expected_temps[i];

        if (std::fabs(node_temp - expected_temp) > tol_temp) {
            all_within_tol = false;
        }

        if (print_diffs) {
            std::cout << "Node " << node_id << ": "
                      << "Computed Temp = " << node_temp << " K, "
                      << "Expected Temp = " << expected_temp << " K, "
                      << "Diff = " << std::fabs(node_temp - expected_temp)
                      << " K\n";
        }
    }

    return all_within_tol;
}

} // namespace

TEST_CASE("SSLU solves a simple model", "[solver][sslu]") {
    auto model = make_model();

    pycanha::SSLU solver(model);
    solver.MAX_ITERS = 100;
    solver.abstol_temp = 1e-6;

    solver.initialize();

    REQUIRE(solver.solver_initialized);

    solver.solve();

    REQUIRE(solver.solver_converged);

    REQUIRE(solver.solver_iter < solver.MAX_ITERS);

    // In case of error, set print_diffs to true to see detailed comparison 
    REQUIRE(compare_temps(*model, false));
}
