#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <type_traits>
#include <vector>

#include "pycanha-core/solvers/sslu.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

static_assert(!std::is_same_v<pycanha::CouplingMatrices, void>,
              "CouplingMatrices must be a concrete type for solver tests");

TEST_CASE("SSLU solves a mixed coupling network", "[solver][sslu]") {
    auto tmm =
        std::make_shared<pycanha::ThermalMathematicalModel>("sslu-test-model");

    constexpr double initial_temp = 273.15;
    constexpr int node_count = 5;

    std::vector<double> initial_diffusive_temps;
    initial_diffusive_temps.reserve(static_cast<std::size_t>(node_count));

    for (int node_id = 1; node_id <= node_count; ++node_id) {
        pycanha::Node node(node_id);
        node.set_T(initial_temp + static_cast<double>(node_id));

        if (node_id == 1 || node_id == node_count) {
            node.set_type(pycanha::BOUNDARY_NODE);
            const double boundary_offset = node_id == 1 ? 10.0 : 100.0;
            node.set_T(initial_temp + boundary_offset);
        } else {
            initial_diffusive_temps.push_back(node.get_T());
        }

        tmm->add_node(node);
    }

    for (int node_id = 1; node_id < node_count; ++node_id) {
        tmm->add_conductive_coupling(node_id, node_id + 1, 1.0);
        tmm->add_radiative_coupling(node_id, node_id + 1, 0.1);
    }

    pycanha::SSLU solver(tmm);
    solver.MAX_ITERS = 50;

    solver.initialize();
    REQUIRE(solver.solver_initialized);

    solver.solve();

    REQUIRE(solver.solver_converged);
    REQUIRE(solver.solver_iter < solver.MAX_ITERS);

    auto& nodes = tmm->nodes();
    const double lower_bound = nodes.get_node_from_node_num(1).get_T();
    const double upper_bound = nodes.get_node_from_node_num(node_count).get_T();

    std::size_t diffusive_index = 0;
    for (int node_id = 2; node_id < node_count; ++node_id) {
        const double solved_temp =
            nodes.get_node_from_node_num(node_id).get_T();

        REQUIRE(solved_temp >= lower_bound);
        REQUIRE(solved_temp <= upper_bound);
        REQUIRE(std::abs(solved_temp -
                         initial_diffusive_temps.at(diffusive_index)) > 1.0e-3);
        ++diffusive_index;
    }
}
