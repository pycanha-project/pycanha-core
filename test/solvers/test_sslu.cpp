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

static_assert(!std::is_same_v<pycanha::CouplingMatrices, void>,
              "CouplingMatrices must be a concrete type for solver tests");

namespace {

// NOLINTBEGIN(readability-identifier-naming)
constexpr double kInitialTemp = 273.15;
constexpr int kNodeCount = 5;
// NOLINTEND(readability-identifier-naming)

struct SSLUTestContext {
    std::shared_ptr<pycanha::ThermalMathematicalModel> model;
    std::vector<double> initial_diffusive_temps;
};

SSLUTestContext make_test_context() {
    SSLUTestContext context{
        std::make_shared<pycanha::ThermalMathematicalModel>("sslu-test-model"),
        {}};
    context.initial_diffusive_temps.reserve(
        static_cast<std::size_t>(kNodeCount));

    for (int node_id = 1; node_id <= kNodeCount; ++node_id) {
        pycanha::Node node(node_id);
        node.set_T(kInitialTemp + static_cast<double>(node_id));

        if (node_id == 1 || node_id == kNodeCount) {
            node.set_type(pycanha::BOUNDARY_NODE);
            const double boundary_offset = node_id == 1 ? 10.0 : 100.0;
            node.set_T(kInitialTemp + boundary_offset);
        } else {
            context.initial_diffusive_temps.push_back(node.get_T());
        }

        context.model->add_node(node);
    }

    for (int node_id = 1; node_id < kNodeCount; ++node_id) {
        context.model->add_conductive_coupling(node_id, node_id + 1, 1.0);
        context.model->add_radiative_coupling(node_id, node_id + 1, 0.1);
    }

    return context;
}

void expect_diffusive_temperatures(
    pycanha::ThermalMathematicalModel& model,
    const std::vector<double>& initial_diffusive_temps) {
    auto& nodes = model.nodes();
    const double lower_bound = nodes.get_node_from_node_num(1).get_T();
    const double upper_bound = nodes.get_node_from_node_num(kNodeCount).get_T();

    std::size_t diffusive_index = 0;
    for (int node_id = 2; node_id < kNodeCount; ++node_id) {
        const double solved_temp =
            nodes.get_node_from_node_num(node_id).get_T();

        // NOLINTNEXTLINE(bugprone-chained-comparison)
        REQUIRE(solved_temp >= lower_bound);
        // NOLINTNEXTLINE(bugprone-chained-comparison)
        REQUIRE(solved_temp <= upper_bound);
        // NOLINTNEXTLINE(bugprone-chained-comparison)
        REQUIRE(std::abs(solved_temp -
                         initial_diffusive_temps.at(diffusive_index)) > 1.0e-3);
        ++diffusive_index;
    }
}

}  // namespace

TEST_CASE("SSLU solves a mixed coupling network", "[solver][sslu]") {
    auto context = make_test_context();

    pycanha::SSLU solver(context.model);
    solver.MAX_ITERS = 50;

    solver.initialize();
    REQUIRE(solver.solver_initialized);

    solver.solve();

    REQUIRE(solver.solver_converged);
    // NOLINTNEXTLINE(bugprone-chained-comparison)
    REQUIRE(solver.solver_iter < solver.MAX_ITERS);

    expect_diffusive_temperatures(*context.model,
                                  context.initial_diffusive_temps);
}
