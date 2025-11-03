#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "pycanha-core/config.hpp"
#include "pycanha-core/solvers/tscnrlds.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace {

// NOLINTBEGIN(readability-identifier-naming)
constexpr double kInitialDiffusiveTemp = 280.0;
constexpr double kBoundaryTemp = 290.0;
constexpr double kHeatInput = 5.0;
constexpr double kThermalCapacity = 100.0;
constexpr double kConductiveCoupling = 0.1;
constexpr double kRadiativeCoupling = 0.05;
// NOLINTEND(readability-identifier-naming)

struct SolverContext {
    std::shared_ptr<pycanha::ThermalMathematicalModel> model;
};

SolverContext make_solver_context() {
    SolverContext context{std::make_shared<pycanha::ThermalMathematicalModel>(
        "tscnrlds-test-model")};

    pycanha::Node node1(1);
    node1.set_T(kInitialDiffusiveTemp);
    node1.set_C(kThermalCapacity);
    node1.set_qi(kHeatInput);

    pycanha::Node node2(2);
    node2.set_type(pycanha::BOUNDARY_NODE);
    node2.set_T(kBoundaryTemp);

    context.model->add_node(node1);
    context.model->add_node(node2);

    context.model->add_conductive_coupling(1, 2, kConductiveCoupling);
    context.model->add_radiative_coupling(1, 2, kRadiativeCoupling);

    return context;
}

void initialize_solver(pycanha::TSCNRLDS& solver) {
    REQUIRE_NOTHROW(solver.initialize());
    REQUIRE(solver.solver_initialized);
}

void execute_solver(pycanha::TSCNRLDS& solver) {
    REQUIRE_NOTHROW(solver.solve());
    REQUIRE(solver.solver_converged);
}

void verify_solver_outputs(pycanha::ThermalMathematicalModel& model) {
    auto& thermal_data = model.thermal_data;
    REQUIRE(thermal_data.has_table("TSCNRLDS_OUTPUT"));
    const auto& results = thermal_data.get_table("TSCNRLDS_OUTPUT");
    const bool has_rows = results.rows() > 0;
    const bool has_min_columns = results.cols() >= 2;
    REQUIRE(has_rows);
    REQUIRE(has_min_columns);
}

void verify_node_temperatures(pycanha::ThermalMathematicalModel& model) {
    auto& nodes = model.nodes();
    const double diffusive_temp_after = nodes.get_T(1);
    const double boundary_temp_after = nodes.get_T(2);

    const auto initial_temp = Catch::Approx(kInitialDiffusiveTemp);
    const auto boundary_temp = Catch::Approx(kBoundaryTemp);

    const bool diffusive_temp_changed = diffusive_temp_after != initial_temp;
    const bool boundary_temp_stable = boundary_temp_after == boundary_temp;

    REQUIRE(diffusive_temp_changed);
    REQUIRE(boundary_temp_stable);
}

void shutdown_solver(pycanha::TSCNRLDS& solver) {
    REQUIRE_NOTHROW(solver.deinitialize());
    REQUIRE_FALSE(solver.solver_initialized);
}

}  // namespace

TEST_CASE("TSCNRLDS solves a simple transient case", "[solver][tscnrlds]") {
    if (!pycanha::MKL_ENABLED) {
        SUCCEED("TSCNRLDS requires MKL; test skipped when MKL is disabled");
        return;
    }

    auto context = make_solver_context();

    pycanha::TSCNRLDS solver(context.model);
    solver.MAX_ITERS = 50;
    solver.set_simulation_time(0.0, 1.0, 0.05, 0.1);

    initialize_solver(solver);
    execute_solver(solver);
    verify_solver_outputs(*context.model);
    verify_node_temperatures(*context.model);
    shutdown_solver(solver);
}
