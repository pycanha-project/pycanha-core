#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "pycanha-core/config.hpp"
#include "pycanha-core/solvers/tscnrlds.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace {

constexpr double kInitialDiffusiveTemp = 280.0;
constexpr double kBoundaryTemp = 290.0;
constexpr double kHeatInput = 5.0;
constexpr double kThermalCapacity = 100.0;
constexpr double kConductiveCoupling = 0.1;
constexpr double kRadiativeCoupling = 0.05;

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

    REQUIRE_NOTHROW(solver.initialize());
    REQUIRE(solver.solver_initialized);

    REQUIRE_NOTHROW(solver.solve());
    REQUIRE(solver.solver_converged);

    auto& thermal_data = context.model->thermal_data;
    REQUIRE(thermal_data.has_table("TSCNRLDS_OUTPUT"));
    const auto& results = thermal_data.get_table("TSCNRLDS_OUTPUT");
    REQUIRE(results.rows() > 0);
    REQUIRE(results.cols() >= 2);

    auto& nodes = context.model->nodes();
    const double diffusive_temp_after = nodes.get_T(1);
    const double boundary_temp_after = nodes.get_T(2);

    REQUIRE(diffusive_temp_after != Catch::Approx(kInitialDiffusiveTemp));
    REQUIRE(boundary_temp_after == Catch::Approx(kBoundaryTemp));

    REQUIRE_NOTHROW(solver.deinitialize());
    REQUIRE_FALSE(solver.solver_initialized);
}
