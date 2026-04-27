#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
#include "pycanha-core/solvers/sslu.hpp"
#include "pycanha-core/solvers/tscnrlds.hpp"
#include "pycanha-core/solvers/tscnrlds_jacobian.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"

namespace {

void populate_parameterized_steady_model(pycanha::ThermalModel& tm) {
    pycanha::Node plate(1);
    plate.set_T(300.0);
    plate.set_C(100.0);

    pycanha::Node sink(2);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(250.0);

    tm.tmm().add_node(plate);
    tm.tmm().add_node(sink);
    tm.tmm().add_conductive_coupling(1, 2, 0.0);

    tm.parameters().add_parameter("heater_power", 50.0);
    tm.parameters().add_parameter("conductance", 5.0);
    static_cast<void>(
        tm.formulas().add_parameter_formula("QI1", "heater_power"));
    static_cast<void>(
        tm.formulas().add_parameter_formula("GL(1,2)", "conductance"));
    tm.formulas().apply_formulas();
}

void populate_transient_model(pycanha::ThermalModel& tm) {
    pycanha::Node plate(1);
    plate.set_T(300.0);
    plate.set_C(10.0);
    plate.set_qi(25.0);

    pycanha::Node sink(2);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(250.0);

    tm.tmm().add_node(plate);
    tm.tmm().add_node(sink);
    tm.tmm().add_conductive_coupling(1, 2, 5.0);
}

void populate_jacobian_model(pycanha::ThermalModel& tm) {
    pycanha::Node diffusive_node(1);
    diffusive_node.set_T(0.0);
    diffusive_node.set_C(1.0);
    diffusive_node.set_qi(1.0);

    pycanha::Node boundary_node(2);
    boundary_node.set_type(pycanha::BOUNDARY_NODE);
    boundary_node.set_T(1.0);

    tm.tmm().add_node(diffusive_node);
    tm.tmm().add_node(boundary_node);
    tm.tmm().add_conductive_coupling(1, 2, 1.0);

    tm.parameters().add_parameter("k", 1.0);
    tm.parameters().add_parameter("C", 1.0);
    static_cast<void>(tm.formulas().add_parameter_formula("GL(1,2)", "k"));
    static_cast<void>(tm.formulas().add_parameter_formula("C1", "C"));
    tm.formulas().apply_formulas();
}

}  // namespace

TEST_CASE("SolverRegistry keeps a stable, model-owned solver surface",
          "[api][solvers]") {
    pycanha::ThermalModel tm("solvers_demo");
    populate_parameterized_steady_model(tm);

    auto& registry = tm.solvers();
    REQUIRE(&registry == &tm.tmm().solvers());
    REQUIRE(&registry.sslu() == &tm.solvers().sslu());
    REQUIRE(&registry.tscnrlds() == &tm.solvers().tscnrlds());
    REQUIRE(&registry.tscnrlds_jacobian() == &tm.solvers().tscnrlds_jacobian());

    // Solvers are configured first, then initialized explicitly. That keeps
    // the lifecycle visible to readers and mirrors how the production API is
    // meant to be used from bindings.
    auto& steady = registry.sslu();
    steady.max_iters = 50;
    steady.abstol_temp = 1.0e-9;
    steady.initialize();
    steady.solve();

    REQUIRE(steady.solver_initialized);
    REQUIRE(steady.solver_converged);
    REQUIRE(tm.tmm().nodes().get_T(1) == Catch::Approx(260.0));

    steady.deinitialize();
    steady.initialize();
    steady.solve();
    REQUIRE(tm.tmm().nodes().get_T(1) == Catch::Approx(260.0));
    steady.deinitialize();
}

TEST_CASE("Transient solvers write reusable output models", "[api][solvers]") {
    SECTION(
        "TSCNRLDS stores its output in ThermalData and exposes a shortcut") {
        pycanha::ThermalModel tm("transient_demo");
        populate_transient_model(tm);

        auto& transient = tm.solvers().tscnrlds();
        transient.max_iters = 20;
        transient.abstol_temp = 1.0e-9;
        transient.set_simulation_time(0.0, 2.0, 0.1, 0.5);
        transient.initialize();
        transient.solve();
        transient.deinitialize();

        REQUIRE(tm.thermal_data().models().size() == 1);
        const auto model_names = tm.thermal_data().models().model_names();
        REQUIRE(model_names.size() == 1);
        REQUIRE(&transient.output_model() ==
                &tm.thermal_data().models().get_model(model_names.front()));
        REQUIRE(transient.output_model().T().num_timesteps() > 1);
        REQUIRE(transient.output_model().T().num_columns() == 2);
    }

    SECTION("TSCNRLDS_JACOBIAN honors the registered derivative parameters") {
        pycanha::ThermalModel tm("jacobian_demo");
        populate_jacobian_model(tm);
        tm.formulas().parameters_with_derivatives().add_parameter("C");
        tm.formulas().parameters_with_derivatives().add_parameter("k");

        auto& jacobian_solver = tm.solvers().tscnrlds_jacobian();
        jacobian_solver.max_iters = 50;
        jacobian_solver.abstol_temp = 1.0e-9;
        jacobian_solver.set_simulation_time(0.0, 5.0, 0.01, 0.1);
        jacobian_solver.initialize();
        jacobian_solver.solve();
        jacobian_solver.deinitialize();

        REQUIRE(jacobian_solver.derivative_parameter_names().size() == 2);
        REQUIRE(jacobian_solver.derivative_parameter_names().at(0) == "C");
        REQUIRE(jacobian_solver.derivative_parameter_names().at(1) == "k");
        REQUIRE(jacobian_solver.output_model().jacobian().cols() == 2);
        REQUIRE(jacobian_solver.output_model().jacobian().rows() == 1);
    }
}
