#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

#include "pycanha-core/solvers/callback_registry.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
#include "pycanha-core/solvers/tscnrlds.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"

namespace {

void populate_callback_model(pycanha::ThermalModel& tm, double heater_power) {
    pycanha::Node plate(1);
    plate.set_T(300.0);
    plate.set_C(10.0);
    plate.set_qi(heater_power);

    pycanha::Node sink(2);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(250.0);

    tm.tmm().add_node(plate);
    tm.tmm().add_node(sink);
    tm.tmm().add_conductive_coupling(1, 2, 5.0);
}

void run_transient(pycanha::ThermalModel& tm) {
    auto& solver = tm.solvers().tscnrlds();
    solver.max_iters = 20;
    solver.abstol_temp = 1.0e-9;
    solver.set_simulation_time(0.0, 2.0, 0.1, 0.5);
    solver.initialize();
    solver.solve();
    solver.deinitialize();
}

}  // namespace

TEST_CASE("CallbackContext exposes the whole model-owned solver workflow",
          "[api][callbacks]") {
    pycanha::ThermalModel tm("callback_demo");
    populate_callback_model(tm, 25.0);

    pycanha::ThermalModel* seen_tm = nullptr;
    pycanha::ThermalMathematicalModel* seen_tmm = nullptr;
    std::string seen_solver_name;
    std::vector<double> seen_times;

    tm.callbacks().after_timestep = [&](pycanha::CallbackContext& context) {
        seen_tm = &context.tm();
        seen_tmm = &context.tmm();
        seen_solver_name = context.solver().solver_name;
        seen_times.push_back(context.time());
    };

    run_transient(tm);

    REQUIRE(seen_tm == &tm);
    REQUIRE(seen_tmm == &tm.tmm());
    REQUIRE(seen_solver_name == "TSCNRLDS");
    REQUIRE_FALSE(seen_times.empty());
    REQUIRE(seen_times.back() > 0.0);
}

TEST_CASE("Callbacks can teach active control logic without leaving the model",
          "[api][callbacks]") {
    SECTION("after_timestep can switch off a heater mid-transient") {
        pycanha::ThermalModel tm("thermostat_demo");
        populate_callback_model(tm, 25.0);

        int callback_count = 0;
        tm.callbacks().after_timestep = [&](pycanha::CallbackContext& context) {
            ++callback_count;
            if (context.tmm().nodes().get_T(1) > 280.0) {
                context.tmm().nodes().set_qi(1, 0.0);
            }
        };

        run_transient(tm);

        REQUIRE(callback_count > 0);
        REQUIRE(tm.tmm().nodes().get_qi(1) == Catch::Approx(0.0));
    }

    SECTION("active=false disables callback side effects") {
        pycanha::ThermalModel tm("disabled_callback_demo");
        populate_callback_model(tm, 25.0);

        int callback_count = 0;
        tm.callbacks().active = false;
        tm.callbacks().after_timestep = [&](pycanha::CallbackContext& context) {
            ++callback_count;
            context.tmm().nodes().set_qi(1, 0.0);
        };

        run_transient(tm);

        REQUIRE(callback_count == 0);
        REQUIRE(tm.tmm().nodes().get_qi(1) == Catch::Approx(25.0));
    }
}