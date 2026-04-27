#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/solvers/callback_registry.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
#include "pycanha-core/solvers/tscnrlds.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"

TEST_CASE("ThermalModel callbacks expose root model solver and time",
          "[solver][callbacks]") {
    pycanha::ThermalModel tm("callback_model");

    auto node_1 = pycanha::Node(1);
    auto node_2 = pycanha::Node(2);

    node_1.set_T(300.0);
    node_1.set_C(100.0);
    node_1.set_qi(25.0);
    node_2.set_T(250.0);
    node_2.set_type(pycanha::BOUNDARY_NODE);

    tm.tmm().add_node(node_1);
    tm.tmm().add_node(node_2);
    tm.tmm().add_conductive_coupling(1, 2, 5.0);

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

    auto& solver = tm.solvers().tscnrlds();
    solver.max_iters = 20;
    solver.abstol_temp = 1.0e-9;
    solver.set_simulation_time(0.0, 20.0, 1.0, 5.0);

    solver.initialize();
    solver.solve();
    solver.deinitialize();

    REQUIRE(seen_tm == &tm);
    REQUIRE(seen_tmm == &tm.tmm());
    REQUIRE(seen_solver_name == "TSCNRLDS");
    REQUIRE_FALSE(seen_times.empty());
    REQUIRE(seen_times.back() > 0.0);
}

TEST_CASE("CallbackRegistry direct invocations expose context guards",
          "[solver][callbacks]") {
    pycanha::ThermalModel tm("direct_callback_model");

    REQUIRE(tm.tmm().python_callbacks_active);

    tm.tmm().time = 4.5;

    int solver_loop_calls = 0;
    int time_change_calls = 0;
    int after_timestep_calls = 0;

    tm.callbacks().solver_loop = [&](pycanha::CallbackContext& context) {
        ++solver_loop_calls;
        REQUIRE(&context.tm() == &tm);
        REQUIRE(&context.tmm() == &tm.tmm());
        REQUIRE(context.time() == Catch::Approx(4.5));
        const auto& const_context = std::as_const(context);
        REQUIRE(&const_context.tm() == &tm);
        REQUIRE(&const_context.tmm() == &tm.tmm());
        REQUIRE(const_context.time() == Catch::Approx(4.5));
        REQUIRE_THROWS_AS(context.solver(), std::runtime_error);
        REQUIRE_THROWS_AS(const_context.solver(), std::runtime_error);
    };
    tm.callbacks().time_change = [&](pycanha::CallbackContext& context) {
        ++time_change_calls;
        REQUIRE(context.time() == Catch::Approx(4.5));
    };
    tm.callbacks().after_timestep = [&](pycanha::CallbackContext& context) {
        ++after_timestep_calls;
        REQUIRE(context.time() == Catch::Approx(4.5));
    };

    tm.callbacks().invoke_solver_loop();
    tm.callbacks().invoke_time_change();
    tm.callbacks().invoke_after_timestep();

    REQUIRE(solver_loop_calls == 1);
    REQUIRE(time_change_calls == 1);
    REQUIRE(after_timestep_calls == 1);

    tm.callbacks().active = false;
    tm.callbacks().invoke_solver_loop();
    tm.callbacks().invoke_time_change();
    tm.callbacks().invoke_after_timestep();

    REQUIRE(solver_loop_calls == 1);
    REQUIRE(time_change_calls == 1);
    REQUIRE(after_timestep_calls == 1);
}
