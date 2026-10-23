#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/solvers/model_changed_error.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/solvers/sslu.hpp"
#include "pycanha-core/solvers/sslu_cgs.hpp"
#include "pycanha-core/solvers/tscnrlds.hpp"
#include "pycanha-core/solvers/tscnrlds_jacobian.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "solver_test_models.hpp"

namespace {

using pycanha::ModelChangedError;
using pycanha::Solver;
using pycanha::ThermalMathematicalModel;
using pycanha::test_models::Model;
namespace models = pycanha::test_models;

enum class Kind : std::uint8_t { SSLU, SSLU_CGS, TSCNRLDS, TSCNRLDS_JACOBIAN };

std::string_view name_of(Kind kind) {
    switch (kind) {
        case Kind::SSLU:
            return "SSLU";
        case Kind::SSLU_CGS:
            return "SSLU_CGS";
        case Kind::TSCNRLDS:
            return "TSCNRLDS";
        case Kind::TSCNRLDS_JACOBIAN:
            break;
    }
    return "TSCNRLDS_JACOBIAN";
}

bool is_transient(Kind kind) {
    return kind == Kind::TSCNRLDS || kind == Kind::TSCNRLDS_JACOBIAN;
}

// Every solver type of this build.
std::vector<Kind> solver_kinds() {
    std::vector<Kind> kinds{Kind::SSLU, Kind::TSCNRLDS,
                            Kind::TSCNRLDS_JACOBIAN};
    if (pycanha::MKL_ENABLED) {
        kinds.push_back(Kind::SSLU_CGS);
    }
    return kinds;
}

// A 3 x 3 radiative plate whose coupling GL(1,2) follows the parameter k,
// which the Jacobian solver differentiates.
Model make_model() {
    auto model = models::make_plate(3, /*radiation=*/true);
    model->parameters().add_parameter("k", 0.5);
    static_cast<void>(model->formulas().add_parameter_formula("GL(1,2)", "k"));
    model->formulas().apply_formulas();
    model->formulas().parameters_with_derivatives().add_parameter("k");
    return model;
}

void set_transient_time(pycanha::TSCNRLDS& solver) {
    solver.set_simulation_time(0.0, 100.0, 10.0, 20.0);
}

std::unique_ptr<Solver> make_solver(Kind kind, const Model& model) {
    std::unique_ptr<Solver> solver;
    if (kind == Kind::SSLU) {
        solver = std::make_unique<pycanha::SSLU>(model);
    } else if (kind == Kind::SSLU_CGS) {
        solver = std::make_unique<pycanha::SSLU_CGS>(model);
    } else if (kind == Kind::TSCNRLDS) {
        auto transient = std::make_unique<pycanha::TSCNRLDS>(model);
        set_transient_time(*transient);
        solver = std::move(transient);
    } else {
        auto jacobian = std::make_unique<pycanha::TSCNRLDS_JACOBIAN>(model);
        set_transient_time(*jacobian);
        solver = std::move(jacobian);
    }
    solver->max_iters = 100;
    solver->abstol_temp = 1e-6;
    return solver;
}

// The message of the ModelChangedError solve() throws, empty when it does
// not throw one.
std::string model_change_message(Solver& solver) {
    try {
        solver.solve();
    } catch (const ModelChangedError& error) {
        return error.what();
    }
    return {};
}

bool contains(const std::string& text, std::string_view part) {
    return text.contains(part);
}

// After a refused solve, a new initialize() makes the solver usable again.
bool solves_after_initialize(Solver& solver) {
    solver.initialize();
    solver.solve();
    return solver.solver_initialized;
}

// State of the C callbacks below, which cannot capture.
struct ChangeState {
    int calls = 0;
    int change_on_call = 1;
};

ChangeState& change_state() {
    static ChangeState state;
    return state;
}

// Adds the conductive coupling (1, 9), absent from the plate, on the
// change_on_call-th call.
void add_coupling_in_callback(ThermalMathematicalModel* model) {
    auto& state = change_state();
    ++state.calls;
    if (state.calls == state.change_on_call) {
        model->add_conductive_coupling(1, 9, 0.5);
    }
}

enum class Slot : std::uint8_t { SOLVER_LOOP, TIME_CHANGE, AFTER_TIMESTEP };

std::string_view name_of(Slot slot) {
    switch (slot) {
        case Slot::SOLVER_LOOP:
            return "solver loop";
        case Slot::TIME_CHANGE:
            return "time change";
        case Slot::AFTER_TIMESTEP:
            break;
    }
    return "after time step";
}

void install_change(const Model& model, Slot slot, int change_on_call) {
    change_state() = ChangeState{.calls = 0, .change_on_call = change_on_call};
    if (slot == Slot::SOLVER_LOOP) {
        model->c_extern_callback_solver_loop = add_coupling_in_callback;
    } else if (slot == Slot::TIME_CHANGE) {
        model->c_extern_callback_transient_time_change =
            add_coupling_in_callback;
    } else {
        model->c_extern_callback_transient_after_timestep =
            add_coupling_in_callback;
    }
    model->c_callbacks_active = true;
}

}  // namespace

TEST_CASE("Solvers refuse couplings added after initialize()",
          "[solver][model_changed]") {
    const auto kind = GENERATE(from_range(solver_kinds()));
    const bool radiative = GENERATE(false, true);
    INFO(name_of(kind) << (radiative ? ", radiative" : ", conductive"));
    const auto model = make_model();
    const auto solver = make_solver(kind, model);
    solver->initialize();
    if (radiative) {
        model->add_radiative_coupling(1, 9, 0.1);
    } else {
        model->add_conductive_coupling(1, 9, 0.5);
    }

    const std::string message = model_change_message(*solver);
    REQUIRE(contains(message, radiative ? "the model's radiative couplings"
                                        : "the model's conductive couplings"));
    REQUIRE(contains(message, "after initialize()"));
    REQUIRE(solves_after_initialize(*solver));
}

TEST_CASE("Solvers refuse nodes retyped, added or removed after initialize()",
          "[solver][model_changed]") {
    const auto kind = GENERATE(from_range(solver_kinds()));
    const int change = GENERATE(0, 1, 2);
    INFO(name_of(kind) << ", change " << change);
    const auto model = make_model();
    const auto solver = make_solver(kind, model);
    solver->initialize();
    if (change == 0) {
        REQUIRE(model->nodes().set_type(5, 'B'));
    } else if (change == 1) {
        model->add_node(10);
        model->add_conductive_coupling(9, 10, 0.5);
    } else {
        model->network().remove_node(5);
    }

    REQUIRE(contains(model_change_message(*solver), "the model's nodes"));
    REQUIRE(solves_after_initialize(*solver));
}

TEST_CASE("Solvers refuse a coupling swap that keeps the coupling count",
          "[solver][model_changed]") {
    // Removing node 100 drops its only coupling and the new one takes its
    // place: the stored entries differ although their number is the same.
    const auto kind = GENERATE(from_range(solver_kinds()));
    INFO(name_of(kind));
    const auto model = make_model();
    pycanha::Node extra(100);
    extra.set_type(pycanha::BOUNDARY_NODE);
    extra.set_T(250.0);
    model->add_node(extra);
    model->add_conductive_coupling(1, 100, 0.5);
    const auto solver = make_solver(kind, model);
    solver->initialize();
    const auto count =
        model->network().conductive_matrices().get_num_total_couplings();
    model->network().remove_node(100);
    model->add_conductive_coupling(1, 9, 0.5);
    REQUIRE(model->network().conductive_matrices().get_num_total_couplings() ==
            count);

    REQUIRE(contains(model_change_message(*solver), "conductive couplings"));
}

TEST_CASE("Steady-state solvers refuse a coupling added in a callback",
          "[solver][model_changed]") {
    const auto kind = GENERATE(from_range(solver_kinds()));
    if (is_transient(kind)) {
        SKIP("covered by the transient case");
    }
    INFO(name_of(kind));
    const auto model = make_model();
    const auto solver = make_solver(kind, model);
    solver->initialize();
    install_change(model, Slot::SOLVER_LOOP, 1);

    const std::string message = model_change_message(*solver);
    REQUIRE(contains(message,
                     "conductive couplings changed in the solver "
                     "loop callback"));
    REQUIRE_FALSE(model->parameters().is_structure_locked());
}

TEST_CASE("Transient solvers refuse a coupling added in any callback",
          "[solver][model_changed]") {
    const auto kind = GENERATE(Kind::TSCNRLDS, Kind::TSCNRLDS_JACOBIAN);
    const auto slot =
        GENERATE(Slot::SOLVER_LOOP, Slot::TIME_CHANGE, Slot::AFTER_TIMESTEP);
    // The third call of each callback falls inside the run, never at its
    // start.
    INFO(name_of(kind) << ", " << name_of(slot));
    const auto model = make_model();
    const auto solver = make_solver(kind, model);
    solver->initialize();
    install_change(model, slot, 3);

    const std::string message = model_change_message(*solver);
    REQUIRE(contains(message, "conductive couplings changed in the " +
                                  std::string(name_of(slot)) + " callback"));
    REQUIRE_FALSE(model->parameters().is_structure_locked());
    model->c_callbacks_active = false;
    REQUIRE(solves_after_initialize(*solver));
}

TEST_CASE("Value changes after initialize() never refuse the solve",
          "[solver][model_changed]") {
    const auto kind = GENERATE(from_range(solver_kinds()));
    INFO(name_of(kind));
    const auto model = make_model();
    const auto solver = make_solver(kind, model);
    solver->initialize();

    model->conductive_couplings().set_coupling_value(2, 3, 0.7);
    model->radiative_couplings().set_coupling_value(1, models::space_node, 0.2);
    model->nodes().set_T(1, 260.0);
    model->nodes().set_C(2, 3.0);
    model->nodes().set_qi(3, 4.0);
    model->parameters().set_parameter("k", 0.6);
    REQUIRE_NOTHROW(solver->solve());
    REQUIRE(solver->solver_converged);
}

TEST_CASE("A transient solve reports a missing initialize() first",
          "[solver][model_changed]") {
    const auto model = make_model();
    const auto solver = make_solver(Kind::TSCNRLDS, model);
    model->add_conductive_coupling(1, 9, 0.5);
    REQUIRE_THROWS_AS(solver->solve(), std::invalid_argument);
}
