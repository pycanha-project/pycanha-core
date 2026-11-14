// NOLINTBEGIN(readability-function-cognitive-complexity)
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/jacobian_propagation.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/retracer.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "solver_test_models.hpp"

namespace {

using pycanha::Index;
using pycanha::solvers::Retracer;
using pycanha::solvers::RetracerCounters;
using pycanha::solvers::Settings;
using pycanha::solvers::SolveStatus;
using pycanha::solvers::sensitivity::JacobianPropagation;
using pycanha::solvers::steady::SteadyState;
using pycanha::solvers::transient::CrankNicolson;
namespace models = pycanha::test_models;

constexpr double time_step = 10.0;
constexpr double end_time = 30.0;
constexpr double internal_watts = 40.0;
// Enough for the radiative test model to converge in a steady state.
constexpr int steady_iterations = 50;

// Records what the solver calls and answers as configured.
class FakeRetracer : public Retracer {
  public:
    // The calls in order: 'B' begin_solve, 'U' update, 'A'
    // accepts_convergence, 'E' end_solve; the model callbacks add 'C'.
    // Mutable: accepts_convergence() is const in the interface.
    mutable std::string calls;
    // accepts_convergence() refuses its first `refusals` calls.
    int refusals = 0;
    bool changes_structure = false;
    std::vector<double> load;
    SparseMatrix overflow_couplings;
    RetracerCounters reported{
        .retraces = 3, .overflow_couplings = 1, .overflow_share = 0.25};
    // update() writes this radiative coupling, when set.
    std::shared_ptr<pycanha::ThermalMathematicalModel> model;
    int written_node_1 = 0;
    int written_node_2 = 0;
    double written_value = 0.0;

    void begin_solve() override { calls += 'B'; }
    void update() override {
        calls += 'U';
        if (model != nullptr) {
            model->radiative_couplings().set_coupling_value(
                written_node_1, written_node_2, written_value);
        }
    }
    [[nodiscard]] bool accepts_convergence(
        double /*tolerance*/) const override {
        calls += 'A';
        return ++_accept_calls > refusals;
    }
    [[nodiscard]] std::span<const double> internal_load() const override {
        return load;
    }
    [[nodiscard]] const SparseMatrix& overflow() const override {
        return overflow_couplings;
    }
    [[nodiscard]] RetracerCounters counters() const override {
        return reported;
    }
    bool end_solve() override {
        calls += 'E';
        return changes_structure;
    }

  private:
    mutable int _accept_calls = 0;
};

Index index_of(const models::Model& model, int number) {
    const auto index = model->nodes().get_idx_from_node_num(number);
    REQUIRE(index.has_value());
    return index.value_or(-1);
}

std::vector<double> load_on(const models::Model& model, int node,
                            double watts) {
    std::vector<double> load(model->nodes().T_vector.size(), 0.0);
    load.at(pycanha::to_sizet(index_of(model, node))) = watts;
    return load;
}

void log_callbacks(const models::Model& model,
                   const std::shared_ptr<FakeRetracer>& retracer) {
    model->python_extern_callback_solver_loop = [retracer]() {
        retracer->calls += 'C';
    };
    model->python_extern_callback_transient_time_change = [retracer]() {
        retracer->calls += 'C';
    };
    model->python_extern_callback_transient_after_timestep = [retracer]() {
        retracer->calls += 'C';
    };
    model->python_callbacks_active = true;
}

// A steady state of `model` with room to converge.
std::unique_ptr<SteadyState> make_steady(const models::Model& model) {
    auto solver = std::make_unique<SteadyState>(model);
    solver->set("max_iterations", std::int64_t{steady_iterations});
    return solver;
}

void set_transient(CrankNicolson& solver) {
    Settings settings;
    settings.output_name = "retracer_run";
    solver.set_settings(settings);
    solver.set_simulation_time(0.0, end_time, time_step, time_step);
}

}  // namespace

TEST_CASE("JacobianPropagation refuses a retracer", "[solver][retracer]") {
    auto model = models::make_small_radiative_model();
    JacobianPropagation solver(model);
    REQUIRE_THROWS_AS(solver.set_retracer(std::make_shared<FakeRetracer>()),
                      std::invalid_argument);
    REQUIRE(solver.retracer() == nullptr);
}

TEST_CASE("fixed_couplings refuses a retracer", "[solver][retracer]") {
    auto model = models::make_small_radiative_model();
    auto solver_ptr = make_steady(model);
    SteadyState& solver = *solver_ptr;
    solver.set("fixed_couplings", true);
    solver.set_retracer(std::make_shared<FakeRetracer>());
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);

    solver.set_retracer(nullptr);
    solver.initialize();
    REQUIRE_THROWS_AS(solver.set_retracer(std::make_shared<FakeRetracer>()),
                      std::invalid_argument);
}

TEST_CASE("The internal load of a retracer is added to the node loads",
          "[solver][retracer]") {
    auto reference = models::make_small_radiative_model();
    reference->nodes().set_qr(20, internal_watts);
    auto reference_solver_ptr = make_steady(reference);
    SteadyState& reference_solver = *reference_solver_ptr;
    reference_solver.initialize();
    reference_solver.solve();
    REQUIRE(reference_solver.converged());

    auto model = models::make_small_radiative_model();
    auto retracer = std::make_shared<FakeRetracer>();
    retracer->load = load_on(model, 20, internal_watts);
    auto solver_ptr = make_steady(model);
    SteadyState& solver = *solver_ptr;
    solver.set_retracer(retracer);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(models::max_difference(models::temperatures(model),
                                   models::temperatures(reference)) < 1e-9);

    SECTION("and enters both halves of a Crank-Nicolson step") {
        models::reset_temperatures(reference, 273.15);
        models::reset_temperatures(model, 273.15);
        CrankNicolson reference_transient(reference);
        set_transient(reference_transient);
        reference_transient.initialize();
        reference_transient.solve();

        CrankNicolson transient(model);
        set_transient(transient);
        transient.set_retracer(retracer);
        transient.initialize();
        transient.solve();
        REQUIRE(models::max_difference(models::temperatures(model),
                                       models::temperatures(reference)) < 1e-9);
    }
}

TEST_CASE("Crank-Nicolson calls the retracer before every callback",
          "[solver][retracer]") {
    auto model = models::make_small_radiative_model();
    auto retracer = std::make_shared<FakeRetracer>();
    log_callbacks(model, retracer);
    CrankNicolson solver(model);
    set_transient(solver);
    solver.set_retracer(retracer);
    solver.initialize();
    solver.solve();

    const std::string& calls = retracer->calls;
    REQUIRE(calls.front() == 'B');
    REQUIRE(calls.back() == 'E');
    std::size_t callbacks = 0;
    for (std::size_t pos = 0; pos < calls.size(); ++pos) {
        if (calls[pos] == 'C') {
            ++callbacks;
            REQUIRE(pos > 0);
            REQUIRE(calls[pos - 1] == 'U');
        }
        if (calls[pos] == 'U') {
            REQUIRE(calls.at(pos + 1) == 'C');
        }
    }
    REQUIRE(callbacks > 0);
    // Asked only once the solver's own tolerance holds: here once per step,
    // since the retracer accepts at once.
    REQUIRE(std::ranges::count(calls, 'A') == solver.report().steps);
}

TEST_CASE("A formula on a radiative coupling wins over the retracer",
          "[solver][retracer]") {
    auto model = models::make_small_radiative_model();
    model->formulas().add_formula("GR(15,25)", 0.5);
    auto retracer = std::make_shared<FakeRetracer>();
    retracer->model = model;
    retracer->written_node_1 = 15;
    retracer->written_node_2 = 25;
    retracer->written_value = 0.05;
    double seen = -1.0;
    model->python_extern_callback_solver_loop = [&model, &seen]() {
        seen = model->radiative_couplings().get_coupling_value(15, 25);
    };
    model->python_callbacks_active = true;

    auto solver_ptr = make_steady(model);
    SteadyState& solver = *solver_ptr;
    solver.set_retracer(retracer);
    solver.initialize();
    solver.solve();
    REQUIRE(seen == Catch::Approx(0.5));
}

TEST_CASE("The retracer can veto convergence", "[solver][retracer]") {
    auto model = models::make_small_radiative_model();
    auto plain_ptr = make_steady(model);
    SteadyState& plain = *plain_ptr;
    plain.initialize();
    plain.solve();
    const int plain_iterations = plain.report().iterations;

    SECTION("steady state: a refused iteration is not the last") {
        models::reset_temperatures(model, 273.15);
        auto retracer = std::make_shared<FakeRetracer>();
        retracer->refusals = 100;
        auto solver_ptr = make_steady(model);
        SteadyState& solver = *solver_ptr;
        solver.set_retracer(retracer);
        solver.initialize();
        solver.solve();
        REQUIRE(solver.report().status == SolveStatus::NOT_CONVERGED);
        REQUIRE(solver.report().iterations > plain_iterations);

        models::reset_temperatures(model, 273.15);
        auto lenient = std::make_shared<FakeRetracer>();
        lenient->refusals = 2;
        solver.set_retracer(lenient);
        solver.solve();
        REQUIRE(solver.converged());
        REQUIRE(solver.report().iterations == plain_iterations + 2);
    }

    SECTION("Crank-Nicolson: every step stops at max_iterations") {
        auto retracer = std::make_shared<FakeRetracer>();
        retracer->refusals = 1000;
        CrankNicolson solver(model);
        set_transient(solver);
        solver.set_retracer(retracer);
        solver.initialize();
        solver.solve();
        const auto& report = solver.report();
        REQUIRE(report.status == SolveStatus::NOT_CONVERGED);
        REQUIRE(report.nonconverged_steps == report.steps);
    }
}

TEST_CASE("The report carries the retracer counters", "[solver][retracer]") {
    auto model = models::make_small_radiative_model();
    auto retracer = std::make_shared<FakeRetracer>();
    auto solver_ptr = make_steady(model);
    SteadyState& solver = *solver_ptr;
    solver.set_retracer(retracer);
    solver.initialize();
    solver.solve();
    const auto& report = solver.report();
    REQUIRE(report.retraces == 3);
    REQUIRE(report.overflow_couplings == 1);
    REQUIRE(report.overflow_share == Catch::Approx(0.25));
    REQUIRE(report.to_string().contains("retraces"));
    REQUIRE(std::get<std::int64_t>(report.to_dict().at("retraces")) == 3);
}

TEST_CASE("A structural write-back deinitializes the solver",
          "[solver][retracer]") {
    auto model = models::make_small_radiative_model();
    auto retracer = std::make_shared<FakeRetracer>();
    retracer->changes_structure = true;

    CrankNicolson transient(model);
    set_transient(transient);
    transient.set_retracer(retracer);
    transient.initialize();
    transient.solve();
    REQUIRE_FALSE(transient.initialized());
    REQUIRE_THROWS_AS(transient.solve(), std::invalid_argument);

    auto steady_ptr = make_steady(model);
    SteadyState& steady = *steady_ptr;
    steady.set_retracer(retracer);
    steady.initialize();
    steady.solve();
    REQUIRE_FALSE(steady.initialized());
}

TEST_CASE("The radiative coupling output includes the overflow",
          "[solver][retracer]") {
    auto model = models::make_small_radiative_model();
    const auto idx_10 = index_of(model, 10);
    const auto idx_20 = index_of(model, 20);
    const auto size = static_cast<Index>(model->nodes().T_vector.size());
    auto retracer = std::make_shared<FakeRetracer>();
    retracer->overflow_couplings.resize(size, size);
    retracer->overflow_couplings.insert(std::min(idx_10, idx_20),
                                        std::max(idx_10, idx_20)) = 0.3;
    retracer->overflow_couplings.makeCompressed();

    CrankNicolson solver(model);
    Settings settings;
    settings.output_name = "overflow_run";
    settings.output_attributes = {pycanha::DataModelAttribute::KR};
    solver.set_settings(settings);
    solver.set_simulation_time(0.0, end_time, time_step, time_step);
    solver.set_retracer(retracer);
    solver.initialize();
    solver.solve();

    const auto& couplings = solver.output_model().radiative_couplings().at(0);
    REQUIRE(couplings.coeff(idx_10, idx_20) == Catch::Approx(0.3));
    REQUIRE(couplings.coeff(idx_20, idx_10) == Catch::Approx(0.3));
    // A coupling with a slot is unchanged.
    const auto idx_15 = index_of(model, 15);
    const auto idx_25 = index_of(model, 25);
    REQUIRE(couplings.coeff(idx_15, idx_25) == Catch::Approx(0.2));
}

TEST_CASE("Explicit zero radiative couplings survive compression",
          "[solver][retracer]") {
    auto model = models::make_small_radiative_model();
    model->add_radiative_coupling(10, 20, 0.0);
    auto& matrices = model->radiative_couplings().matrices();
    const Index stored = matrices.get_num_total_couplings();
    matrices.compress();
    REQUIRE(matrices.get_num_total_couplings() == stored);
    REQUIRE(matrices.coupling_exists_from_idxs(index_of(model, 10),
                                               index_of(model, 20)));

    auto solver_ptr = make_steady(model);
    SteadyState& solver = *solver_ptr;
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(matrices.get_num_total_couplings() == stored);
}

// NOLINTEND(readability-function-cognitive-complexity)
