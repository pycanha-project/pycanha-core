#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/ts.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace {

std::shared_ptr<pycanha::ThermalMathematicalModel> make_probe_model() {
    auto model =
        std::make_shared<pycanha::ThermalMathematicalModel>("probe_model");

    pycanha::Node node_10(10);
    pycanha::Node node_20(20);
    pycanha::Node node_99(99);

    node_10.set_T(300.0);
    node_20.set_T(310.0);
    node_99.set_T(4.0);

    node_10.set_C(5.0);
    node_20.set_C(7.5);
    node_99.set_type(pycanha::BOUNDARY_NODE);

    model->add_node(node_10);
    model->add_node(node_20);
    model->add_node(node_99);

    model->add_conductive_coupling(10, 20, 2.0);
    model->add_conductive_coupling(10, 99, 3.0);
    model->add_radiative_coupling(20, 99, 4.0);

    return model;
}

struct NonZeroCounts {
    pycanha::Index kldd;
    pycanha::Index krdd;
    pycanha::Index kldb;
    pycanha::Index krdb;
};

class ProbeTransientSolver final : public pycanha::TransientSolver {
  public:
    explicit ProbeTransientSolver(
        std::shared_ptr<pycanha::ThermalMathematicalModel> model)
        : TransientSolver(std::move(model)) {
        solver_name = "probe_transient";
    }

    void initialize() override {
        initialize_common();
        solver_initialized = true;
    }

    void solve() override {}

    void deinitialize() override { solver_initialized = false; }

    void set_output_name(std::string name) {
        output_model_name = std::move(name);
    }

    void add_output(pycanha::DataModelAttribute attribute) {
        output_config.add(attribute);
    }

    void set_state(double current_time, int current_iter,
                   int current_output_index) {
        time = current_time;
        tmm.time = current_time;
        time_iter = current_iter;
        idata_out = current_output_index;
    }

    void save_first_or_last_output() { outputs_first_last(); }

    void save_regular_output() { outputs(); }

    void restart_for_test() { restart_solve(); }

    void expand_couplings_for_test() { expand_coupling_matrices_with_zeros(); }

    void restore_couplings_for_test() { restore_expanded_coupling_matrices(); }

    void set_converged(bool value) noexcept { solver_converged = value; }

    [[nodiscard]] int wait_n_dtimes_value() const noexcept {
        return wait_n_dtimes;
    }

    [[nodiscard]] int num_outputs_value() const noexcept { return num_outputs; }

    [[nodiscard]] double current_time() const noexcept { return time; }

    [[nodiscard]] int current_time_iter() const noexcept { return time_iter; }

    [[nodiscard]] int current_output_index() const noexcept {
        return idata_out;
    }

    [[nodiscard]] NonZeroCounts coupling_nonzero_counts() const noexcept {
        return {KLdd.nonZeros(), KRdd.nonZeros(), KLdb.nonZeros(),
                KRdb.nonZeros()};
    }
};

}  // namespace

TEST_CASE("TransientSolver validates simulation time arguments",
          "[solver][transient]") {
    ProbeTransientSolver solver(make_probe_model());

    REQUIRE_THROWS_AS(solver.set_simulation_time(2.0, 1.0, 1.0, 0.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(solver.set_simulation_time(0.0, 1.0, 0.0, 0.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(solver.set_simulation_time(0.0, 1.0, 1.0, -0.5),
                      std::invalid_argument);
}

TEST_CASE("TransientSolver initialization requires timing and output name",
          "[solver][transient]") {
    ProbeTransientSolver solver(make_probe_model());

    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);

    solver.set_simulation_time(0.0, 3.0, 1.0, 2.0);
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
}

TEST_CASE("TransientSolver allocates and writes configured outputs",
          "[solver][transient]") {
    auto model = make_probe_model();
    ProbeTransientSolver solver(model);

    solver.set_output_name("probe_output");
    solver.add_output(pycanha::DataModelAttribute::KL);
    solver.add_output(pycanha::DataModelAttribute::KR);
    solver.set_simulation_time(0.0, 3.0, 1.0, 2.0);
    solver.initialize();

    REQUIRE(solver.solver_initialized);
    REQUIRE(solver.wait_n_dtimes_value() == 2);
    REQUIRE(solver.num_outputs_value() == 3);

    auto& output = solver.output_model();
    REQUIRE(output.node_numbers() == std::vector<pycanha::Index>{10, 20, 99});
    REQUIRE(output.T().num_timesteps() == 3);
    REQUIRE(output.T().num_columns() == 3);
    REQUIRE(output.conductive_couplings().num_timesteps() == 0);
    REQUIRE(output.radiative_couplings().num_timesteps() == 0);

    solver.set_state(0.0, 0, 0);
    solver.save_first_or_last_output();
    solver.save_first_or_last_output();

    REQUIRE(output.T().times()(0) == Catch::Approx(0.0));
    REQUIRE(output.T().values()(0, 0) == Catch::Approx(300.0));
    REQUIRE(output.T().values()(0, 1) == Catch::Approx(310.0));
    REQUIRE(output.T().values()(0, 2) == Catch::Approx(4.0));
    REQUIRE(output.conductive_couplings().num_timesteps() == 1);
    REQUIRE(output.radiative_couplings().num_timesteps() == 1);
    REQUIRE(output.conductive_couplings().time_at(0) == Catch::Approx(0.0));
    REQUIRE(output.radiative_couplings().time_at(0) == Catch::Approx(0.0));
    REQUIRE(output.conductive_couplings().at(0).coeff(0, 1) ==
            Catch::Approx(2.0));
    REQUIRE(output.conductive_couplings().at(0).coeff(1, 0) ==
            Catch::Approx(2.0));
    REQUIRE(output.conductive_couplings().at(0).coeff(0, 2) ==
            Catch::Approx(3.0));
    REQUIRE(output.radiative_couplings().at(0).coeff(1, 2) ==
            Catch::Approx(4.0));

    REQUIRE(model->nodes().set_T(10, 301.0));
    REQUIRE(model->nodes().set_T(20, 311.0));
    REQUIRE(model->nodes().set_T(99, 5.0));
    solver.set_state(2.0, 1, 0);
    solver.save_regular_output();

    REQUIRE(output.T().times()(1) == Catch::Approx(2.0));
    REQUIRE(output.T().values()(1, 0) == Catch::Approx(301.0));
    REQUIRE(output.T().values()(1, 1) == Catch::Approx(311.0));
    REQUIRE(output.T().values()(1, 2) == Catch::Approx(5.0));
    REQUIRE(output.conductive_couplings().num_timesteps() == 2);
    REQUIRE(output.radiative_couplings().num_timesteps() == 2);

    REQUIRE(model->nodes().set_T(10, 302.0));
    REQUIRE(model->nodes().set_T(20, 312.0));
    REQUIRE(model->nodes().set_T(99, 6.0));
    solver.set_state(3.0, 2, 1);
    solver.save_first_or_last_output();

    REQUIRE(output.T().times()(2) == Catch::Approx(3.0));
    REQUIRE(output.T().values()(2, 0) == Catch::Approx(302.0));
    REQUIRE(output.T().values()(2, 1) == Catch::Approx(312.0));
    REQUIRE(output.T().values()(2, 2) == Catch::Approx(6.0));
    REQUIRE(output.conductive_couplings().num_timesteps() == 3);
    REQUIRE(output.radiative_couplings().num_timesteps() == 3);
    REQUIRE(output.conductive_couplings().time_at(2) == Catch::Approx(3.0));
    REQUIRE(output.radiative_couplings().time_at(2) == Catch::Approx(3.0));
}

TEST_CASE("TransientSolver restart resets runtime state",
          "[solver][transient]") {
    auto model = make_probe_model();
    ProbeTransientSolver solver(model);

    REQUIRE_THROWS_AS(solver.restart_for_test(), std::invalid_argument);

    solver.set_output_name("restart_output");
    solver.set_simulation_time(0.0, 3.0, 1.0, 2.0);
    solver.initialize();

    solver.set_converged(true);
    solver.set_state(2.5, 4, 2);
    solver.restart_for_test();

    REQUIRE_FALSE(solver.solver_converged);
    REQUIRE(solver.current_time() == Catch::Approx(0.0));
    REQUIRE(model->time == Catch::Approx(0.0));
    REQUIRE(solver.current_time_iter() == -1);
    REQUIRE(solver.current_output_index() == 0);
}

TEST_CASE("Solver expands and restores coupling sparsity patterns",
          "[solver][transient]") {
    ProbeTransientSolver solver(make_probe_model());

    solver.set_output_name("expand_restore_output");
    solver.set_simulation_time(0.0, 3.0, 1.0, 2.0);
    solver.initialize();

    const NonZeroCounts before = solver.coupling_nonzero_counts();
    solver.expand_couplings_for_test();
    const NonZeroCounts expanded = solver.coupling_nonzero_counts();

    REQUIRE(expanded.kldd == expanded.krdd);
    REQUIRE(expanded.kldb == expanded.krdb);
    REQUIRE(expanded.kldd >= before.kldd);
    REQUIRE(expanded.krdd >= before.krdd);
    REQUIRE(expanded.kldb >= before.kldb);
    REQUIRE(expanded.krdb >= before.krdb);

    solver.restore_couplings_for_test();
    const NonZeroCounts restored = solver.coupling_nonzero_counts();

    REQUIRE(restored.kldd == before.kldd);
    REQUIRE(restored.krdd == before.krdd);
    REQUIRE(restored.kldb == before.kldb);
    REQUIRE(restored.krdb == before.krdb);
}