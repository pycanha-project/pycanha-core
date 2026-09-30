#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/solvers/linear_solver.hpp"
#include "pycanha-core/solvers/sslu.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "solver_test_models.hpp"

namespace {

using pycanha::DirectSolverType;
using pycanha::SolverEngine;
using pycanha::test_models::Model;
namespace models = pycanha::test_models;

struct DirectConfig {
    SolverEngine engine = SolverEngine::EIGEN;
    DirectSolverType type = DirectSolverType::DEFAULT;
    bool allow_cholesky = true;

    [[nodiscard]] std::string name() const {
        return std::string(pycanha::to_string(engine)) + " " +
               std::string(pycanha::to_string(type)) +
               (allow_cholesky ? "" : " (no Cholesky)");
    }
};

// Every engine and type available in this build. LDLT only without
// radiation.
std::vector<DirectConfig> direct_configs(bool radiation) {
    std::vector<DirectConfig> configs = {
        {.engine = SolverEngine::EIGEN, .type = DirectSolverType::COLAMD},
        {.engine = SolverEngine::EIGEN, .type = DirectSolverType::AMD},
    };
    if (!radiation) {
        configs.push_back(
            {.engine = SolverEngine::EIGEN, .type = DirectSolverType::LDLT});
    }
    if (pycanha::MKL_ENABLED) {
        for (const auto type :
             {DirectSolverType::TWO_LEVEL, DirectSolverType::ONE_LEVEL,
              DirectSolverType::MIN_DEGREE}) {
            configs.push_back({.engine = SolverEngine::MKL, .type = type});
        }
        configs.push_back({.engine = SolverEngine::MKL,
                           .type = DirectSolverType::TWO_LEVEL,
                           .allow_cholesky = false});
    }
    return configs;
}

void configure(pycanha::SSLU& solver, const DirectConfig& config,
               double abstol) {
    solver.engine = config.engine;
    solver.solver_type = config.type;
    solver.allow_cholesky = config.allow_cholesky;
    solver.max_iters = 100;
    solver.abstol_temp = abstol;
}

// The reference steady state of the small radiative model.
bool has_small_model_solution(const Model& model) {
    auto& nodes = model->nodes();
    constexpr double tolerance = 1e-2;
    return std::abs(nodes.get_T(10) - 132.38706) < tolerance &&
           std::abs(nodes.get_T(15) - 306.56526) < tolerance &&
           std::abs(nodes.get_T(20) - 111.78443) < tolerance &&
           std::abs(nodes.get_T(25) - 200.32387) < tolerance &&
           nodes.get_T(99) == 3.15;
}

// The engines of this build.
std::vector<SolverEngine> available_engines() {
#if PYCANHA_USE_MKL
    return {SolverEngine::EIGEN, SolverEngine::MKL};
#else
    return {SolverEngine::EIGEN};
#endif
}

Eigen::VectorXd solve_reference(const Model& model, double abstol) {
    pycanha::SSLU solver(model);
    configure(solver,
              {.engine = SolverEngine::EIGEN, .type = DirectSolverType::COLAMD},
              abstol);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.solver_converged);
    return models::temperatures(model);
}

}  // namespace

TEST_CASE("SSLU solves a simple model", "[solver][sslu]") {
    const auto engine = GENERATE(from_range(available_engines()));
    const auto model = models::make_small_radiative_model();
    pycanha::SSLU solver(model);
    solver.engine = engine;
    solver.max_iters = 100;
    solver.abstol_temp = 1e-6;
    solver.initialize();
    solver.solve();
    REQUIRE(solver.solver_converged);
    REQUIRE(has_small_model_solution(model));
}

TEST_CASE("SSLU defaults", "[solver][sslu]") {
    const auto model = models::make_small_radiative_model();
    const pycanha::SSLU solver(model);
    REQUIRE(solver.engine == pycanha::default_solver_engine());
    REQUIRE(solver.engine ==
            (pycanha::MKL_ENABLED ? SolverEngine::MKL : SolverEngine::EIGEN));
    REQUIRE(solver.solver_type == DirectSolverType::DEFAULT);
    REQUIRE(solver.pardiso_iparm_3 == 0);
    REQUIRE(solver.allow_cholesky);
}

TEST_CASE("Every SSLU type gives the same conduction solution",
          "[solver][sslu]") {
    const auto model = models::make_plate(20, /*radiation=*/false);
    const auto reference = solve_reference(model, 1e-6);

    const auto config =
        GENERATE(from_range(direct_configs(/*radiation=*/false)));
    INFO(config.name());
    models::reset_temperatures(model, 293.15);
    pycanha::SSLU solver(model);
    configure(solver, config, 1e-6);
    solver.initialize();
    REQUIRE(solver.solver_initialized);
    REQUIRE(solver.uses_cholesky() ==
            (config.type == DirectSolverType::LDLT ||
             (config.engine == SolverEngine::MKL && config.allow_cholesky)));
    solver.solve();
    REQUIRE(solver.solver_converged);
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-9);
    // Linear model: the second pass reuses the factors.
    REQUIRE(solver.solver_iter == 1);
    REQUIRE(solver.num_factorizations() == 1);
}

TEST_CASE("Every SSLU type gives the same radiative solution",
          "[solver][sslu]") {
    constexpr double abstol = 1e-8;
    const auto model = models::make_plate(15, /*radiation=*/true);
    const auto reference = solve_reference(model, abstol);

    const auto config =
        GENERATE(from_range(direct_configs(/*radiation=*/true)));
    INFO(config.name());
    models::reset_temperatures(model, 250.0);
    pycanha::SSLU solver(model);
    configure(solver, config, abstol);
    solver.initialize();
    REQUIRE_FALSE(solver.uses_cholesky());
    solver.solve();
    REQUIRE(solver.solver_converged);
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-6);
    // Radiation changes the matrix at every pass.
    REQUIRE(solver.num_factorizations() == solver.solver_iter + 1);
}

TEST_CASE("SSLU rejects LDLT with radiation", "[solver][sslu]") {
    const auto model = models::make_small_radiative_model();
    pycanha::SSLU solver(model);
    solver.engine = SolverEngine::EIGEN;
    solver.solver_type = DirectSolverType::LDLT;
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
    REQUIRE_FALSE(solver.solver_initialized);
}

TEST_CASE("SSLU rejects a type of the other engine", "[solver][sslu]") {
    const auto model = models::make_small_radiative_model();
    pycanha::SSLU solver(model);
    solver.engine = SolverEngine::EIGEN;
    solver.solver_type = DirectSolverType::TWO_LEVEL;
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
    solver.engine = SolverEngine::MKL;
    solver.solver_type = DirectSolverType::COLAMD;
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
    REQUIRE_FALSE(solver.solver_initialized);
}

TEST_CASE("SSLU rejects the iterative step", "[solver][sslu]") {
    const auto model = models::make_small_radiative_model();
    pycanha::SSLU solver(model);
    solver.pardiso_iparm_3 = 61;
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
    REQUIRE_FALSE(solver.solver_initialized);
}

#if PYCANHA_USE_MKL
TEST_CASE("SSLU accepts the MKL engine when built with MKL", "[solver][sslu]") {
    const auto model = models::make_small_radiative_model();
    pycanha::SSLU solver(model);
    solver.engine = SolverEngine::MKL;
    solver.initialize();
    REQUIRE(solver.solver_initialized);
}
#else
TEST_CASE("SSLU rejects the MKL engine when built without MKL",
          "[solver][sslu]") {
    const auto model = models::make_small_radiative_model();
    pycanha::SSLU solver(model);
    solver.engine = SolverEngine::MKL;
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
    REQUIRE_FALSE(solver.solver_initialized);
}
#endif

TEST_CASE("SSLU refactorises when a coupling changes every pass",
          "[solver][sslu]") {
    const auto engine = GENERATE(from_range(available_engines()));
    const auto model = models::make_plate(10, /*radiation=*/false);
    models::install_callback(model, models::change_coupling_every_pass, 1, 2);

    pycanha::SSLU solver(model);
    solver.engine = engine;
    solver.max_iters = 3;
    solver.abstol_temp = 0.0;
    solver.initialize();
    solver.solve();
    model->c_callbacks_active = false;
    REQUIRE(solver.num_factorizations() == 3);
}

TEST_CASE("SSLU reports a floating group of nodes without crashing",
          "[solver][sslu]") {
    // The matrix is singular: the factorisation stops at a zero pivot or
    // PARDISO perturbs it. Both are reported as a failure and leave the
    // temperatures untouched.
    const auto config =
        GENERATE(from_range(direct_configs(/*radiation=*/true)));
    const bool radiation = GENERATE(false, true);
    INFO(config.name() << (radiation ? ", radiation" : ", conduction"));
    const auto model = models::make_floating_model(radiation);
    pycanha::SSLU solver(model);
    configure(solver, config, 1e-6);
    solver.initialize();
    solver.solve();
    REQUIRE_FALSE(solver.solver_converged);
    REQUIRE(model->nodes().get_T(1) == 300.0);
}

TEST_CASE("SSLU refuses to solve a model changed after initialize()",
          "[solver][sslu]") {
    const auto model = models::make_plate(5, /*radiation=*/false);
    pycanha::SSLU solver(model);
    solver.initialize();
    model->add_conductive_coupling(1, 7, 0.5);

    solver.solve();
    REQUIRE_FALSE(solver.solver_converged);
    REQUIRE(model->nodes().get_T(1) == 293.15);

    solver.initialize();
    solver.solve();
    REQUIRE(solver.solver_converged);
}

TEST_CASE("SSLU lifecycle", "[solver][sslu]") {
    const auto model = models::make_plate(4, /*radiation=*/true);
    {
        const pycanha::SSLU never_initialized(model);
    }
    pycanha::SSLU solver(model);
    solver.deinitialize();
    solver.initialize();
    solver.initialize();
    REQUIRE(solver.solver_initialized);
    solver.solve();
    REQUIRE(solver.solver_converged);
    solver.deinitialize();
    solver.deinitialize();
    REQUIRE_FALSE(solver.solver_initialized);
    solver.solve();
    solver.initialize();
    solver.solve();
    REQUIRE(solver.solver_converged);
}

TEST_CASE("SSLU solves a conduction model with empty radiative matrices",
          "[solver][sslu]") {
    // Nodes added through the container leave the radiative matrices 0 x 0
    // until a radiative coupling is added.
    const auto config =
        GENERATE(from_range(direct_configs(/*radiation=*/false)));
    INFO(config.name());
    auto model =
        std::make_shared<pycanha::ThermalMathematicalModel>("container");
    for (const int num : {1, 2}) {
        pycanha::Node node(num);
        node.set_T(300.0);
        model->nodes().add_node(node);
    }
    pycanha::Node sink(3);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(250.0);
    model->nodes().add_node(sink);
    model->add_conductive_coupling(1, 2, 2.0);
    model->add_conductive_coupling(2, 3, 2.0);
    model->nodes().set_qi(1, 10.0);

    pycanha::SSLU solver(model);
    configure(solver, config, 1e-6);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.solver_converged);
    // 10 W through two 2 W/K couplings in series.
    REQUIRE(model->nodes().get_T(2) == Catch::Approx(255.0));
    REQUIRE(model->nodes().get_T(1) == Catch::Approx(260.0));
}

TEST_CASE("SSLU applies PARDISO settings", "[solver][sslu]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    constexpr double abstol = 1e-8;
    const auto model = models::make_plate(12, /*radiation=*/true);
    const auto reference = solve_reference(model, abstol);

    models::reset_temperatures(model, 250.0);
    pycanha::SSLU solver(model);
    solver.engine = SolverEngine::MKL;
    solver.max_iters = 100;
    solver.abstol_temp = abstol;
    // Minimum degree ordering, one-level factorisation, two threads.
    solver.pardiso_iparm_overrides = {{1, 0}, {23, 0}};
    solver.mkl_threads = 2;
    solver.initialize();
    solver.solve();
    REQUIRE(solver.solver_converged);
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-9);

    solver.pardiso_iparm_overrides = {{64, 1}};
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
}
