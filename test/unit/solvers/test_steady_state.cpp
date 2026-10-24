#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/solvers/model_changed_error.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "solver_test_models.hpp"

namespace {

using pycanha::solvers::Backend;
using pycanha::solvers::EigenOrdering;
using pycanha::solvers::Factorization;
using pycanha::solvers::PardisoLevel;
using pycanha::solvers::PardisoOrdering;
using pycanha::solvers::Settings;
using pycanha::solvers::steady::SteadyState;
using pycanha::test_models::Model;
namespace models = pycanha::test_models;

// One backend configuration, as a set of settings.
struct Config {
    std::string name;
    Settings settings;
};

Config eigen(EigenOrdering ordering) {
    Config config{.name = std::string("EIGEN ") +
                          std::string(pycanha::solvers::to_string(ordering)),
                  .settings = {}};
    config.settings.backend = Backend::EIGEN;
    config.settings.eigen.ordering = ordering;
    return config;
}

Config eigen_cholesky() {
    Config config{.name = "EIGEN CHOLESKY", .settings = {}};
    config.settings.backend = Backend::EIGEN;
    config.settings.factorization = Factorization::CHOLESKY;
    return config;
}

Config pardiso(PardisoLevel level, std::optional<PardisoOrdering> ordering,
               std::optional<Factorization> factorization = std::nullopt) {
    Config config{
        .name = std::string("PARDISO ") +
                std::string(pycanha::solvers::to_string(level)) +
                (ordering.has_value()
                     ? " " + std::string(pycanha::solvers::to_string(*ordering))
                     : "") +
                (factorization.has_value() ? " LU" : ""),
        .settings = {}};
    config.settings.backend = Backend::PARDISO;
    config.settings.pardiso.level = level;
    config.settings.pardiso.ordering = ordering;
    config.settings.factorization = factorization;
    return config;
}

// Every backend configuration of this build. Cholesky only without
// radiation.
std::vector<Config> direct_configs(bool radiation) {
    std::vector<Config> configs = {eigen(EigenOrdering::COLAMD),
                                   eigen(EigenOrdering::AMD)};
    if (!radiation) {
        configs.push_back(eigen_cholesky());
    }
    if (pycanha::MKL_ENABLED) {
        configs.push_back(pardiso(PardisoLevel::TWO_LEVEL, std::nullopt));
        configs.push_back(pardiso(PardisoLevel::ONE_LEVEL,
                                  PardisoOrdering::NESTED_DISSECTION));
        configs.push_back(
            pardiso(PardisoLevel::ONE_LEVEL, PardisoOrdering::MIN_DEGREE));
        configs.push_back(
            pardiso(PardisoLevel::TWO_LEVEL, std::nullopt, Factorization::LU));
    }
    return configs;
}

// Cholesky is chosen explicitly, or by AUTO for PARDISO without radiation.
bool expects_cholesky(const Config& config, bool radiation) {
    const auto& settings = config.settings;
    if (settings.factorization == Factorization::CHOLESKY) {
        return true;
    }
    return !radiation && settings.backend == Backend::PARDISO &&
           !settings.factorization.has_value();
}

void configure(SteadyState& solver, Settings settings, double tolerance) {
    settings.max_iterations = 100;
    settings.tolerance = tolerance;
    solver.set_settings(settings);
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

// The backends of this build.
std::vector<Backend> available_backends() {
    if (pycanha::MKL_ENABLED) {
        return {Backend::EIGEN, Backend::PARDISO};
    }
    return {Backend::EIGEN};
}

Eigen::VectorXd solve_reference(const Model& model, double tolerance) {
    SteadyState solver(model);
    configure(solver, eigen(EigenOrdering::COLAMD).settings, tolerance);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    return models::temperatures(model);
}

}  // namespace

TEST_CASE("SteadyState solves a simple model", "[solver][steady]") {
    const auto backend = GENERATE(from_range(available_backends()));
    const auto model = models::make_small_radiative_model();
    SteadyState solver(model);
    Settings settings;
    settings.backend = backend;
    configure(solver, settings, 1e-6);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(has_small_model_solution(model));
}

TEST_CASE("SteadyState defaults", "[solver][steady]") {
    const auto model = models::make_small_radiative_model();
    SteadyState solver(model);
    const auto resolved = solver.resolved_settings();
    REQUIRE(resolved.backend ==
            (pycanha::MKL_ENABLED ? Backend::PARDISO : Backend::EIGEN));
    REQUIRE(resolved.pardiso.level == PardisoLevel::TWO_LEVEL);
    REQUIRE_FALSE(resolved.reuse_factors);
    REQUIRE(solver.settings().solver_type ==
            pycanha::solvers::SolverType::STEADY_STATE);
    REQUIRE(solver.name() == "SteadyState");
}

TEST_CASE(
    "Every steady-state configuration gives the same conduction "
    "solution",
    "[solver][steady]") {
    const auto model = models::make_plate(20, /*radiation=*/false);
    const auto reference = solve_reference(model, 1e-6);

    const auto config =
        GENERATE(from_range(direct_configs(/*radiation=*/false)));
    INFO(config.name);
    models::reset_temperatures(model, 293.15);
    SteadyState solver(model);
    configure(solver, config.settings, 1e-6);
    solver.initialize();
    REQUIRE(solver.initialized());
    REQUIRE(solver.uses_cholesky() ==
            expects_cholesky(config, /*radiation=*/false));
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-9);
    // Linear model: the second iteration reuses the factors.
    REQUIRE(solver.iterations() == 2);
    REQUIRE(solver.num_factorizations() == 1);
}

TEST_CASE(
    "Every steady-state configuration gives the same radiative "
    "solution",
    "[solver][steady]") {
    constexpr double tolerance = 1e-8;
    const auto model = models::make_plate(15, /*radiation=*/true);
    const auto reference = solve_reference(model, tolerance);

    const auto config =
        GENERATE(from_range(direct_configs(/*radiation=*/true)));
    INFO(config.name);
    models::reset_temperatures(model, 250.0);
    SteadyState solver(model);
    configure(solver, config.settings, tolerance);
    solver.initialize();
    REQUIRE_FALSE(solver.uses_cholesky());
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-6);
    // Radiation changes the matrix at every iteration.
    REQUIRE(solver.num_factorizations() == solver.iterations());
}

TEST_CASE("SteadyState refuses Cholesky with radiation", "[solver][steady]") {
    const auto model = models::make_small_radiative_model();
    SteadyState solver(model);
    solver.set_settings(eigen_cholesky().settings);
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
    REQUIRE_FALSE(solver.initialized());
}

TEST_CASE("SteadyState ignores the settings of the other backend",
          "[solver][steady]") {
    const auto model = models::make_small_radiative_model();
    SteadyState solver(model);
    Settings settings = eigen(EigenOrdering::AMD).settings;
    settings.pardiso.level = PardisoLevel::ONE_LEVEL;
    configure(solver, settings, 1e-6);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(has_small_model_solution(model));
}

TEST_CASE("SteadyState refuses PARDISO in a build without MKL",
          "[solver][steady]") {
    if (pycanha::MKL_ENABLED) {
        SKIP("MKL built");
    }
    const auto model = models::make_small_radiative_model();
    SteadyState solver(model);
    Settings settings;
    settings.backend = Backend::PARDISO;
    solver.set_settings(settings);
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
    REQUIRE_FALSE(solver.initialized());
}

TEST_CASE("SteadyState refactorizes when a coupling changes every iteration",
          "[solver][steady]") {
    const auto backend = GENERATE(from_range(available_backends()));
    const auto model = models::make_plate(10, /*radiation=*/false);
    models::install_callback(model, models::change_coupling_every_pass, 1, 2);

    SteadyState solver(model);
    Settings settings;
    settings.backend = backend;
    settings.max_iterations = 3;
    settings.tolerance = 1e-300;
    solver.set_settings(settings);
    solver.initialize();
    solver.solve();
    model->c_callbacks_active = false;
    REQUIRE(solver.num_factorizations() == 3);
}

TEST_CASE("SteadyState reports a floating group of nodes without crashing",
          "[solver][steady]") {
    // The matrix is singular: the factorization stops at a zero pivot or
    // PARDISO perturbs it. Both are reported as a failure and leave the
    // temperatures untouched.
    const auto config =
        GENERATE(from_range(direct_configs(/*radiation=*/true)));
    const bool radiation = GENERATE(false, true);
    INFO(config.name << (radiation ? ", radiation" : ", conduction"));
    const auto model = models::make_floating_model(radiation);
    SteadyState solver(model);
    configure(solver, config.settings, 1e-6);
    solver.initialize();
    solver.solve();
    REQUIRE_FALSE(solver.converged());
    REQUIRE(model->nodes().get_T(1) == 300.0);
}

TEST_CASE("SteadyState refuses to solve a model changed after initialize()",
          "[solver][steady]") {
    const auto model = models::make_plate(5, /*radiation=*/false);
    SteadyState solver(model);
    solver.initialize();
    model->add_conductive_coupling(1, 7, 0.5);

    REQUIRE_THROWS_AS(solver.solve(), pycanha::solvers::ModelChangedError);
    REQUIRE_FALSE(solver.converged());
    REQUIRE(model->nodes().get_T(1) == 293.15);

    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
}

TEST_CASE("SteadyState lifecycle", "[solver][steady]") {
    const auto model = models::make_plate(4, /*radiation=*/true);
    {
        const SteadyState never_initialized(model);
    }
    SteadyState solver(model);
    solver.deinitialize();
    solver.initialize();
    solver.initialize();
    REQUIRE(solver.initialized());
    solver.solve();
    REQUIRE(solver.converged());
    solver.deinitialize();
    solver.deinitialize();
    REQUIRE_FALSE(solver.initialized());
    solver.solve();
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
}

TEST_CASE("SteadyState freezes init settings while initialized",
          "[solver][steady]") {
    const auto model = models::make_plate(4, /*radiation=*/false);
    SteadyState solver(model);
    solver.initialize();
    // Live: picked up by the next solve.
    solver.set("max_iterations",
               pycanha::solvers::SettingValue{std::int64_t{1}});
    solver.solve();
    REQUIRE(solver.iterations() == 1);
    REQUIRE_THROWS_AS(
        solver.set("backend",
                   pycanha::solvers::SettingValue{std::string("EIGEN")}),
        std::runtime_error);
    solver.deinitialize();
    REQUIRE_NOTHROW(solver.set(
        "backend", pycanha::solvers::SettingValue{std::string("EIGEN")}));
}

TEST_CASE(
    "SteadyState solves a conduction model with empty radiative "
    "matrices",
    "[solver][steady]") {
    // Nodes added through the container leave the radiative matrices 0 x 0
    // until a radiative coupling is added.
    const auto config =
        GENERATE(from_range(direct_configs(/*radiation=*/false)));
    INFO(config.name);
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

    SteadyState solver(model);
    configure(solver, config.settings, 1e-6);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    // 10 W through two 2 W/K couplings in series.
    REQUIRE(model->nodes().get_T(2) == Catch::Approx(255.0));
    REQUIRE(model->nodes().get_T(1) == Catch::Approx(260.0));
}

TEST_CASE("SteadyState applies PARDISO settings", "[solver][steady]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    constexpr double tolerance = 1e-8;
    const auto model = models::make_plate(12, /*radiation=*/true);
    const auto reference = solve_reference(model, tolerance);

    models::reset_temperatures(model, 250.0);
    SteadyState solver(model);
    Settings settings;
    settings.backend = Backend::PARDISO;
    // Minimum degree ordering and one-level factorization through iparm.
    settings.pardiso.iparm = std::map<int, int>{{1, 0}, {23, 0}};
    settings.threads = 2;
    configure(solver, settings, tolerance);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-9);

    solver.deinitialize();
    settings.pardiso.iparm = std::map<int, int>{{64, 1}};
    solver.set_settings(settings);
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Factor reuse (PARDISO only)
// ---------------------------------------------------------------------------

TEST_CASE("Factor reuse needs PARDISO", "[solver][steady][reuse]") {
    const auto model = models::make_small_radiative_model();
    SteadyState solver(model);
    Settings settings;
    settings.backend = Backend::EIGEN;
    settings.reuse_factors = true;
    solver.set_settings(settings);
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
    REQUIRE_FALSE(solver.initialized());
}

TEST_CASE("Factor reuse matches a direct solve on a radiative plate",
          "[solver][steady][reuse]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    constexpr double tolerance = 1e-8;
    const auto model = models::make_plate(20, /*radiation=*/true);
    const auto reference = solve_reference(model, tolerance);

    // The iteration stops at a relative residual of 10^-L (iparm[3] = 10 L +
    // 1), and that residual stays in the converged temperatures.
    const auto ordering = GENERATE(PardisoOrdering::MIN_DEGREE,
                                   PardisoOrdering::NESTED_DISSECTION);
    const int iparm_3 = GENERATE(61, 91);
    models::reset_temperatures(model, 250.0);
    SteadyState solver(model);
    Settings settings;
    settings.reuse_factors = true;
    settings.pardiso.ordering = ordering;
    settings.pardiso.iparm = std::map<int, int>{{3, iparm_3}};
    configure(solver, settings, tolerance);
    solver.initialize();
    REQUIRE_FALSE(solver.uses_cholesky());
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-6);
    // One factorization, the other iterations iterate on it.
    REQUIRE(solver.num_factorizations() == 1);
}

TEST_CASE("Factor reuse uses Cholesky and CG on a conduction model",
          "[solver][steady][reuse]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    const auto model = models::make_plate(15, /*radiation=*/false);
    const auto reference = solve_reference(model, 1e-6);
    models::reset_temperatures(model, 293.15);

    SteadyState solver(model);
    Settings settings;
    settings.reuse_factors = true;
    solver.set_settings(settings);
    solver.initialize();
    REQUIRE(solver.uses_cholesky());
    REQUIRE(solver.resolved_settings().pardiso.level ==
            PardisoLevel::ONE_LEVEL);
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-9);
}

TEST_CASE("Factor reuse never solves with factors of an older matrix",
          "[solver][steady][reuse]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    // Iteration 2 sees a doubled coupling and is solved with CG on the
    // iteration-1 factors, which PARDISO keeps. Iteration 3 sees the same
    // values as iteration 2 but must not take the solve-only shortcut: the
    // factors still belong to the iteration-1 matrix.
    const auto model = models::make_plate(12, /*radiation=*/false);
    const int node_1 = 1;
    const int node_2 = 2;
    const double original =
        model->conductive_couplings().get_coupling_value(node_1, node_2);
    model->conductive_couplings().set_coupling_value(node_1, node_2,
                                                     2.0 * original);
    const auto reference = solve_reference(model, 1e-9);
    model->conductive_couplings().set_coupling_value(node_1, node_2, original);
    models::reset_temperatures(model, 293.15);

    models::install_callback(model, models::double_coupling_once, node_1,
                             node_2);
    SteadyState solver(model);
    Settings settings;
    settings.reuse_factors = true;
    settings.max_iterations = 3;
    settings.tolerance = 1e-300;
    solver.set_settings(settings);
    solver.initialize();
    solver.solve();
    model->c_callbacks_active = false;

    REQUIRE(solver.num_factorizations() == 1);
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-4);
}
