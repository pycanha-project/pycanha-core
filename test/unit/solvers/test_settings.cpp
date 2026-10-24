#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "solver_test_models.hpp"

namespace {

using pycanha::DataModelAttribute;
using pycanha::solvers::Backend;
using pycanha::solvers::EigenOrdering;
using pycanha::solvers::Factorization;
using pycanha::solvers::Mutability;
using pycanha::solvers::NotApplied;
using pycanha::solvers::Origin;
using pycanha::solvers::PardisoLevel;
using pycanha::solvers::PardisoOrdering;
using pycanha::solvers::ResolveContext;
using pycanha::solvers::ResolvedSettings;
using pycanha::solvers::Settings;
using pycanha::solvers::SettingValue;
using pycanha::solvers::SolverType;
namespace solvers = pycanha::solvers;

// Contexts that do not depend on how the library was built: PARDISO is
// available and the model has no radiation unless the name says otherwise.
ResolveContext context(SolverType type) {
    return {
        .solver_type = type, .has_radiation = false, .pardiso_available = true};
}
ResolveContext steady() { return context(SolverType::STEADY_STATE); }
ResolveContext crank_nicolson() { return context(SolverType::CRANK_NICOLSON); }
ResolveContext steady_without_pardiso() {
    ResolveContext without = steady();
    without.pardiso_available = false;
    return without;
}
ResolveContext steady_with_radiation() {
    ResolveContext radiative = steady();
    radiative.has_radiation = true;
    return radiative;
}
ResolveContext steady_without_model() {
    ResolveContext unknown = steady();
    unknown.has_radiation.reset();
    return unknown;
}

// Settings with the time step that transient solvers need.

Settings with_time_step() {
    Settings settings;
    settings.time_step = 10.0;
    return settings;
}

// The message of the std::invalid_argument resolve() throws, empty when it
// does not throw.
std::string contradiction_of(const Settings& settings,
                             const ResolveContext& resolve_context) {
    try {
        static_cast<void>(solvers::resolve(settings, resolve_context));
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}

bool contains(const std::string& text, std::string_view part) {
    return text.contains(part);
}

// A doc, at least one solver type, and a group that prefixes the name.
bool is_documented(const solvers::SettingField& field) {
    const bool applies_somewhere =
        field.applies_to(SolverType::STEADY_STATE) ||
        field.applies_to(SolverType::CRANK_NICOLSON) ||
        field.applies_to(SolverType::JACOBIAN_PROPAGATION);
    const bool group_matches = field.group.empty()
                                   ? !field.name.contains('.')
                                   : field.name.starts_with(field.group);
    return !field.doc.empty() && applies_somewhere && group_matches;
}

// Every field set, to a value different from its default.
Settings fully_set() {
    Settings settings;
    settings.solver_type = SolverType::CRANK_NICOLSON;
    settings.tolerance = 1e-4;
    settings.max_iterations = 20;
    settings.backend = Backend::EIGEN;
    settings.factorization = Factorization::LU;
    settings.reuse_factors = true;
    settings.threads = 4;
    settings.start_time = 10.0;
    settings.end_time = 100.0;
    settings.time_step = 5.0;
    settings.output_interval = 20.0;
    settings.output_attributes =
        std::set{DataModelAttribute::T, DataModelAttribute::QI};
    settings.output_name = "run";
    settings.capacity_delta = 1e-6;
    settings.pardiso.ordering = PardisoOrdering::MIN_DEGREE;
    settings.pardiso.level = PardisoLevel::ONE_LEVEL;
    settings.pardiso.iparm = std::map<int, int>{{3, 61}};
    settings.pardiso.verbose = true;
    settings.eigen.ordering = EigenOrdering::AMD;
    return settings;
}

}  // namespace

TEST_CASE("Every setting is documented once", "[solver][settings]") {
    std::set<std::string_view> names;
    for (const auto& field : solvers::setting_fields()) {
        names.insert(field.name);
    }
    REQUIRE(names.size() == solvers::setting_fields().size());
    REQUIRE(std::ranges::all_of(solvers::setting_fields(), is_documented));
    REQUIRE(solvers::setting_field("pardiso.level").mutability ==
            Mutability::INIT);
    REQUIRE_THROWS_AS(solvers::setting_field("relaxation"),
                      std::invalid_argument);
}

TEST_CASE("Unset settings resolve to the defaults", "[solver][settings]") {
    const ResolvedSettings with_mkl = solvers::resolve({}, steady());
    REQUIRE(with_mkl.tolerance == Catch::Approx(1e-3));
    REQUIRE(with_mkl.max_iterations == 5);
    REQUIRE(with_mkl.backend == Backend::PARDISO);
    REQUIRE(with_mkl.pardiso.level == PardisoLevel::TWO_LEVEL);
    REQUIRE(with_mkl.pardiso.ordering == PardisoOrdering::NESTED_DISSECTION);
    REQUIRE(with_mkl.origin("backend") == Origin::DEFAULT);

    const ResolvedSettings without_mkl =
        solvers::resolve({}, steady_without_pardiso());
    REQUIRE(without_mkl.backend == Backend::EIGEN);
    REQUIRE(without_mkl.eigen.ordering == EigenOrdering::COLAMD);
}

TEST_CASE("Transient defaults", "[solver][settings]") {
    const auto type =
        GENERATE(SolverType::CRANK_NICOLSON, SolverType::JACOBIAN_PROPAGATION);
    const ResolvedSettings resolved =
        solvers::resolve(with_time_step(), context(type));
    REQUIRE(resolved.output_name == (type == SolverType::CRANK_NICOLSON
                                         ? "crank_nicolson"
                                         : "jacobian_propagation"));
    REQUIRE(resolved.output_attributes ==
            std::set<DataModelAttribute>{DataModelAttribute::T});
    REQUIRE(resolved.capacity_delta == Catch::Approx(1e-7));
    REQUIRE(resolved.output_interval == 0.0);
    REQUIRE(resolved.factorization == Factorization::LU);
}

TEST_CASE("Factor reuse selects a stall-safe one-level factorization",
          "[solver][settings]") {
    Settings settings;
    settings.reuse_factors = true;
    settings.tolerance = 1e-4;
    const ResolvedSettings resolved = solvers::resolve(settings, steady());
    REQUIRE(resolved.pardiso.level == PardisoLevel::ONE_LEVEL);
    REQUIRE(resolved.pardiso.ordering == PardisoOrdering::MIN_DEGREE);
    REQUIRE(resolved.origin("pardiso.level") == Origin::DEFAULT);
    REQUIRE(resolved.warnings.empty());
}

TEST_CASE("An iterating iparm[3] selects the one-level factorization",
          "[solver][settings]") {
    Settings settings;
    settings.pardiso.iparm = std::map<int, int>{{3, 61}};
    const ResolvedSettings resolved = solvers::resolve(settings, steady());
    REQUIRE(resolved.pardiso.level == PardisoLevel::ONE_LEVEL);
    REQUIRE(resolved.pardiso.ordering == PardisoOrdering::MIN_DEGREE);

    // iparm[3] = 0 does not iterate.
    settings.pardiso.iparm = std::map<int, int>{{3, 0}};
    REQUIRE(solvers::resolve(settings, steady()).pardiso.level ==
            PardisoLevel::TWO_LEVEL);

    settings.pardiso.iparm = std::map<int, int>{{3, 61}};
    settings.pardiso.level = PardisoLevel::TWO_LEVEL;
    REQUIRE(contains(contradiction_of(settings, steady()),
                     "pardiso.iparm[3] iterates"));
}

TEST_CASE("PARDISO level and ordering default from each other",
          "[solver][settings]") {
    Settings min_degree;
    min_degree.pardiso.ordering = PardisoOrdering::MIN_DEGREE;
    REQUIRE(solvers::resolve(min_degree, steady()).pardiso.level ==
            PardisoLevel::ONE_LEVEL);

    Settings one_level;
    one_level.pardiso.level = PardisoLevel::ONE_LEVEL;
    REQUIRE(solvers::resolve(one_level, steady()).pardiso.ordering ==
            PardisoOrdering::MIN_DEGREE);

    // Explicit nested dissection one-level is accepted, with a warning.
    one_level.pardiso.ordering = PardisoOrdering::NESTED_DISSECTION;
    const ResolvedSettings risky = solvers::resolve(one_level, steady());
    REQUIRE(risky.warnings.size() == 1);
    REQUIRE(contains(risky.warnings.front(), "livelock"));
}

TEST_CASE("AUTO factorization follows the backend, solver and model",
          "[solver][settings]") {
    const auto factorization = [](const ResolveContext& resolve_context) {
        Settings settings = with_time_step();
        settings.factorization = Factorization::AUTO;
        return solvers::resolve(settings, resolve_context).factorization;
    };
    REQUIRE(factorization(steady()) == Factorization::CHOLESKY);
    REQUIRE(factorization(steady_with_radiation()) == Factorization::LU);
    REQUIRE(factorization(steady_without_model()) == Factorization::AUTO);
    REQUIRE(factorization(steady_without_pardiso()) == Factorization::LU);
    REQUIRE(factorization(crank_nicolson()) == Factorization::LU);
}

TEST_CASE("Contradictions are refused", "[solver][settings]") {
    struct Case {
        std::string_view name;
        std::function<void(Settings&)> change;
        ResolveContext context;
        std::string_view message;
    };
    const auto test_case = GENERATE(
        Case{"PARDISO without MKL",
             [](Settings& settings) { settings.backend = Backend::PARDISO; },
             steady_without_pardiso(), "needs a build with MKL"},
        Case{"reuse with Eigen",
             [](Settings& settings) {
                 settings.backend = Backend::EIGEN;
                 settings.reuse_factors = true;
             },
             steady(), "Eigen cannot reuse factors"},
        Case{"reuse with two-level",
             [](Settings& settings) {
                 settings.reuse_factors = true;
                 settings.pardiso.level = PardisoLevel::TWO_LEVEL;
             },
             steady(), "reuse_factors = true needs pardiso.level = ONE_LEVEL"},
        Case{"minimum degree with two-level",
             [](Settings& settings) {
                 settings.pardiso.ordering = PardisoOrdering::MIN_DEGREE;
                 settings.pardiso.level = PardisoLevel::TWO_LEVEL;
             },
             steady(), "MIN_DEGREE needs pardiso.level = ONE_LEVEL"},
        Case{"Cholesky with radiation",
             [](Settings& settings) {
                 settings.factorization = Factorization::CHOLESKY;
             },
             steady_with_radiation(), "without radiative couplings"},
        Case{"Cholesky for a transient",
             [](Settings& settings) {
                 settings.factorization = Factorization::CHOLESKY;
             },
             crank_nicolson(), "not available for CRANK_NICOLSON"},
        Case{"no time step",
             [](Settings& settings) { settings.time_step.reset(); },
             crank_nicolson(), "CRANK_NICOLSON needs time_step"},
        Case{"end before start",
             [](Settings& settings) {
                 settings.start_time = 10.0;
                 settings.end_time = 5.0;
             },
             crank_nicolson(), "end_time is before start_time"},
        Case{"value out of range",
             [](Settings& settings) { settings.tolerance = -1.0; }, steady(),
             "tolerance must be positive"});
    INFO(test_case.name);
    Settings settings = with_time_step();
    test_case.change(settings);
    REQUIRE(contains(contradiction_of(settings, test_case.context),
                     test_case.message));
}

TEST_CASE("Every contradiction is listed at once", "[solver][settings]") {
    Settings settings;
    settings.max_iterations = 0;
    settings.factorization = Factorization::CHOLESKY;
    const std::string message =
        contradiction_of(settings, context(SolverType::CRANK_NICOLSON));
    REQUIRE(contains(message, "max_iterations must be at least 1"));
    REQUIRE(contains(message, "needs time_step"));
    REQUIRE(contains(message, "CHOLESKY is not available"));
}

TEST_CASE("Set fields that do not apply are ignored with the reason",
          "[solver][settings]") {
    struct Case {
        std::string_view field;
        std::function<void(Settings&)> change;
        ResolveContext context;
        std::string_view reason;
    };
    const auto test_case = GENERATE(
        Case{"time_step", [](Settings& settings) { settings.time_step = 1.0; },
             steady(), "not used by STEADY_STATE"},
        Case{"reuse_factors",
             [](Settings& settings) { settings.reuse_factors = true; },
             context(SolverType::JACOBIAN_PROPAGATION),
             "not used by JACOBIAN_PROPAGATION"},
        Case{"pardiso.level",
             [](Settings& settings) {
                 settings.pardiso.level = PardisoLevel::ONE_LEVEL;
             },
             steady_without_pardiso(), "only with backend = PARDISO"},
        Case{"threads", [](Settings& settings) { settings.threads = 2; },
             steady_without_pardiso(), "only with backend = PARDISO"},
        Case{"eigen.ordering",
             [](Settings& settings) {
                 settings.eigen.ordering = EigenOrdering::AMD;
             },
             steady(), "only with backend = EIGEN"},
        Case{"eigen.ordering",
             [](Settings& settings) {
                 settings.backend = Backend::EIGEN;
                 settings.factorization = Factorization::CHOLESKY;
                 settings.eigen.ordering = EigenOrdering::AMD;
             },
             steady(), "an LU factorization"});
    INFO(test_case.field);
    Settings settings = with_time_step();
    test_case.change(settings);
    const ResolvedSettings resolved =
        solvers::resolve(settings, test_case.context);
    const bool listed = std::ranges::any_of(
        resolved.not_applied, [&test_case](const NotApplied& ignored) {
            return ignored.name == test_case.field &&
                   contains(ignored.reason, test_case.reason);
        });
    REQUIRE(listed);
    REQUIRE_FALSE(resolved.applies(test_case.field));
    REQUIRE(resolved.origin(test_case.field) == Origin::DEFAULT);
}

TEST_CASE("Settings are read and written by name", "[solver][settings]") {
    Settings settings;
    settings.set("pardiso.level", SettingValue{std::string("ONE_LEVEL")});
    settings.set("max_iterations", SettingValue{std::int64_t{12}});
    settings.set("solver_type", SettingValue{std::string("CRANK_NICOLSON")});
    REQUIRE(settings.pardiso.level == PardisoLevel::ONE_LEVEL);
    REQUIRE(settings.max_iterations == 12);
    REQUIRE(settings.solver_type == SolverType::CRANK_NICOLSON);
    REQUIRE(settings.get("pardiso.level") ==
            SettingValue{std::string("ONE_LEVEL")});
    REQUIRE_FALSE(settings.get("tolerance").has_value());

    settings.unset("pardiso.level");
    REQUIRE_FALSE(settings.is_set("pardiso.level"));
}

TEST_CASE("Bad names and values are refused and change nothing",
          "[solver][settings]") {
    Settings settings;
    settings.tolerance = 1e-3;
    const Settings before = settings;
    REQUIRE_THROWS_AS(settings.set("relaxation", SettingValue{1.0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(settings.set("tolerance", SettingValue{true}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(settings.set("tolerance", SettingValue{-1.0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        settings.set("backend", SettingValue{std::string("LAPACK")}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(settings.set("pardiso.iparm",
                                   SettingValue{std::map<int, int>{{64, 1}}}),
                      std::invalid_argument);
    REQUIRE(settings == before);
}

TEST_CASE("Settings round-trip through a dictionary", "[solver][settings]") {
    const Settings settings = fully_set();
    const solvers::SettingsDict dict = settings.to_dict();
    // solver_type plus every field.
    REQUIRE(dict.size() == solvers::setting_fields().size() + 1);
    REQUIRE(Settings::from_dict(dict) == settings);
    REQUIRE(Settings{}.to_dict().empty());

    const solvers::SettingsDict unknown{{"relaxation", SettingValue{0.5}}};
    REQUIRE_THROWS_AS(Settings::from_dict(unknown), std::invalid_argument);
}

TEST_CASE("Printing without a solver type shows the set fields",
          "[solver][settings]") {
    REQUIRE(contains(Settings{}.to_string(), "nothing set"));
    Settings settings;
    settings.tolerance = 0.5;
    settings.output_attributes =
        std::set{DataModelAttribute::T, DataModelAttribute::QI};
    const std::string text = settings.to_string();
    REQUIRE(contains(text, "tolerance"));
    REQUIRE(contains(text, "0.5"));
    REQUIRE(contains(text, "{T, QI}"));
    REQUIRE_FALSE(contains(text, "max_iterations"));
}

TEST_CASE("Printing with a solver type shows every field that applies",
          "[solver][settings]") {
    Settings settings = with_time_step();
    settings.solver_type = SolverType::JACOBIAN_PROPAGATION;
    settings.reuse_factors = true;
    const std::string text = settings.to_string();
    REQUIRE(contains(text, "Settings for JACOBIAN_PROPAGATION"));
    REQUIRE(contains(text, "max_iterations     5  (default)"));
    REQUIRE(contains(text, "Set but not applied to JACOBIAN_PROPAGATION"));
    REQUIRE(contains(text, "reuse_factors      true  not used by"));

    settings.time_step.reset();
    REQUIRE(contains(settings.to_string(), "Error: "));
}

TEST_CASE("Live settings may change while a solver is initialized",
          "[solver][settings]") {
    Settings before;
    before.reuse_factors = true;
    Settings after = before;
    after.tolerance = 1e-5;
    after.threads = 3;
    after.pardiso.verbose = true;
    REQUIRE(solvers::changed_init_fields(before, after, steady()).empty());
    REQUIRE_NOTHROW(solvers::check_settings_change(before, after, steady(),
                                                   /*initialized=*/true));
}

TEST_CASE("Time and output settings are read by initialize()",
          "[solver][settings]") {
    for (const char* name :
         {"start_time", "end_time", "time_step", "output_interval",
          "output_attributes", "output_name", "capacity_delta"}) {
        INFO(name);
        REQUIRE(solvers::setting_field(name).mutability == Mutability::INIT);
    }
    REQUIRE(solvers::setting_field("tolerance").mutability == Mutability::LIVE);
}

TEST_CASE("Init settings are frozen while a solver is initialized",
          "[solver][settings]") {
    const Settings before;
    Settings after;
    after.backend = Backend::EIGEN;
    after.pardiso.level = PardisoLevel::ONE_LEVEL;  // not applied with Eigen
    REQUIRE(solvers::changed_init_fields(before, after, steady()) ==
            std::vector<std::string_view>{"backend"});
    REQUIRE_THROWS_AS(solvers::check_settings_change(before, after, steady(),
                                                     /*initialized=*/true),
                      std::runtime_error);
    REQUIRE_NOTHROW(solvers::check_settings_change(before, after, steady(),
                                                   /*initialized=*/false));

    // Setting a field to its default value is not a change.
    Settings same;
    same.backend = Backend::PARDISO;
    REQUIRE(solvers::changed_init_fields(before, same, steady()).empty());
}

TEST_CASE("The context reads radiation from the model", "[solver][settings]") {
    const auto radiative =
        pycanha::test_models::make_plate(2, /*radiation=*/true);
    const auto conductive =
        pycanha::test_models::make_plate(2, /*radiation=*/false);
    REQUIRE(solvers::resolve_context(SolverType::STEADY_STATE, *radiative)
                .has_radiation.value_or(false));
    REQUIRE_FALSE(
        solvers::resolve_context(SolverType::STEADY_STATE, *conductive)
            .has_radiation.value_or(true));
}

TEST_CASE("Output presets", "[solver][settings]") {
    const auto dense = solvers::all_dense_output_attributes();
    const auto all = solvers::all_output_attributes();
    REQUIRE(dense.size() == 13);
    REQUIRE(all.size() == 16);
    REQUIRE(all.contains(DataModelAttribute::JAC));
    REQUIRE_FALSE(dense.contains(DataModelAttribute::KL));
}
