#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <utility>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/parameters/variable.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("TimeVariable updates an internal parameter", "[variables][time]") {
    Parameters parameters;
    double time = 0.5;

    TimeVariable variable(
        "load",
        LookupTable1D((Eigen::Vector2d{} << 0.0, 1.0).finished(),
                      (Eigen::Vector2d{} << 10.0, 30.0).finished()),
        parameters, std::addressof(time));

    const auto initial_value =
        std::get<double>(parameters.get_parameter("load"));
    const bool resources_created = parameters.is_internal_parameter("load");

    REQUIRE(resources_created);
    REQUIRE(variable.current_value() == Catch::Approx(20.0));
    REQUIRE(initial_value == Catch::Approx(20.0));

    const auto updated_value = [&variable, &time]() {
        time = 1.0;
        variable.update();
        return variable.current_value();
    }();
    const auto stored_updated_value =
        std::get<double>(parameters.get_parameter("load"));

    REQUIRE(updated_value == Catch::Approx(30.0));
    REQUIRE(stored_updated_value == Catch::Approx(30.0));
}

TEST_CASE("TimeVariable cleans up owned resources on destruction",
          "[variables][time]") {
    Parameters parameters;

    {
        double time = 0.5;
        const TimeVariable variable(
            "load",
            LookupTable1D((Eigen::Vector2d{} << 0.0, 1.0).finished(),
                          (Eigen::Vector2d{} << 10.0, 30.0).finished()),
            parameters, std::addressof(time));

        REQUIRE(parameters.contains("load"));
    }

    REQUIRE_FALSE(parameters.contains("load"));
}

TEST_CASE("TimeVariable supports move semantics", "[variables][time]") {
    Parameters parameters;
    double time = 0.0;

    TimeVariable variable(
        "moved",
        LookupTable1D((Eigen::Vector2d{} << 0.0, 2.0).finished(),
                      (Eigen::Vector2d{} << 4.0, 8.0).finished()),
        parameters, std::addressof(time));
    TimeVariable moved = std::move(variable);

    const auto moved_value = [&moved, &time]() {
        time = 2.0;
        moved.update();
        return moved.current_value();
    }();

    REQUIRE(moved_value == Catch::Approx(8.0));
    REQUIRE(std::get<double>(parameters.get_parameter("moved")) ==
            Catch::Approx(8.0));
}

TEST_CASE("TemperatureVariable evaluates its lookup table",
          "[variables][temperature]") {
    const TemperatureVariable variable(
        "kappa",
        LookupTable1D((Eigen::Vector3d{} << 200.0, 300.0, 400.0).finished(),
                      (Eigen::Vector3d{} << 1.0, 2.0, 5.0).finished()));

    REQUIRE(variable.name() == "kappa");
    REQUIRE(variable.evaluate(350.0) == Catch::Approx(3.5));
    REQUIRE(variable.lookup_table().size() == 3);
}

TEST_CASE(
    "ThermalMathematicalModel synchronizes time and time variables before "
    "formulas",
    "[variables][tmm][formulas]") {
    auto model = std::make_shared<ThermalMathematicalModel>("variable-model");
    model->add_node(1);
    model->nodes().set_qi(1, 0.0);

    model->add_time_variable(
        "load", (Eigen::Vector3d{} << 0.0, 5.0, 10.0).finished(),
        (Eigen::Vector3d{} << 0.0, 50.0, 100.0).finished());

    const auto formula = model->formulas.create_formula(
        Entity::qi(model->network(), 1), "load + time");
    model->formulas.add_formula(formula);
    model->formulas.validate_for_execution();
    model->formulas.compile_formulas();
    model->formulas.lock_parameters_for_execution();

    model->time = 6.0;
    model->callback_solver_loop();

    REQUIRE(std::get<double>(model->parameters.get_parameter("time")) ==
            Catch::Approx(6.0));
    REQUIRE(std::get<double>(model->parameters.get_parameter("load")) ==
            Catch::Approx(60.0));
    REQUIRE(model->nodes().get_qi(1) == Catch::Approx(66.0));

    model->formulas.unlock_parameters();
}
