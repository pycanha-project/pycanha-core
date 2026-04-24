#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

TEST_CASE("Parameter formulas propagate parameter values", "[formulas]") {
    auto network = std::make_shared<pycanha::ThermalNetwork>();
    auto parameters = std::make_shared<pycanha::Parameters>();

    pycanha::Formulas formulas(network, parameters);

    pycanha::Node node1(1);
    pycanha::Node node2(2);
    network->add_node(node1);
    network->add_node(node2);

    network->nodes().set_qi(1, 0.0);
    network->nodes().set_T(2, 0.0);
    network->conductive_couplings().add_coupling(1, 2, 2.0);

    parameters->add_parameter("P1", 10.0);
    parameters->add_parameter("P2", 11.0);
    parameters->add_parameter("P3", 12.0);

    const pycanha::Entity heat_load = pycanha::Entity::qi(*network, 1);
    const pycanha::Entity temperature = pycanha::Entity::t(*network, 2);
    const pycanha::Entity conductive = pycanha::Entity::gl(*network, 1, 2);

    formulas.add_formula(
        pycanha::ParameterFormula(heat_load, *parameters, "P1"));
    formulas.add_formula(
        pycanha::ParameterFormula(temperature, *parameters, "P2"));
    formulas.add_formula(
        pycanha::ParameterFormula(conductive, *parameters, "P3"));

    formulas.apply_formulas();

    // NOLINTBEGIN(bugprone-chained-comparison)
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(10.0));
    REQUIRE(network->nodes().get_T(2) == Catch::Approx(11.0));
    REQUIRE(network->conductive_couplings().get_coupling_value(1, 2) ==
            Catch::Approx(12.0));
    // NOLINTEND(bugprone-chained-comparison)

    parameters->set_parameter("P1", 21.0);
    parameters->set_parameter("P2", 22.0);
    parameters->set_parameter("P3", 23.0);

    // NOLINTBEGIN(bugprone-chained-comparison)
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(10.0));
    REQUIRE(network->nodes().get_T(2) == Catch::Approx(11.0));
    REQUIRE(network->conductive_couplings().get_coupling_value(1, 2) ==
            Catch::Approx(12.0));
    // NOLINTEND(bugprone-chained-comparison)

    formulas.apply_formulas();

    // NOLINTBEGIN(bugprone-chained-comparison)
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(21.0));
    REQUIRE(network->nodes().get_T(2) == Catch::Approx(22.0));
    REQUIRE(network->conductive_couplings().get_coupling_value(1, 2) ==
            Catch::Approx(23.0));
    // NOLINTEND(bugprone-chained-comparison)
}

TEST_CASE("Value formulas capture static snapshots", "[formulas]") {
    auto network = std::make_shared<pycanha::ThermalNetwork>();
    pycanha::Node node1(1);
    network->add_node(node1);

    network->nodes().set_T(1, 42.0);

    const pycanha::Entity temperature = pycanha::Entity::t(*network, 1);
    pycanha::ValueFormula snapshot(temperature);

    snapshot.compile_formula();
    network->nodes().set_T(1, -5.0);
    snapshot.apply_compiled_formula();

    // NOLINTBEGIN(bugprone-chained-comparison)
    REQUIRE(network->nodes().get_T(1) == Catch::Approx(42.0));

    snapshot.set_value(77.0);
    snapshot.apply_formula();

    REQUIRE(network->nodes().get_T(1) == Catch::Approx(77.0));
    // NOLINTEND(bugprone-chained-comparison)
}

TEST_CASE(
    "ParameterFormula keeps slot binding across rename in interpreted mode",
    "[formulas]") {
    auto network = std::make_shared<pycanha::ThermalNetwork>();
    pycanha::Node node1(1);
    network->add_node(node1);
    network->nodes().set_qi(1, 0.0);

    pycanha::Parameters parameters;
    parameters.add_parameter("P1", 10.0);

    const pycanha::Entity heat_load = pycanha::Entity::qi(*network, 1);
    pycanha::ParameterFormula formula(heat_load, parameters, "P1");

    parameters.rename_parameter("P1", "P1_renamed");
    parameters.set_parameter("P1_renamed", 14.0);

    formula.apply_formula();

    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(14.0));
}

TEST_CASE("Formulas validate, compile and lock parameters explicitly",
          "[formulas]") {
    auto network = std::make_shared<pycanha::ThermalNetwork>();
    auto parameters = std::make_shared<pycanha::Parameters>();
    pycanha::Formulas formulas(network, parameters);

    pycanha::Node node1(1);
    network->add_node(node1);
    network->nodes().set_qi(1, 0.0);

    parameters->add_parameter("P1", 10.0);
    const pycanha::Entity heat_load = pycanha::Entity::qi(*network, 1);
    formulas.add_formula(
        pycanha::ParameterFormula(heat_load, *parameters, "P1"));

    REQUIRE_FALSE(formulas.is_validation_current());

    formulas.validate_for_execution();
    REQUIRE(formulas.is_validation_current());

    formulas.compile_formulas();
    formulas.lock_parameters_for_execution();
    REQUIRE(parameters->is_structure_locked());

    parameters->set_parameter("P1", 18.0);
    formulas.apply_compiled_formulas();
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(18.0));

    parameters->add_parameter("P2", 2.0);
    REQUIRE_FALSE(parameters->contains("P2"));

    formulas.unlock_parameters();
    REQUIRE_FALSE(parameters->is_structure_locked());
}

TEST_CASE("Formulas require fresh validation before locking", "[formulas]") {
    auto network = std::make_shared<pycanha::ThermalNetwork>();
    auto parameters = std::make_shared<pycanha::Parameters>();
    pycanha::Formulas formulas(network, parameters);

    pycanha::Node node1(1);
    network->add_node(node1);

    parameters->add_parameter("P1", 10.0);
    const pycanha::Entity heat_load = pycanha::Entity::qi(*network, 1);
    formulas.add_formula(
        pycanha::ParameterFormula(heat_load, *parameters, "P1"));

    formulas.validate_for_execution();
    parameters->add_parameter("P2", 5.0);

    REQUIRE_FALSE(formulas.is_validation_current());
    REQUIRE_THROWS_AS(formulas.lock_parameters_for_execution(),
                      std::runtime_error);
}

TEST_CASE("ThermalMathematicalModel uses compiled formulas while locked",
          "[formulas]") {
    auto model = std::make_shared<pycanha::ThermalMathematicalModel>("model");
    model->add_node(1);
    model->nodes().set_qi(1, 0.0);

    model->parameters().add_parameter("P1", 10.0);
    model->formulas().add_formula(pycanha::ParameterFormula(
        model->entity("QI1"), model->parameters(), "P1"));

    model->formulas().validate_for_execution();
    model->formulas().compile_formulas();
    model->formulas().lock_parameters_for_execution();

    model->parameters().set_parameter("P1", 15.0);
    model->callback_solver_loop();

    REQUIRE(model->nodes().get_qi(1) == Catch::Approx(15.0));

    model->formulas().unlock_parameters();
}
