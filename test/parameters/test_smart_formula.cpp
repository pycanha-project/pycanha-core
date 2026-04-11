#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>
#include <unordered_set>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("Formulas create smart formula types", "[formulas][factory]") {
    auto network = std::make_shared<ThermalNetwork>();
    auto parameters = std::make_shared<Parameters>();
    Formulas formulas(network, parameters);

    Node node1(1);
    network->add_node(node1);
    network->nodes().set_qi(1, 0.0);

    parameters->add_parameter("gain", 3.0);
    parameters->add_parameter("offset", 2.0);

    const auto entity = Entity::qi(*network, 1);

    const auto value_formula = formulas.create_formula(entity, "42.0");
    REQUIRE(dynamic_cast<ValueFormula*>(value_formula.get()) != nullptr);

    const auto parameter_formula = formulas.create_formula(entity, "gain");
    REQUIRE(dynamic_cast<ParameterFormula*>(parameter_formula.get()) !=
            nullptr);

    const auto expression_formula =
        formulas.create_formula(entity, "gain + offset");
    REQUIRE(dynamic_cast<ExpressionFormula*>(expression_formula.get()) !=
            nullptr);

    formulas.add_formula(value_formula);
    formulas.add_formula(parameter_formula);
    formulas.add_formula(expression_formula);

    formulas.apply_formulas();
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(5.0));
}

TEST_CASE("ExpressionFormula resolves entity symbols on the right-hand side",
          "[formulas][factory][entity]") {
    auto network = std::make_shared<ThermalNetwork>();
    auto parameters = std::make_shared<Parameters>();
    Formulas formulas(network, parameters);

    Node node1(1);
    Node node2(2);
    network->add_node(node1);
    network->add_node(node2);
    network->nodes().set_qi(1, 0.0);
    network->nodes().set_T(2, 7.0);
    network->conductive_couplings().add_coupling(1, 2, 11.0);

    auto formula =
        formulas.create_formula(Entity::qi(*network, 1), "GL(2, 1) + T2");
    auto* expression = dynamic_cast<ExpressionFormula*>(formula.get());

    REQUIRE(expression != nullptr);
    expression->apply_formula();
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(18.0));

    expression->compile_formula();
    network->nodes().set_T(2, 9.0);
    expression->apply_compiled_formula();
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(20.0));
    REQUIRE_THROWS_AS(expression->calculate_derivatives(), std::runtime_error);
}

TEST_CASE("Smart formula factory rejects temperature variables for now",
          "[formulas][factory][temperature-variable]") {
    auto network = std::make_shared<ThermalNetwork>();
    auto parameters = std::make_shared<Parameters>();
    Formulas formulas(network, parameters);

    const std::unordered_set<std::string> temperature_variable_names{"kappa"};
    formulas.set_temperature_variable_names(&temperature_variable_names);

    Node node1(1);
    network->add_node(node1);
    network->nodes().set_qi(1, 0.0);

    REQUIRE_THROWS_AS(formulas.create_formula(Entity::qi(*network, 1), "kappa"),
                      std::invalid_argument);
}