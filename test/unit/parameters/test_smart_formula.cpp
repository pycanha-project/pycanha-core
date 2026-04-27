#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>
#include <string>
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

    const auto parameter_expression_formula =
        formulas.create_formula(entity, "gain + offset");
    REQUIRE(dynamic_cast<ParameterFormula*>(
                parameter_expression_formula.get()) != nullptr);

    formulas.add_formula(parameter_expression_formula);

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

TEST_CASE("Formulas overloads and association guards stay explicit",
          "[formulas][factory]") {
    auto network = std::make_shared<ThermalNetwork>();
    auto parameters = std::make_shared<Parameters>();

    Node node1(1);
    Node node2(2);
    node2.set_type(BOUNDARY_NODE);
    node2.set_T(7.0);

    network->add_node(node1);
    network->add_node(node2);
    network->nodes().set_qi(1, 0.0);

    parameters->add_parameter("gain", 3.0);
    parameters->add_parameter("offset", 2.0);

    const auto heat = Entity::qi(*network, 1);

    Formulas detached;
    REQUIRE_THROWS_AS(detached.create_formula(heat, "gain"),
                      std::runtime_error);
    REQUIRE_THROWS_AS(detached.create_parameter_formula(heat, "gain"),
                      std::runtime_error);
    REQUIRE_THROWS_AS(detached.add_expression_formula(heat, "gain"),
                      std::runtime_error);

    Formulas formulas(network, parameters);

    auto& static_formula = formulas.add_formula(heat, 4.0);
    REQUIRE(dynamic_cast<ValueFormula*>(&static_formula) != nullptr);
    formulas.apply_formulas();
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(4.0));

    REQUIRE(formulas.remove_formula(heat));
    REQUIRE_FALSE(formulas.parameter_dependencies().contains("gain"));

    auto& parameter_formula = formulas.add_parameter_formula("QI1", "gain");
    auto& auto_selected_formula = formulas.add_formula("T2", "gain + offset");
    auto& expression_formula =
        formulas.add_expression_formula("QR1", "QI1 * 2.0");
    REQUIRE(dynamic_cast<ParameterFormula*>(&parameter_formula) != nullptr);
    REQUIRE(dynamic_cast<ParameterFormula*>(&auto_selected_formula) != nullptr);
    REQUIRE(dynamic_cast<ExpressionFormula*>(&expression_formula) != nullptr);

    auto self_formula = formulas.create_formula(heat, "QI1");
    REQUIRE(dynamic_cast<ExpressionFormula*>(self_formula.get()) != nullptr);

    formulas.apply_formulas();
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(3.0));
    REQUIRE(network->nodes().get_T(2) == Catch::Approx(5.0));
    REQUIRE(network->nodes().get_qr(1) == Catch::Approx(6.0));
    REQUIRE(formulas.parameter_dependencies().at("gain").size() == 2U);
    REQUIRE(formulas.parameter_dependencies().at("offset").size() == 1U);

    REQUIRE(formulas.remove_formula(Entity::t(*network, 2)));
    REQUIRE(formulas.parameter_dependencies().at("gain").size() == 1U);
    REQUIRE_FALSE(formulas.parameter_dependencies().contains("offset"));
}

TEST_CASE("Sparse formula targets materialize storage on registration",
          "[formulas][factory][sparse-targets]") {
    auto network = std::make_shared<ThermalNetwork>();
    auto parameters = std::make_shared<Parameters>();
    Formulas formulas(network, parameters);

    Node node1(1);
    network->add_node(node1);

    parameters->add_parameter("gain", 3.0);
    parameters->add_parameter("offset", 2.0);

    REQUIRE(network->nodes().qi_vector.nonZeros() == 0);
    REQUIRE(network->nodes().qs_vector.nonZeros() == 0);
    REQUIRE(network->nodes().qr_vector.nonZeros() == 0);

    auto& parameter_formula = formulas.add_parameter_formula("QI1", "gain");
    auto& auto_selected_formula = formulas.add_formula("QS1", "gain + offset");
    auto& expression_formula =
        formulas.add_expression_formula("QR1", "QI1 * 2.0");

    REQUIRE(dynamic_cast<ParameterFormula*>(&parameter_formula) != nullptr);
    REQUIRE(dynamic_cast<ParameterFormula*>(&auto_selected_formula) != nullptr);
    REQUIRE(dynamic_cast<ExpressionFormula*>(&expression_formula) != nullptr);

    REQUIRE(network->nodes().qi_vector.nonZeros() == 1);
    REQUIRE(network->nodes().qs_vector.nonZeros() == 1);
    REQUIRE(network->nodes().qr_vector.nonZeros() == 1);
    REQUIRE(network->nodes().get_qi_value_ref(1) != nullptr);
    REQUIRE(network->nodes().get_qs_value_ref(1) != nullptr);
    REQUIRE(network->nodes().get_qr_value_ref(1) != nullptr);
}
