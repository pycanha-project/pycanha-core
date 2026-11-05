#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("Formulas compute the kind of each formula from its inputs",
          "[formulas][factory]") {
    auto network = std::make_shared<ThermalNetwork>();
    auto parameters = std::make_shared<Parameters>();
    Formulas formulas(network, parameters);

    Node node1(1);
    network->add_node(node1);
    network->nodes().set_qi(1, 0.0);

    parameters->add_parameter("gain", 3.0);
    parameters->add_parameter("offset", 2.0);

    const auto entity = Entity::qi(*network, 1);

    REQUIRE(formulas.create_formula(entity, "42.0")->kind() ==
            FormulaKind::Constant);
    REQUIRE(formulas.create_formula(entity, "6 * 7")->get_value() ==
            Catch::Approx(42.0));
    REQUIRE(formulas.create_formula(entity, "gain")->kind() ==
            FormulaKind::Parameters);
    REQUIRE(formulas.create_formula(entity, "gain * T1")->kind() ==
            FormulaKind::Entities);

    const auto parameter_expression_formula =
        formulas.create_formula(entity, "gain + offset");
    REQUIRE(parameter_expression_formula->kind() == FormulaKind::Parameters);

    formulas.add_formula(parameter_expression_formula);

    formulas.apply_formulas();
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(5.0));
}

TEST_CASE("Constant formulas keep their value and expression",
          "[formulas][factory]") {
    auto network = std::make_shared<ThermalNetwork>();
    auto parameters = std::make_shared<Parameters>();
    Formulas formulas(network, parameters);
    Node node1(1);
    network->add_node(node1);

    const auto& number = formulas.add_formula("QI1", 0.25);
    const auto& text = formulas.add_formula("QS1", "2 * 0.5");
    REQUIRE(number.kind() == FormulaKind::Constant);
    REQUIRE(dynamic_cast<const ExpressionFormula&>(number).expression() ==
            "0.25");
    REQUIRE(dynamic_cast<const ExpressionFormula&>(text).expression() ==
            "2 * 0.5");
    REQUIRE(number.parameter_dependencies().empty());
    REQUIRE(number.entity_dependencies().empty());

    formulas.compile_formulas();
    formulas.apply_compiled_formulas();
    REQUIRE(network->nodes().get_qi(1) == 0.25);
    REQUIRE(network->nodes().get_qs(1) == 1.0);
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
    REQUIRE_THROWS_AS(detached.add_formula("QI1", "gain"), std::runtime_error);

    Formulas formulas(network, parameters);

    const auto& static_formula = formulas.add_formula(heat, 4.0);
    REQUIRE(static_formula.kind() == FormulaKind::Constant);
    formulas.apply_formulas();
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(4.0));

    REQUIRE(formulas.remove_formula(heat));
    REQUIRE_FALSE(formulas.dependencies().parameters.contains("gain"));

    const auto& parameter_formula = formulas.add_formula("QI1", "gain");
    const auto& two_parameter_formula =
        formulas.add_formula("T2", "gain + offset");
    const auto& entity_formula = formulas.add_formula("QR1", "QI1 * 2.0");
    REQUIRE(parameter_formula.kind() == FormulaKind::Parameters);
    REQUIRE(two_parameter_formula.kind() == FormulaKind::Parameters);
    REQUIRE(entity_formula.kind() == FormulaKind::Entities);

    auto self_formula = formulas.create_formula(heat, "QI1");
    REQUIRE(self_formula->kind() == FormulaKind::Entities);

    formulas.apply_formulas();
    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(3.0));
    REQUIRE(network->nodes().get_T(2) == Catch::Approx(5.0));
    REQUIRE(network->nodes().get_qr(1) == Catch::Approx(6.0));
    REQUIRE(formulas.dependencies().parameters.at("gain").size() == 2U);
    REQUIRE(formulas.dependencies().parameters.at("offset").size() == 1U);

    REQUIRE(formulas.remove_formula(Entity::t(*network, 2)));
    REQUIRE(formulas.dependencies().parameters.at("gain").size() == 1U);
    REQUIRE_FALSE(formulas.dependencies().parameters.contains("offset"));
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

    REQUIRE(network->nodes().qi_vector.empty());
    REQUIRE(network->nodes().qs_vector.empty());
    REQUIRE(network->nodes().qr_vector.empty());

    static_cast<void>(formulas.add_formula("QI1", "gain"));
    static_cast<void>(formulas.add_formula("QS1", "gain + offset"));
    static_cast<void>(formulas.add_formula("QR1", "QI1 * 2.0"));

    REQUIRE(network->nodes().qi_vector.size() == 1U);
    REQUIRE(network->nodes().qs_vector.size() == 1U);
    REQUIRE(network->nodes().qr_vector.size() == 1U);
    REQUIRE(network->nodes().get_qi_value_ref(1) != nullptr);
    REQUIRE(network->nodes().get_qs_value_ref(1) != nullptr);
    REQUIRE(network->nodes().get_qr_value_ref(1) != nullptr);
}

namespace {

// Three nodes, QI on each set by a formula reading one parameter.
struct IndexedModel {
    std::shared_ptr<ThermalNetwork> network =
        std::make_shared<ThermalNetwork>();
    std::shared_ptr<Parameters> parameters = std::make_shared<Parameters>();
    Formulas formulas{network, parameters};

    IndexedModel() {
        for (int number = 1; number <= 3; ++number) {
            Node node(number);
            network->add_node(node);
        }
        parameters->add_parameter("gain", 2.0);
        for (int number = 1; number <= 3; ++number) {
            static_cast<void>(
                formulas.add_formula("QI" + std::to_string(number),
                                     std::to_string(number) + " * gain"));
        }
    }

    [[nodiscard]] std::vector<std::string> targets() const {
        std::vector<std::string> names;
        std::ranges::transform(
            formulas.formulas(), std::back_inserter(names),
            [](const auto& formula) {
                return formula->entity().string_representation();
            });
        return names;
    }
};

}  // namespace

TEST_CASE("Formulas refuse a second formula for the same target",
          "[formulas][index]") {
    IndexedModel model;
    REQUIRE_THROWS_AS(model.formulas.add_formula("QI2", "gain"),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        model.formulas.add_formula(Entity::qi(*model.network, 3), 1.0),
        std::invalid_argument);
    REQUIRE(model.targets() == std::vector<std::string>{"QI1", "QI2", "QI3"});
}

TEST_CASE("Removed formulas leave the insertion order of the others",
          "[formulas][index]") {
    IndexedModel model;
    REQUIRE(model.formulas.remove_formula("QI2"));
    REQUIRE_FALSE(model.formulas.remove_formula("QI2"));
    REQUIRE(model.targets() == std::vector<std::string>{"QI1", "QI3"});

    // A target added back goes to the end, and the index still finds the
    // formulas the compaction moved.
    static_cast<void>(model.formulas.add_formula("QI2", "5 * gain"));
    REQUIRE(model.formulas.remove_formula("QI3"));
    REQUIRE(model.targets() == std::vector<std::string>{"QI1", "QI2"});
    REQUIRE_THROWS_AS(model.formulas.add_formula("QI1", "gain"),
                      std::invalid_argument);
}

TEST_CASE("Formulas apply correctly around removal holes",
          "[formulas][index]") {
    IndexedModel model;
    REQUIRE(model.formulas.remove_formula("QI1"));
    REQUIRE(model.formulas.remove_formula("QI3"));
    static_cast<void>(model.formulas.add_formula("QI3", "7 * gain"));

    model.formulas.validate_for_execution();
    model.formulas.compile_formulas();
    model.formulas.lock_parameters_for_execution();
    model.parameters->set_parameter("gain", 3.0);
    model.formulas.apply_compiled_formulas();
    model.formulas.unlock_parameters();

    REQUIRE(model.network->nodes().get_qi(1) == 0.0);
    REQUIRE(model.network->nodes().get_qi(2) == Catch::Approx(6.0));
    REQUIRE(model.network->nodes().get_qi(3) == Catch::Approx(21.0));
}

TEST_CASE("Formula inputs are split into parameters and entities",
          "[formulas][dependencies]") {
    auto network = std::make_shared<ThermalNetwork>();
    auto parameters = std::make_shared<Parameters>();
    Formulas formulas(network, parameters);
    Node node1(1);
    Node node2(2);
    network->add_node(node1);
    network->add_node(node2);
    network->conductive_couplings().add_coupling(1, 2, 4.0);
    parameters->add_parameter("gain", 3.0);

    // GL(2,1) is GL(1,2) spelled the other way round.
    const auto& formula =
        formulas.add_formula("QI1", "gain * T2 + GL(1,2) - GL(2,1) / 2");
    REQUIRE(formula.parameter_dependencies() ==
            Formula::DependencyList{"gain"});
    const auto coupling = Entity::gl(*network, 1, 2);
    const auto temperature = Entity::t(*network, 2);
    REQUIRE(formula.entity_dependencies() ==
            std::vector<Entity>{coupling, coupling, temperature});
    REQUIRE(formulas.dependencies().entities.at(coupling).size() == 1U);
}

TEST_CASE("Formulas dependencies are built on request",
          "[formulas][dependencies]") {
    IndexedModel model;
    static_cast<void>(model.formulas.add_formula("QR1", "QI1 + QI1 * gain"));

    const auto dependencies = model.formulas.dependencies();
    REQUIRE(dependencies.parameters.at("gain").size() == 4U);
    REQUIRE(dependencies.entities.size() == 1U);
    const auto& readers =
        dependencies.entities.at(Entity::qi(*model.network, 1));
    REQUIRE(readers.size() == 1U);
    REQUIRE(readers.front()->entity().string_representation() == "QR1");
}

TEST_CASE("Formula derivatives are built on first use",
          "[formulas][derivatives]") {
    auto network = std::make_shared<ThermalNetwork>();
    Node node1(1);
    network->add_node(node1);
    Parameters parameters;
    parameters.add_parameter("p1", 2.0);
    parameters.add_parameter("p2", 5.0);

    ExpressionFormula formula(Entity::qi(*network, 1), parameters,
                              "p1**2 * p2");
    REQUIRE(formula.get_derivative_values()->empty());

    formula.compile_formula();
    formula.calculate_derivatives();
    REQUIRE(*formula.get_derivative_values() == std::vector<double>{20.0, 4.0});

    // A compile drops the compiled derivatives; the next call rebuilds them
    // for the new parameter values.
    parameters.set_parameter("p1", 3.0);
    formula.compile_formula();
    formula.calculate_derivatives();
    REQUIRE(*formula.get_derivative_values() == std::vector<double>{30.0, 9.0});
}
