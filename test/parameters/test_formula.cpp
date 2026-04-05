#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/tscnrlds_jacobian.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace {

using DerivativeLookup = std::unordered_map<std::string, double>;

DerivativeLookup derivatives_by_dependency(pycanha::Formula& formula) {
    const auto& dependencies = formula.parameter_dependencies();
    const auto* derivative_values = formula.get_derivative_values();

    REQUIRE(derivative_values != nullptr);
    REQUIRE(dependencies.size() == derivative_values->size());

    DerivativeLookup lookup;
    for (std::size_t index = 0U; index < dependencies.size(); ++index) {
        lookup.emplace(dependencies[index], derivative_values->at(index));
    }
    return lookup;
}

std::shared_ptr<pycanha::ThermalNetwork> make_single_node_network() {
    auto network = std::make_shared<pycanha::ThermalNetwork>();
    pycanha::Node node(1);
    network->add_node(node);
    network->nodes().set_qi(1, 0.0);
    return network;
}

std::shared_ptr<pycanha::ThermalMathematicalModel> make_jacobian_model() {
    auto model =
        std::make_shared<pycanha::ThermalMathematicalModel>("expr_jacobian");

    pycanha::Node diffusive_node(1);
    pycanha::Node boundary_node(2);

    diffusive_node.set_T(0.0);
    diffusive_node.set_C(1.0);
    diffusive_node.set_qi(1.0);

    boundary_node.set_type(pycanha::BOUNDARY_NODE);
    boundary_node.set_T(1.0);

    model->add_node(diffusive_node);
    model->add_node(boundary_node);
    model->add_conductive_coupling(1, 2, 1.0);

    model->parameters.add_parameter("k", 1.0);
    model->parameters.add_parameter("C", 1.0);

    pycanha::ConductiveCouplingEntity conductive(model->network(), 1, 2);
    pycanha::AttributeEntity capacity(model->network(), "C", 1);

    model->formulas.add_formula(std::make_shared<pycanha::ExpressionFormula>(
        conductive, model->parameters, "k"));
    model->formulas.add_formula(std::make_shared<pycanha::ExpressionFormula>(
        capacity, model->parameters, "C"));
    model->formulas.apply_formulas();
    model->formulas.calculate_derivatives();
    return model;
}

}  // namespace

TEST_CASE("ExpressionFormula parses scalar expressions",
          "[formulas][expression]") {
    auto network = make_single_node_network();
    pycanha::Parameters parameters;
    parameters.add_parameter("p1", 12.5);

    pycanha::AttributeEntity heat_load(*network, "QI", 1);
    pycanha::ExpressionFormula formula(heat_load, parameters, "p1");

    REQUIRE(formula.expression() == "p1");
    REQUIRE(formula.parameter_dependencies().size() == 1);
    REQUIRE(formula.parameter_dependencies().at(0) == "p1");

    formula.apply_formula();

    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(12.5));
    REQUIRE(formula.get_value() == Catch::Approx(12.5));
}

TEST_CASE(
    "ExpressionFormula evaluates compound expressions in interpreted and "
    "compiled mode",
    "[formulas][expression]") {
    auto network = make_single_node_network();
    pycanha::Parameters parameters;
    parameters.add_parameter("p1", 6.0);
    parameters.add_parameter("p2", 4.0);

    pycanha::AttributeEntity heat_load(*network, "QI", 1);
    pycanha::ExpressionFormula formula(heat_load, parameters,
                                       "p1*p2/123.0+0.01");

    formula.apply_formula();
    REQUIRE(network->nodes().get_qi(1) ==
            Catch::Approx(6.0 * 4.0 / 123.0 + 0.01));

    formula.compile_formula();
    parameters.set_parameter("p1", 8.0);
    parameters.set_parameter("p2", 3.0);
    formula.apply_compiled_formula();

    REQUIRE(network->nodes().get_qi(1) ==
            Catch::Approx(8.0 * 3.0 / 123.0 + 0.01));
}

TEST_CASE("ExpressionFormula calculates analytical derivatives",
          "[formulas][expression]") {
    auto network = make_single_node_network();
    pycanha::Parameters parameters;
    parameters.add_parameter("p1", 2.0);
    parameters.add_parameter("p2", 5.0);

    pycanha::AttributeEntity heat_load(*network, "QI", 1);
    pycanha::ExpressionFormula formula(heat_load, parameters, "p1*p2");

    formula.compile_formula();
    formula.calculate_derivatives();

    const auto lookup = derivatives_by_dependency(formula);
    REQUIRE(lookup.at("p1") == Catch::Approx(5.0));
    REQUIRE(lookup.at("p2") == Catch::Approx(2.0));
}

TEST_CASE("ExpressionFormula accepts Python-style power syntax",
          "[formulas][expression]") {
    auto network = make_single_node_network();
    pycanha::Parameters parameters;
    parameters.add_parameter("p1", 3.0);

    pycanha::AttributeEntity heat_load(*network, "QI", 1);
    pycanha::ExpressionFormula formula(heat_load, parameters, "p1**2");

    formula.apply_formula();
    formula.calculate_derivatives();
    const auto lookup = derivatives_by_dependency(formula);

    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(9.0));
    REQUIRE(lookup.at("p1") == Catch::Approx(6.0));
}

TEST_CASE("ExpressionFormula supports common math functions",
          "[formulas][expression]") {
    auto network = make_single_node_network();
    pycanha::Parameters parameters;
    parameters.add_parameter("p1", 0.5);
    parameters.add_parameter("p2", 1.25);

    pycanha::AttributeEntity heat_load(*network, "QI", 1);
    pycanha::ExpressionFormula formula(heat_load, parameters,
                                       "sin(p1) + exp(p2)");

    formula.apply_formula();

    REQUIRE(network->nodes().get_qi(1) ==
            Catch::Approx(std::sin(0.5) + std::exp(1.25)));
}

TEST_CASE("ExpressionFormula rejects matrix and array syntax for now",
          "[formulas][expression]") {
    auto network = make_single_node_network();
    pycanha::Parameters parameters;
    pycanha::Parameters::MatrixRXd matrix(2, 2);
    matrix << 1.0, 2.0, 3.0, 4.0;

    parameters.add_parameter("mat", matrix);
    parameters.add_parameter("scale", 2.0);

    pycanha::AttributeEntity heat_load(*network, "QI", 1);

    REQUIRE_THROWS_AS(
        pycanha::ExpressionFormula(heat_load, parameters, "mat[0,1] * scale"),
        std::invalid_argument);
}

TEST_CASE(
    "ExpressionFormula rejects invalid expressions and missing parameters",
    "[formulas][expression]") {
    auto network = make_single_node_network();
    pycanha::Parameters parameters;
    parameters.add_parameter("p1", 1.0);

    pycanha::AttributeEntity heat_load(*network, "QI", 1);

    REQUIRE_THROWS_AS(pycanha::ExpressionFormula(heat_load, parameters, "sin("),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        pycanha::ExpressionFormula(heat_load, parameters, "missing + p1"),
        std::invalid_argument);
}

TEST_CASE(
    "Formulas collection applies expression formulas and propagates "
    "derivatives",
    "[formulas][expression]") {
    auto network = std::make_shared<pycanha::ThermalNetwork>();
    auto parameters = std::make_shared<pycanha::Parameters>();
    pycanha::Formulas formulas(network, parameters);

    pycanha::Node node1(1);
    pycanha::Node node2(2);
    network->add_node(node1);
    network->add_node(node2);

    network->nodes().set_qi(1, 0.0);
    network->nodes().set_T(2, 0.0);

    parameters->add_parameter("load", 10.0);
    parameters->add_parameter("offset", 2.0);
    parameters->add_parameter("temp", 3.0);

    pycanha::AttributeEntity heat_load(*network, "QI", 1);
    pycanha::AttributeEntity temperature(*network, "T", 2);

    auto load_formula = std::make_shared<pycanha::ExpressionFormula>(
        heat_load, *parameters, "load + offset");
    auto temperature_formula = std::make_shared<pycanha::ExpressionFormula>(
        temperature, *parameters, "temp");

    formulas.add_formula(load_formula);
    formulas.add_formula(temperature_formula);

    formulas.apply_formulas();
    formulas.calculate_derivatives();

    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(12.0));
    REQUIRE(network->nodes().get_T(2) == Catch::Approx(3.0));
    REQUIRE(formulas.parameter_dependencies().at("load").size() == 1);
    REQUIRE(formulas.parameter_dependencies().at("offset").size() == 1);
    REQUIRE(formulas.parameter_dependencies().at("temp").size() == 1);

    const auto load_derivatives = derivatives_by_dependency(*load_formula);
    REQUIRE(load_derivatives.at("load") == Catch::Approx(1.0));
    REQUIRE(load_derivatives.at("offset") == Catch::Approx(1.0));
}

TEST_CASE("ExpressionFormula integrates with TSCNRLDS_JACOBIAN",
          "[formulas][expression][solver]") {
    auto model = make_jacobian_model();

    pycanha::TSCNRLDS_JACOBIAN solver(model);
    solver.MAX_ITERS = 50;
    solver.abstol_temp = 1.0e-9;
    solver.set_simulation_time(0.0, 5.0, 0.01, 0.1);
    solver.initialize();
    solver.solve();

    REQUIRE(model->thermal_data.has_table("TSCNRLDS_JACOBIAN_OUTPUT"));
    REQUIRE(solver.parameter_names().size() == 2);
    REQUIRE(solver.parameter_names().at(0) == "k");
    REQUIRE(solver.parameter_names().at(1) == "C");

    const auto& jacobian_output =
        model->thermal_data.get_table("TSCNRLDS_JACOBIAN_OUTPUT");
    REQUIRE(jacobian_output.rows() >= 2);
    REQUIRE(jacobian_output.cols() == 3);

    const auto last_row = jacobian_output.rows() - 1;
    REQUIRE(jacobian_output(last_row, 0) == Catch::Approx(5.0).margin(1.0e-9));
    REQUIRE(jacobian_output(last_row, 1) ==
            Catch::Approx(-0.92588396).margin(5.0e-6));
    REQUIRE(jacobian_output(last_row, 2) ==
            Catch::Approx(-0.06737837).margin(5.0e-6));
}