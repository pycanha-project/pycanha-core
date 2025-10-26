#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/node.hpp"
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

    pycanha::AttributeEntity heat_load(*network, "QI", 1);
    pycanha::AttributeEntity temperature(*network, "T", 2);
    pycanha::ConductiveCouplingEntity conductive(*network, 1, 2);

    formulas.add_formula(
        pycanha::ParameterFormula(heat_load, *parameters, "P1"));
    formulas.add_formula(
        pycanha::ParameterFormula(temperature, *parameters, "P2"));
    formulas.add_formula(
        pycanha::ParameterFormula(conductive, *parameters, "P3"));

    formulas.apply_formulas();

    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(10.0));
    REQUIRE(network->nodes().get_T(2) == Catch::Approx(11.0));
    REQUIRE(network->conductive_couplings().get_coupling_value(1, 2) ==
            Catch::Approx(12.0));

    parameters->set_parameter("P1", 21.0);
    parameters->set_parameter("P2", 22.0);
    parameters->set_parameter("P3", 23.0);

    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(10.0));
    REQUIRE(network->nodes().get_T(2) == Catch::Approx(11.0));
    REQUIRE(network->conductive_couplings().get_coupling_value(1, 2) ==
            Catch::Approx(12.0));

    formulas.apply_formulas();

    REQUIRE(network->nodes().get_qi(1) == Catch::Approx(21.0));
    REQUIRE(network->nodes().get_T(2) == Catch::Approx(22.0));
    REQUIRE(network->conductive_couplings().get_coupling_value(1, 2) ==
            Catch::Approx(23.0));
}

TEST_CASE("Value formulas capture static snapshots", "[formulas]") {
    auto network = std::make_shared<pycanha::ThermalNetwork>();
    pycanha::Node node1(1);
    network->add_node(node1);

    network->nodes().set_T(1, 42.0);

    pycanha::AttributeEntity temperature(*network, "T", 1);
    pycanha::ValueFormula snapshot(temperature);

    snapshot.compile_formula();
    network->nodes().set_T(1, -5.0);
    snapshot.apply_compiled_formula();

    REQUIRE(network->nodes().get_T(1) == Catch::Approx(42.0));

    snapshot.set_value(77.0);
    snapshot.apply_formula();

    REQUIRE(network->nodes().get_T(1) == Catch::Approx(77.0));
}
