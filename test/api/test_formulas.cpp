#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"

namespace {

void populate_formula_demo(pycanha::ThermalModel& tm) {
    pycanha::Node plate(1);
    plate.set_T(300.0);
    plate.set_C(100.0);

    pycanha::Node sink(2);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(7.0);

    tm.tmm().add_node(plate);
    tm.tmm().add_node(sink);
    tm.tmm().add_conductive_coupling(1, 2, 5.0);

    tm.parameters().add_parameter("gain", 3.0);
    tm.parameters().add_parameter("offset", 2.0);
}

}  // namespace

TEST_CASE("Smart formulas read like a high-level API lesson",
          "[api][formulas]") {
    pycanha::ThermalModel tm("formulas_demo");
    populate_formula_demo(tm);

    auto& formulas = tm.formulas();

    SECTION("add_formula picks the right formula subtype") {
        // These overloads are meant to be the friendly surface: the library
        // chooses the concrete formula type based on what the right-hand side
        // refers to.
        auto& fixed_heat = formulas.add_formula("QI1", 20.0);
        auto& fixed_string = formulas.add_formula("QS1", "42 + 3");
        auto& parameter_formula = formulas.add_formula("C1", "gain + offset");
        auto& expression_formula = formulas.add_formula("QR1", "GL(1,2) + T2");

        REQUIRE(dynamic_cast<pycanha::ValueFormula*>(&fixed_heat) != nullptr);
        REQUIRE(dynamic_cast<pycanha::ValueFormula*>(&fixed_string) != nullptr);
        REQUIRE(dynamic_cast<pycanha::ParameterFormula*>(&parameter_formula) !=
                nullptr);
        REQUIRE(dynamic_cast<pycanha::ExpressionFormula*>(
                    &expression_formula) != nullptr);

        REQUIRE_FALSE(formulas.remove_formula("QE1"));
        REQUIRE_THROWS_AS(formulas.add_formula("QI1", "gain"),
                          std::invalid_argument);

        formulas.apply_formulas();
        REQUIRE(tm.tmm().nodes().get_qi(1) == Catch::Approx(20.0));
        REQUIRE(tm.tmm().nodes().get_qs(1) == Catch::Approx(45.0));
        REQUIRE(tm.tmm().nodes().get_C(1) == Catch::Approx(5.0));
        REQUIRE(tm.tmm().nodes().get_qr(1) == Catch::Approx(12.0));
    }

    SECTION("Compiled formulas and derivative ordering stay predictable") {
        static_cast<void>(
            formulas.add_parameter_formula("QI1", "gain + offset"));
        static_cast<void>(formulas.add_parameter_formula("GL(1,2)", "gain"));

        formulas.apply_formulas();
        REQUIRE(tm.tmm().nodes().get_qi(1) == Catch::Approx(5.0));
        REQUIRE(tm.tmm().conductive_couplings().get_coupling_value(1, 2) ==
                Catch::Approx(3.0));

        formulas.compile_formulas();
        tm.parameters().set_parameter("gain", 5.0);
        tm.parameters().set_parameter("offset", 1.0);
        formulas.apply_compiled_formulas();
        REQUIRE(tm.tmm().nodes().get_qi(1) == Catch::Approx(6.0));
        REQUIRE(tm.tmm().conductive_couplings().get_coupling_value(1, 2) ==
                Catch::Approx(5.0));

        auto& derivative_parameters = formulas.parameters_with_derivatives();
        derivative_parameters.add_parameter("gain");
        derivative_parameters.add_parameter("offset");
        REQUIRE(derivative_parameters.parameter_names().size() == 2);
        REQUIRE(derivative_parameters.parameter_names().at(0) == "gain");
        REQUIRE(derivative_parameters.parameter_names().at(1) == "offset");

        REQUIRE(derivative_parameters.remove_parameter("gain"));
        derivative_parameters.add_parameter("gain");
        REQUIRE(derivative_parameters.parameter_names().size() == 2);
        REQUIRE(derivative_parameters.parameter_names().at(0) == "offset");
        REQUIRE(derivative_parameters.parameter_names().at(1) == "gain");
    }
}