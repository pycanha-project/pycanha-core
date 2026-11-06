#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/origin.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/parameters/table.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

// Nodes 1..4 at T = 1..4 K, a conductive coupling GL(1,2) = 0.5, and the
// parameter `a` = 0.3.
struct ProgramModel {
    std::shared_ptr<ThermalNetwork> network =
        std::make_shared<ThermalNetwork>();
    std::shared_ptr<Parameters> parameters = std::make_shared<Parameters>();
    Formulas formulas{network, parameters};

    ProgramModel() {
        for (int number = 1; number <= 4; ++number) {
            Node node(number);
            node.set_T(static_cast<double>(number));
            network->add_node(node);
        }
        network->conductive_couplings().add_coupling(1, 2, 0.5);
        parameters->add_parameter("a", 0.3);
        parameters->add_internal_parameter("time", 0.0);
    }

    [[nodiscard]] double qi(int node) const {
        return network->nodes().get_qi(node);
    }

    // Applies `expression` to QI1 through the compiled program and through
    // the formula's own evaluation, and checks both against `expected`.
    void check(const std::string& expression, double expected) {
        INFO(expression);
        const auto target = Entity::qi(*network, 1);
        static_cast<void>(formulas.remove_formula(target));
        const auto& formula = formulas.add_formula(target, expression);
        formulas.apply_formulas();
        REQUIRE(qi(1) == Catch::Approx(expected));
        REQUIRE(formula.get_value() == Catch::Approx(expected));
    }
};

// cp(T) on [0, 10] K: 100 J/(kg K) at 0 K, 200 J/(kg K) at 10 K.
LookupTable1D cp_table() {
    return {Eigen::Vector2d(0.0, 10.0), Eigen::Vector2d(100.0, 200.0)};
}

}  // namespace

TEST_CASE("Arithmetic and powers match the expression", "[formulas][program]") {
    ProgramModel model;
    model.check("T2 + T3 - T4", 1.0);
    model.check("2 * T2 * T3 / T4", 3.0);
    model.check("-T2 + 1", -1.0);
    model.check("T3**2", 9.0);
    model.check("T2**-3", 0.125);
    model.check("T2**0.5", std::numbers::sqrt2);
    model.check("1 / sqrt(T4)", 0.5);
    model.check("T2**a", std::pow(2.0, 0.3));
    model.check("T2**T3", 8.0);
    model.check("(T1 + T2) / 2", 1.5);
    model.check("GL(1,2) * T2 + GL(2,1)", 1.5);
    model.check("T2**100", std::pow(2.0, 100.0));
}

TEST_CASE("Functions match their definitions", "[formulas][program]") {
    ProgramModel model;
    const double x = 0.3;
    model.check("exp(a) + log(T2)", std::exp(x) + std::numbers::ln2);
    model.check("sin(a) + cos(a) + tan(a)",
                std::sin(x) + std::cos(x) + std::tan(x));
    model.check(
        "cot(a) + sec(a) + csc(a)",
        (1.0 / std::tan(x)) + (1.0 / std::cos(x)) + (1.0 / std::sin(x)));
    model.check("asin(a) + acos(a) + atan(a)",
                std::asin(x) + std::acos(x) + std::atan(x));
    model.check("acot(a) + asec(T2) + acsc(T2)",
                std::atan(1.0 / x) + std::acos(0.5) + std::asin(0.5));
    model.check("sinh(a) + cosh(a) + tanh(a)",
                std::sinh(x) + std::cosh(x) + std::tanh(x));
    model.check(
        "coth(a) + sech(a) + csch(a)",
        (1.0 / std::tanh(x)) + (1.0 / std::cosh(x)) + (1.0 / std::sinh(x)));
    model.check("asinh(a) + acosh(T2) + atanh(a)",
                std::asinh(x) + std::acosh(2.0) + std::atanh(x));
    model.check("abs(a - T2) + sign(a - T2) + floor(a) + ceiling(a)",
                1.7 - 1.0 + 0.0 + 1.0);
    model.check("max(T1, T3, T2) + min(T4, a)", 3.3);
    model.check("atan2(T1, T2)", std::atan2(1.0, 2.0));
    model.check("pi * T1 + E", std::numbers::pi + std::numbers::e);
}

TEST_CASE("Functions without an operation are evaluated by SymEngine",
          "[formulas][program]") {
    ProgramModel model;
    model.check("gamma(T4) + erf(a)", 6.0 + std::erf(0.3));
    model.check("T2 * gamma(T3)", 4.0);
}

TEST_CASE("Formulas call tables with any argument", "[formulas][program]") {
    ProgramModel model;
    model.parameters->add_table("cp", cp_table());
    model.check("cp(T2)", 120.0);
    model.check("0.5 * cp((T2 + T4) / 2)", 65.0);
    model.check("cp(T2 + a) * cp(T4)", 123.0 * 140.0);

    // A call with a constant argument still reads the table.
    const auto& constant = model.formulas.add_formula("QI2", "cp(5)");
    REQUIRE(constant.kind() == FormulaKind::Parameters);
    REQUIRE(constant.table_dependencies() == std::vector<std::string>{"cp"});
    REQUIRE(model.formulas.dependencies().tables.at("cp").size() == 2U);
}

TEST_CASE("Table calls are checked when the formula is added",
          "[formulas][program]") {
    ProgramModel model;
    model.parameters->add_table("cp", cp_table());
    const auto target = Entity::qi(*model.network, 1);
    const auto refuses = [&](const std::string& expression) {
        INFO(expression);
        REQUIRE_THROWS_AS(model.formulas.add_formula(target, expression),
                          std::invalid_argument);
    };
    refuses("unknown(T1)");
    refuses("cp(T1, T2)");
    refuses("cp");
    refuses("gamma(cp(T1))");
    REQUIRE_FALSE(model.formulas.contains(target));
}

TEST_CASE("Independent formulas run first without changing the results",
          "[formulas][program]") {
    ProgramModel model;
    // QI3 reads QI2 before QI2 is set, so QI2 must not run first.
    static_cast<void>(model.formulas.add_formula("QI3", "QI2 + 1"));
    static_cast<void>(model.formulas.add_formula("QI2", "5.0"));
    static_cast<void>(model.formulas.add_formula("QI4", "QI2 * 2"));
    static_cast<void>(model.formulas.add_formula("QI1", "T1 * 7"));

    model.formulas.apply_formulas();
    REQUIRE(model.qi(3) == Catch::Approx(1.0));
    REQUIRE(model.qi(2) == Catch::Approx(5.0));
    REQUIRE(model.qi(4) == Catch::Approx(10.0));
    REQUIRE(model.qi(1) == Catch::Approx(7.0));

    model.formulas.apply_formulas();
    REQUIRE(model.qi(3) == Catch::Approx(6.0));
}

TEST_CASE("A callable formula runs in order between compiled ones",
          "[formulas][program]") {
    ProgramModel model;
    static_cast<void>(model.formulas.add_formula("QI1", "T1 * 2"));
    static_cast<void>(model.formulas.add_callable_formula(
        "QI2", [&model] { return model.qi(1) + 1.0; }));
    static_cast<void>(model.formulas.add_formula("QI3", "T3"));

    model.formulas.apply_formulas();
    REQUIRE(model.qi(2) == Catch::Approx(3.0));
    REQUIRE(model.qi(3) == Catch::Approx(3.0));

    model.formulas.debug_formulas = true;
    model.network->nodes().set_T(1, 5.0);
    model.formulas.apply_formulas();
    REQUIRE(model.qi(2) == Catch::Approx(11.0));
}

TEST_CASE("Edited table values reach compiled formulas without a recompile",
          "[formulas][program]") {
    ProgramModel model;
    model.parameters->add_table("cp", cp_table());
    static_cast<void>(model.formulas.add_formula("QI1", "cp(T2)"));
    model.formulas.apply_formulas();
    REQUIRE(model.qi(1) == Catch::Approx(120.0));

    const std::vector<double> values{0.0, 10.0};
    model.parameters->set_table_values("cp", values);
    REQUIRE(model.formulas.is_compiled_current());
    model.formulas.apply_formulas();
    REQUIRE(model.qi(1) == Catch::Approx(2.0));
}

TEST_CASE("Replacing a table's data recompiles the formulas that call it",
          "[formulas][program]") {
    ProgramModel model;
    REQUIRE(model.parameters->add_generated_table(Origin::conduction_builder,
                                                  "k", cp_table()) ==
            TableGeneration::added);
    static_cast<void>(model.formulas.add_formula("QI1", "k(T3)"));
    model.formulas.apply_formulas();

    // More points move the data the compiled formula points into.
    const LookupTable1D finer(Eigen::Vector4d(0.0, 1.0, 2.0, 10.0),
                              Eigen::Vector4d(0.0, 1.0, 2.0, 10.0));
    static_cast<void>(model.parameters->add_generated_table(
        Origin::conduction_builder, "k", finer));
    REQUIRE_FALSE(model.formulas.is_compiled_current());
    model.formulas.apply_formulas();
    REQUIRE(model.qi(1) == Catch::Approx(3.0));
}

TEST_CASE("Removing a called table is allowed; compiling names the formulas",
          "[formulas][program]") {
    ProgramModel model;
    model.parameters->add_table("cp", cp_table());
    static_cast<void>(model.formulas.add_formula("QI1", "cp(T1)"));
    static_cast<void>(model.formulas.add_formula("QI2", "T2"));
    model.formulas.apply_formulas();

    REQUIRE(model.parameters->remove_table("cp"));
    REQUIRE_THROWS_WITH(model.formulas.apply_formulas(),
                        Catch::Matchers::ContainsSubstring("1 of 2 formulas") &&
                            Catch::Matchers::ContainsSubstring("cp"));
}

TEST_CASE("Out-of-range table calls are recorded with the model time",
          "[formulas][program]") {
    ProgramModel model;
    model.parameters->add_table("cp", cp_table());
    static_cast<void>(model.formulas.add_formula("QI1", "cp(T3 * 10)"));
    model.formulas.apply_formulas();
    REQUIRE(model.qi(1) == Catch::Approx(200.0));

    const auto ranges = model.parameters->table_ranges();
    REQUIRE(ranges.size() == 1U);
    REQUIRE(ranges.front().first_argument == Catch::Approx(30.0));
    REQUIRE(ranges.front().first_time == Catch::Approx(0.0));
}

TEST_CASE("Derivatives go through table slopes", "[formulas][program]") {
    ProgramModel model;
    model.parameters->add_table("cp", cp_table());
    model.parameters->add_parameter("t_ref", 4.0);
    const auto& formula =
        model.formulas.add_formula("QI1", "a * cp(2 * t_ref)");
    REQUIRE(formula.kind() == FormulaKind::Parameters);

    model.formulas.calculate_derivatives();
    const auto* derivatives =
        model.formulas.find("QI1")->get_derivative_values();
    REQUIRE(derivatives != nullptr);
    // Inputs in name order: a, t_ref. d/da = cp(8); d/dt_ref = a * 2 * 10.
    REQUIRE(derivatives->size() == 2U);
    REQUIRE(derivatives->at(0) == Catch::Approx(180.0));
    REQUIRE(derivatives->at(1) == Catch::Approx(0.3 * 2.0 * 10.0));

    model.parameters->add_parameter("u", 3.0);
    const auto& direct = model.formulas.add_formula("QI2", "cp(u)");
    model.formulas.calculate_derivatives();
    REQUIRE(model.formulas.find("QI2")->get_derivative_values()->at(0) ==
            Catch::Approx(10.0));
    REQUIRE(direct.get_value() == Catch::Approx(130.0));
}
