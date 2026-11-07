#include <Eigen/Core>
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formula_shape.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/origin.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

// NOLINTBEGIN(readability-function-cognitive-complexity)

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

constexpr int node_count = 12;

// Nodes 1..12 at T = 10 * n K, couplings GL(n, n+1) = 1, the parameter
// `p` = 2 and the table k(T) = T / 10 on [0, 1000] K.
struct TemplateModel {
    std::shared_ptr<ThermalNetwork> network =
        std::make_shared<ThermalNetwork>();
    std::shared_ptr<Parameters> parameters = std::make_shared<Parameters>();
    Formulas formulas{network, parameters};

    TemplateModel() {
        for (int number = 1; number <= node_count; ++number) {
            Node node(number);
            node.set_T(10.0 * number);
            network->add_node(node);
        }
        for (int number = 1; number < node_count; ++number) {
            network->conductive_couplings().add_coupling(number, number + 1,
                                                         1.0);
        }
        parameters->add_parameter("p", 2.0);
        parameters->add_table("k", LookupTable1D(Eigen::Vector2d(0.0, 1000.0),
                                                 Eigen::Vector2d(0.0, 100.0)));
    }

    [[nodiscard]] double gl(int node_1, int node_2) const {
        return network->conductive_couplings().get_coupling_value(node_1,
                                                                  node_2);
    }
    [[nodiscard]] double qi(int node) const {
        return network->nodes().get_qi(node);
    }

    // Targets in insertion order, as their string form.
    [[nodiscard]] std::vector<std::string> order() const {
        std::vector<std::string> targets;
        std::ranges::transform(
            formulas.formulas(), std::back_inserter(targets),
            [](const auto& formula) {
                return formula->entity().string_representation();
            });
        return targets;
    }

    [[nodiscard]] std::string expression(std::string_view target) const {
        const auto formula = formulas.find(target);
        return dynamic_cast<const ExpressionFormula&>(*formula).expression();
    }
};

// GL(a, a+1) = g * k((Ta + Tb) / 2) for a = first..last.
struct CouplingSlots {
    std::vector<NodeNum> a;
    std::vector<NodeNum> b;
    std::vector<double> g;

    CouplingSlots(int first, int last) {
        for (int node = first; node <= last; ++node) {
            a.push_back(node);
            b.push_back(node + 1);
            g.push_back(0.5 * node);
        }
    }

    [[nodiscard]] std::vector<NodeSlot> nodes() const {
        return {NodeSlot{.name = "a", .values = a},
                NodeSlot{.name = "b", .values = b}};
    }
    [[nodiscard]] std::vector<NumberSlot> numbers() const {
        return {NumberSlot{.name = "g", .values = g}};
    }
};

FormulaTemplate coupling_template() {
    return {.target = "GL({a},{b})",
            .expression = "{g} * k((T{a} + T{b}) / 2)"};
}

// What the coupling template gives for GL(a, a+1): 0.5 a * (10 a + 5) / 10.
[[nodiscard]] double coupling_value(int node) {
    return 0.5 * node * ((10.0 * node) + 5.0) / 10.0;
}

[[nodiscard]] std::string key_of(ThermalNetwork& network, const Entity& target,
                                 const std::string& expression,
                                 const Parameters& parameters) {
    return detail::templatize(target, expression, &parameters, &network)
        .text.key();
}

}  // namespace

TEST_CASE("Formulas written alike apart from their numbers share a shape",
          "[formulas][templates]") {
    const TemplateModel model;
    auto& network = *model.network;
    const auto& parameters = *model.parameters;
    const auto first = key_of(network, Entity::gl(network, 1, 2),
                              "0.0125 * k((T1 + T2) / 2)", parameters);
    REQUIRE(first == key_of(network, Entity::gl(network, 5, 6),
                            "0.5 * k((T5+T6) / 2)", parameters));
    REQUIRE(first != key_of(network, Entity::gl(network, 5, 6),
                            "0.5 * k((T5 + T7) / 2)", parameters));
    // Exponents stay in the shape; parameters too.
    REQUIRE(key_of(network, Entity::qi(network, 1), "T1**2", parameters) !=
            key_of(network, Entity::qi(network, 1), "T1**3", parameters));
    REQUIRE(key_of(network, Entity::qi(network, 1), "p * T1", parameters) ==
            "QI{0}=p*T{0}");
}

TEST_CASE("Formulas of interleaved shapes keep their own values",
          "[formulas][templates]") {
    TemplateModel model;
    // As a reader writes them, in node order: every third formula has
    // another expression.
    for (int node = 1; node < node_count; ++node) {
        const auto a = std::to_string(node);
        const auto b = std::to_string(node + 1);
        std::string expression = (node % 3 == 0)
                                     ? "0.1 * (T"
                                     : std::to_string(0.5 * node) + " * k((T";
        expression += a;
        expression += " + T";
        expression += b;
        expression += (node % 3 == 0) ? ")" : ") / 2)";
        std::string target = "GL(";
        target += a;
        target += ",";
        target += b;
        target += ")";
        model.formulas.add_formula(target, expression);
    }
    model.formulas.apply_formulas();
    for (int node = 1; node < node_count; ++node) {
        INFO(node);
        const double expected =
            (node % 3 == 0) ? 0.1 * ((10.0 * node) + ((10.0 * node) + 10))
                            : coupling_value(node);
        REQUIRE(model.gl(node, node + 1) == Catch::Approx(expected));
    }
    REQUIRE(model.expression("GL(3,4)") == "0.1 * (T3 + T4)");
}

TEST_CASE("A template adds one formula per slot value",
          "[formulas][templates]") {
    TemplateModel model;
    const CouplingSlots slots(1, 5);
    model.formulas.add_formulas(coupling_template(), slots.nodes(),
                                slots.numbers());
    REQUIRE(model.formulas.size() == 5U);
    model.formulas.apply_formulas();
    for (int node = 1; node <= 5; ++node) {
        REQUIRE(model.gl(node, node + 1) ==
                Catch::Approx(coupling_value(node)));
    }

    const auto formula = model.formulas.find("GL(2,3)");
    REQUIRE(formula->origin() == Origin::user);
    REQUIRE(formula->kind() == FormulaKind::Entities);
    REQUIRE(formula->entity_dependencies() ==
            std::vector<Entity>{Entity::t(*model.network, 2),
                                Entity::t(*model.network, 3)});
    REQUIRE(formula->table_dependencies() == std::vector<std::string>{"k"});
    REQUIRE(model.expression("GL(2,3)") == "1 * k((T2 + T3) / 2)");
    REQUIRE(formula->get_value() == Catch::Approx(coupling_value(2)));
}

TEST_CASE("A malformed template call adds nothing", "[formulas][templates]") {
    TemplateModel model;
    const CouplingSlots slots(1, 3);
    auto& formulas = model.formulas;
    const auto refuses = [&](const FormulaTemplate& shape,
                             const std::vector<NodeSlot>& nodes,
                             const std::vector<NumberSlot>& numbers) {
        REQUIRE_THROWS_AS(formulas.add_formulas(shape, nodes, numbers),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(
            formulas.add_generated_formulas(Origin::conduction_builder, shape,
                                            nodes, numbers),
            std::invalid_argument);
    };
    // A slot without values, a slot the template does not use, a slot of
    // the wrong kind, slots of different lengths.
    refuses(coupling_template(), {slots.nodes().front()}, slots.numbers());
    auto extra = slots.numbers();
    extra.push_back(NumberSlot{.name = "h", .values = slots.g});
    refuses(coupling_template(), slots.nodes(), extra);
    refuses(coupling_template(), slots.nodes(),
            {NumberSlot{.name = "a", .values = slots.g}});
    const std::vector<double> short_g{1.0};
    refuses(coupling_template(), slots.nodes(),
            {NumberSlot{.name = "g", .values = short_g}});
    // A target that is not one entity; a name used as a node and a number.
    refuses({.target = "{a} + 1", .expression = "{g}"}, slots.nodes(),
            slots.numbers());
    refuses({.target = "QI{a}", .expression = "{a} * T{a}"},
            {slots.nodes().front()}, {});
    REQUIRE(formulas.size() == 0U);
}

TEST_CASE("A user template checks every formula before adding any",
          "[formulas][templates]") {
    TemplateModel model;
    model.formulas.add_formula("GL(3,4)", "1.0");
    const CouplingSlots slots(1, 4);
    REQUIRE_THROWS_AS(model.formulas.add_formulas(
                          coupling_template(), slots.nodes(), slots.numbers()),
                      std::invalid_argument);
    REQUIRE(model.formulas.size() == 1U);

    // The same target twice.
    const std::vector<NodeNum> nodes{1, 1};
    const std::vector<double> values{1.0, 2.0};
    REQUIRE_THROWS_AS(
        model.formulas.add_formulas(
            {.target = "QI{n}", .expression = "{q}"},
            std::vector<NodeSlot>{{.name = "n", .values = nodes}},
            std::vector<NumberSlot>{{.name = "q", .values = values}}),
        std::invalid_argument);
    REQUIRE(model.formulas.size() == 1U);
}

TEST_CASE("Generated templates follow the collision rules per formula",
          "[formulas][templates]") {
    TemplateModel model;
    model.formulas.add_formula("GL(2,3)", "1.0");
    static_cast<void>(model.formulas.add_generated_formulas(
        Origin::esatan_tmm_reader,
        std::vector<GeneratedFormula>{
            {.target = Entity::gl(*model.network, 3, 4),
             .expression = "2.0"}}));

    // GL(12,13) does not exist, so its temperatures are not read either.
    std::vector<NodeNum> a{1, 2, 3, 12};
    std::vector<NodeNum> b{2, 3, 4, 13};
    std::vector<double> g{1.0, 1.0, 1.0, 1.0};
    const auto report = model.formulas.add_generated_formulas(
        Origin::conduction_builder, coupling_template(),
        std::vector<NodeSlot>{{.name = "a", .values = a},
                              {.name = "b", .values = b}},
        std::vector<NumberSlot>{{.name = "g", .values = g}});
    REQUIRE(report.added == 1U);
    REQUIRE(report.kept_user == 1U);
    REQUIRE(report.replaced == 1U);
    REQUIRE(report.failed.size() == 1U);
    REQUIRE(model.formulas.find("GL(2,3)")->origin() == Origin::user);
    REQUIRE(model.formulas.find("GL(3,4)")->origin() ==
            Origin::conduction_builder);
    REQUIRE(model.order() ==
            std::vector<std::string>{"GL(2,3)", "GL(1,2)", "GL(3,4)"});

    REQUIRE(model.formulas.remove_formulas(Origin::conduction_builder) == 2U);
    REQUIRE(model.order() == std::vector<std::string>{"GL(2,3)"});
}

TEST_CASE("One formula of a template is removed and added back on its own",
          "[formulas][templates]") {
    TemplateModel model;
    const CouplingSlots slots(1, 5);
    static_cast<void>(model.formulas.add_generated_formulas(
        Origin::conduction_builder, coupling_template(), slots.nodes(),
        slots.numbers()));
    model.formulas.apply_formulas();

    REQUIRE(model.formulas.remove_formula("GL(2,3)"));
    REQUIRE_FALSE(model.formulas.contains("GL(2,3)"));
    model.network->conductive_couplings().set_coupling_value(2, 3, 7.0);
    model.formulas.apply_formulas();
    REQUIRE(model.gl(2, 3) == Catch::Approx(7.0));
    REQUIRE(model.gl(4, 5) == Catch::Approx(coupling_value(4)));

    model.formulas.add_formula("GL(2,3)", "3 * p");
    REQUIRE(model.formulas.find("GL(2,3)")->origin() == Origin::user);
    REQUIRE(model.order().back() == "GL(2,3)");
    model.formulas.apply_formulas();
    REQUIRE(model.gl(2, 3) == Catch::Approx(6.0));
}

TEST_CASE("Removing most formulas of a template packs it and keeps the rest",
          "[formulas][templates]") {
    TemplateModel model;
    const CouplingSlots slots(1, 10);
    model.formulas.add_formulas(coupling_template(), slots.nodes(),
                                slots.numbers());
    for (int node = 1; node <= 7; ++node) {
        REQUIRE(model.formulas.remove_formula(
            Entity::gl(*model.network, node, node + 1)));
    }
    model.formulas.apply_formulas();
    REQUIRE(model.order() ==
            std::vector<std::string>{"GL(8,9)", "GL(9,10)", "GL(10,11)"});
    for (int node = 8; node <= 10; ++node) {
        REQUIRE(model.gl(node, node + 1) ==
                Catch::Approx(coupling_value(node)));
        REQUIRE(model.formulas.find(Entity::gl(*model.network, node, node + 1))
                    ->get_value() == Catch::Approx(coupling_value(node)));
    }

    // The packed batch takes new formulas of its shape.
    const CouplingSlots more(1, 2);
    model.formulas.add_formulas(coupling_template(), more.nodes(),
                                more.numbers());
    model.formulas.apply_formulas();
    REQUIRE(model.gl(1, 2) == Catch::Approx(coupling_value(1)));
    REQUIRE(model.order().size() == 5U);
}

TEST_CASE("Adding formulas to a compiled template makes it stale",
          "[formulas][templates]") {
    TemplateModel model;
    const CouplingSlots slots(1, 2);
    model.formulas.add_formulas(coupling_template(), slots.nodes(),
                                slots.numbers());
    model.formulas.compile_formulas();
    REQUIRE(model.formulas.is_compiled_current());

    // Enough formulas of the same shape to move its storage.
    const CouplingSlots more(3, 11);
    model.formulas.add_formulas(coupling_template(), more.nodes(),
                                more.numbers());
    REQUIRE_FALSE(model.formulas.is_compiled_current());
    model.formulas.apply_formulas();
    for (int node = 1; node <= 11; ++node) {
        REQUIRE(model.gl(node, node + 1) ==
                Catch::Approx(coupling_value(node)));
    }
}

TEST_CASE("Formulas of one shape that read each other keep their order",
          "[formulas][templates]") {
    TemplateModel model;
    model.formulas.add_formula("QI1", "4.0");
    for (int node = 2; node <= 5; ++node) {
        model.formulas.add_formula("QI" + std::to_string(node),
                                   "0.5 * QI" + std::to_string(node - 1));
    }
    model.formulas.apply_formulas();
    REQUIRE(model.qi(5) == Catch::Approx(0.25));
}

TEST_CASE("Numbers shared by formulas of one shape give the same results",
          "[formulas][templates]") {
    TemplateModel model;
    // QI1..QI3 share 0.5, QI4..QI6 have their own factors, and the QR
    // formulas read their shared number inside a function without an
    // operation of its own.
    for (int node = 1; node <= 6; ++node) {
        const auto factor = node <= 3 ? 0.5 : static_cast<double>(node);
        model.formulas.add_formula(
            "QI" + std::to_string(node),
            std::to_string(factor) + " * T" + std::to_string(node) + " + 1");
        model.formulas.add_formula(
            "QR" + std::to_string(node),
            "gamma(0.25 * T" + std::to_string(node) + ") / 2");
    }
    model.formulas.apply_formulas();
    for (int node = 1; node <= 6; ++node) {
        INFO(node);
        const auto factor = node <= 3 ? 0.5 : static_cast<double>(node);
        REQUIRE(model.qi(node) == Catch::Approx((factor * 10.0 * node) + 1.0));
        REQUIRE(model.network->nodes().get_qr(node) ==
                Catch::Approx(std::tgamma(2.5 * node) / 2.0));
    }
}

TEST_CASE("Moving a formula changes when it runs", "[formulas][templates]") {
    TemplateModel model;
    model.formulas.add_formula("QI2", "QI1 + 1");
    model.formulas.add_formula("QI1", "5.0");
    model.formulas.add_formula("QI3", "QI1 * 2");
    model.formulas.apply_formulas();
    REQUIRE(model.qi(2) == Catch::Approx(1.0));

    model.network->nodes().set_qi(1, 0.0);
    REQUIRE(model.formulas.move_formula("QI1", "QI2"));
    REQUIRE_FALSE(model.formulas.is_compiled_current());
    REQUIRE(model.order() == std::vector<std::string>{"QI1", "QI2", "QI3"});
    model.formulas.apply_formulas();
    REQUIRE(model.qi(2) == Catch::Approx(6.0));

    REQUIRE(model.formulas.move_formula_to_end("QI2"));
    REQUIRE(model.order() == std::vector<std::string>{"QI1", "QI3", "QI2"});
    REQUIRE_FALSE(model.formulas.move_formula("QI4", "QI1"));
    REQUIRE_FALSE(model.formulas.move_formula_to_end("QI4"));
}

TEST_CASE("Expressions are printed with each formula's numbers",
          "[formulas][templates]") {
    TemplateModel model;
    model.formulas.add_formula("QI1", "T1**2 - 2.5e-3*T2 + p");
    REQUIRE(model.expression("QI1") == "T1**2 - 0.0025*T2 + p");

    const std::vector<NodeNum> nodes{3};
    const std::vector<double> values{-1.5};
    model.formulas.add_formulas(
        {.target = "QI{n}", .expression = "T{n} + {q}"},
        std::vector<NodeSlot>{{.name = "n", .values = nodes}},
        std::vector<NumberSlot>{{.name = "q", .values = values}});
    REQUIRE(model.expression("QI3") == "T3 + (-1.5)");
    model.formulas.apply_formulas();
    REQUIRE(model.qi(3) == Catch::Approx(28.5));
}

TEST_CASE("Templates that read parameters have derivatives per formula",
          "[formulas][templates]") {
    TemplateModel model;
    const std::vector<NodeNum> nodes{1, 2};
    const std::vector<double> gains{3.0, 4.0};
    model.formulas.add_formulas(
        {.target = "QI{n}", .expression = "{g} * p**2"},
        std::vector<NodeSlot>{{.name = "n", .values = nodes}},
        std::vector<NumberSlot>{{.name = "g", .values = gains}});
    const auto second = model.formulas.find("QI2");
    REQUIRE(second->kind() == FormulaKind::Parameters);
    second->calculate_derivatives();
    // d/dp (4 p^2) at p = 2.
    REQUIRE(second->get_derivative_values()->at(0) == Catch::Approx(16.0));
}

TEST_CASE("Names that look like placeholders are read as written",
          "[formulas][templates]") {
    TemplateModel model;
    model.parameters->add_parameter("_fn0", 7.0);
    model.formulas.add_formula("QI1", "_fn0 + 1");
    model.formulas.apply_formulas();
    REQUIRE(model.qi(1) == Catch::Approx(8.0));
}

TEST_CASE("Dependencies list the targets of the formulas that read",
          "[formulas][templates]") {
    TemplateModel model;
    const CouplingSlots slots(1, 3);
    model.formulas.add_formulas(coupling_template(), slots.nodes(),
                                slots.numbers());
    const auto dependencies = model.formulas.dependencies();
    const auto& readers =
        dependencies.entities.at(Entity::t(*model.network, 2));
    REQUIRE(readers == std::vector<Entity>{Entity::gl(*model.network, 1, 2),
                                           Entity::gl(*model.network, 2, 3)});
    REQUIRE(dependencies.tables.at("k").size() == 3U);
}

// NOLINTEND(readability-function-cognitive-complexity)
