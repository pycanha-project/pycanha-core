#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

// Four nodes and their formulas.
struct GeneratedModel {
    std::shared_ptr<ThermalNetwork> network =
        std::make_shared<ThermalNetwork>();
    std::shared_ptr<Parameters> parameters = std::make_shared<Parameters>();
    Formulas formulas{network, parameters};

    GeneratedModel() {
        for (int number = 1; number <= 4; ++number) {
            Node node(number);
            network->add_node(node);
        }
    }

    [[nodiscard]] Entity qi(NodeNum node) const {
        return Entity::qi(*network, node);
    }

    // One generated formula per (node, expression) pair, all on QI.
    [[nodiscard]] GenerationReport generate(
        FormulaOrigin origin,
        const std::vector<std::pair<NodeNum, std::string>>& entries) {
        std::vector<GeneratedFormula> generated;
        generated.reserve(entries.size());
        for (const auto& [node, expression] : entries) {
            generated.push_back(
                GeneratedFormula{.target = qi(node), .expression = expression});
        }
        return formulas.add_generated_formulas(origin, generated);
    }

    [[nodiscard]] FormulaOrigin origin_of(NodeNum node) const {
        return formulas.find(qi(node))->origin();
    }

    // Targets in insertion order, as their string form.
    [[nodiscard]] std::vector<std::string> order() const {
        const auto& list = formulas.formulas();
        std::vector<std::string> targets(list.size());
        std::ranges::transform(list, targets.begin(), [](const auto& formula) {
            return formula->entity().string_representation();
        });
        return targets;
    }
};

}  // namespace

TEST_CASE("Formulas added one at a time are user formulas",
          "[formulas][origin]") {
    GeneratedModel model;
    static_cast<void>(model.formulas.add_formula("QI1", 1.0));
    static_cast<void>(model.formulas.add_formula("QI2", "2 * QI1"));
    static_cast<void>(
        model.formulas.add_callable_formula("QI3", [] { return 3.0; }));

    REQUIRE(model.origin_of(1) == FormulaOrigin::user);
    REQUIRE(model.origin_of(2) == FormulaOrigin::user);
    REQUIRE(model.origin_of(3) == FormulaOrigin::user);
    REQUIRE(to_string(FormulaOrigin::conduction_builder) ==
            "conduction_builder");
}

TEST_CASE("A generated formula added again one at a time is a user formula",
          "[formulas][origin]") {
    GeneratedModel model;
    static_cast<void>(model.generate(FormulaOrigin::etms_reader, {{1, "5.0"}}));
    const auto generated = model.formulas.find("QI1");
    REQUIRE(generated->origin() == FormulaOrigin::etms_reader);

    const auto copy = generated->clone();
    REQUIRE(model.formulas.remove_formula("QI1"));
    model.formulas.add_formula(*copy);
    REQUIRE(model.origin_of(1) == FormulaOrigin::user);
}

TEST_CASE("Generated formulas keep their origin and are found by target",
          "[formulas][origin]") {
    GeneratedModel model;
    const auto report = model.generate(FormulaOrigin::conduction_builder,
                                       {{1, "1.0"}, {2, "2 * QI1"}});

    REQUIRE(report.added == 2U);
    REQUIRE(model.origin_of(2) == FormulaOrigin::conduction_builder);
    REQUIRE(model.formulas.contains(model.qi(1)));
    REQUIRE(model.formulas.contains("QI2"));
    REQUIRE_FALSE(model.formulas.contains("QI3"));
    REQUIRE(model.formulas.find("not a target") == nullptr);
}

TEST_CASE("A generated formula never replaces a user formula",
          "[formulas][origin]") {
    GeneratedModel model;
    static_cast<void>(model.formulas.add_formula("QI1", "5.0"));
    const auto report =
        model.generate(FormulaOrigin::etms_reader, {{1, "7.0"}, {2, "1.0"}});

    REQUIRE(report.kept_user == 1U);
    REQUIRE(report.added == 1U);
    REQUIRE(model.origin_of(1) == FormulaOrigin::user);
    model.formulas.apply_formulas();
    REQUIRE(model.network->nodes().get_qi(1) == Catch::Approx(5.0));
}

TEST_CASE("A generated formula replaces another generated one, at the end",
          "[formulas][origin]") {
    GeneratedModel model;
    static_cast<void>(
        model.generate(FormulaOrigin::etms_reader, {{1, "1.0"}, {2, "2.0"}}));
    const auto report =
        model.generate(FormulaOrigin::conduction_builder, {{1, "3.0"}});

    REQUIRE(report.replaced == 1U);
    REQUIRE(report.added == 0U);
    REQUIRE(model.origin_of(1) == FormulaOrigin::conduction_builder);
    REQUIRE(model.order() == std::vector<std::string>{"QI2", "QI1"});
}

TEST_CASE("A generator replaces its own formula on a repeated target",
          "[formulas][origin]") {
    GeneratedModel model;
    const auto report = model.generate(FormulaOrigin::conduction_builder,
                                       {{1, "1.0"}, {1, "2.0"}});

    REQUIRE(report.added == 1U);
    REQUIRE(report.replaced == 1U);
    REQUIRE(model.formulas.formulas().size() == 1U);
    model.formulas.apply_formulas();
    REQUIRE(model.network->nodes().get_qi(1) == Catch::Approx(2.0));
}

TEST_CASE("A refused generated expression is reported and changes nothing",
          "[formulas][origin]") {
    GeneratedModel model;
    static_cast<void>(
        model.generate(FormulaOrigin::conduction_builder, {{1, "1.0"}}));
    const auto report = model.generate(FormulaOrigin::etms_reader,
                                       {{1, "1 +"}, {2, "QX7"}, {3, "3.0"}});

    REQUIRE(report.failed.size() == 2U);
    REQUIRE(report.failed.front().target.is_same_as(model.qi(1)));
    REQUIRE_FALSE(report.failed.back().message.empty());
    REQUIRE(report.added == 1U);
    REQUIRE(model.origin_of(1) == FormulaOrigin::conduction_builder);
    REQUIRE_FALSE(model.formulas.contains("QI2"));
}

TEST_CASE("Generated formulas need a generator origin", "[formulas][origin]") {
    GeneratedModel model;
    REQUIRE_THROWS_AS(model.generate(FormulaOrigin::user, {{1, "1.0"}}),
                      std::invalid_argument);
    REQUIRE(model.formulas.formulas().empty());
}

TEST_CASE("Removing an origin removes only its formulas",
          "[formulas][origin]") {
    GeneratedModel model;
    static_cast<void>(model.formulas.add_formula("QI1", "1.0"));
    static_cast<void>(
        model.generate(FormulaOrigin::conduction_builder, {{2, "2.0"}}));
    static_cast<void>(model.generate(FormulaOrigin::etms_reader, {{3, "3.0"}}));
    static_cast<void>(
        model.generate(FormulaOrigin::conduction_builder, {{4, "4.0"}}));

    REQUIRE(model.formulas.remove_formulas(FormulaOrigin::conduction_builder) ==
            2U);
    REQUIRE(model.order() == std::vector<std::string>{"QI1", "QI3"});
    REQUIRE_FALSE(model.formulas.contains("QI4"));
    REQUIRE(model.formulas.remove_formulas(FormulaOrigin::conduction_builder) ==
            0U);
}

TEST_CASE("A regeneration after its wipe writes its formulas again",
          "[formulas][origin]") {
    GeneratedModel model;
    static_cast<void>(model.generate(FormulaOrigin::conduction_builder,
                                     {{1, "1.0"}, {2, "2.0"}}));
    static_cast<void>(model.formulas.add_formula("QI3", "QI1 + QI2"));

    static_cast<void>(
        model.formulas.remove_formulas(FormulaOrigin::conduction_builder));
    const auto report = model.generate(FormulaOrigin::conduction_builder,
                                       {{1, "10.0"}, {2, "20.0"}});

    REQUIRE(report.added == 2U);
    REQUIRE(model.order() == std::vector<std::string>{"QI3", "QI1", "QI2"});
    model.formulas.apply_formulas();
    REQUIRE(model.network->nodes().get_qi(1) == Catch::Approx(10.0));
}
