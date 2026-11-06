#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

// Nodes 1..count with T = 300 K, plus formulas over them.
struct StalenessModel {
    std::shared_ptr<ThermalNetwork> network =
        std::make_shared<ThermalNetwork>();
    std::shared_ptr<Parameters> parameters = std::make_shared<Parameters>();
    Formulas formulas{network, parameters};

    explicit StalenessModel(int count) { add_nodes(1, count); }

    void add_nodes(int first, int last) {
        for (int number = first; number <= last; ++number) {
            Node node(number);
            node.set_T(300.0);
            network->add_node(node);
        }
    }

    [[nodiscard]] Nodes& nodes() const { return network->nodes(); }
};

}  // namespace

TEST_CASE("Compiling formulas of an unchanged model does no work",
          "[formulas][staleness]") {
    StalenessModel model(2);
    static_cast<void>(model.formulas.add_formula("QI2", "2 * T1"));
    model.formulas.compile_formulas();
    REQUIRE(model.formulas.is_compiled_current());

    // New values are not a change of structure.
    model.nodes().set_T(1, 5.0);
    model.formulas.compile_formulas();
    REQUIRE(model.formulas.is_compiled_current());
    model.formulas.apply_compiled_formulas();
    REQUIRE(model.nodes().get_qi(2) == Catch::Approx(10.0));
}

TEST_CASE("Adding a formula makes the compiled formulas stale",
          "[formulas][staleness]") {
    StalenessModel model(2);
    static_cast<void>(model.formulas.add_formula("QI2", "2 * T1"));
    model.formulas.compile_formulas();

    static_cast<void>(model.formulas.add_formula("QI1", "1.0"));
    REQUIRE_FALSE(model.formulas.is_validation_current());
    REQUIRE_THROWS_AS(model.formulas.apply_compiled_formulas(),
                      std::runtime_error);
}

TEST_CASE("Adding nodes makes compiled formulas stale until recompiled",
          "[formulas][staleness]") {
    StalenessModel model(2);
    static_cast<void>(model.formulas.add_formula("QI2", "2 * T1"));
    model.formulas.compile_formulas();

    // Enough nodes to move the node storage the compiled pointers point into.
    model.add_nodes(3, 2000);
    REQUIRE_FALSE(model.formulas.is_compiled_current());
    REQUIRE_THROWS_AS(model.formulas.apply_compiled_formulas(),
                      std::runtime_error);

    model.formulas.compile_formulas();
    model.nodes().set_T(1, 7.0);
    model.formulas.apply_compiled_formulas();
    REQUIRE(model.nodes().get_qi(2) == Catch::Approx(14.0));
}

TEST_CASE("Reserving node storage that moves the values is a structure change",
          "[formulas][staleness]") {
    StalenessModel model(2);
    static_cast<void>(model.formulas.add_formula("QI2", "2 * T1"));
    model.formulas.compile_formulas();
    const std::uint64_t before = model.nodes().structure_version();

    model.nodes().reserve(2);
    REQUIRE(model.nodes().structure_version() == before);
    REQUIRE(model.formulas.is_compiled_current());

    model.nodes().reserve(100000);
    REQUIRE(model.nodes().structure_version() != before);
    REQUIRE_FALSE(model.formulas.is_compiled_current());
}

TEST_CASE("A node used by formulas can be removed; compiling names them",
          "[formulas][staleness]") {
    StalenessModel model(4);
    model.network->conductive_couplings().add_coupling(2, 3, 0.1);
    static_cast<void>(model.formulas.add_formula("QI1", "T3"));
    static_cast<void>(model.formulas.add_formula("QI3", "1.0"));
    static_cast<void>(model.formulas.add_formula("GL(2,3)", "0.5"));
    static_cast<void>(model.formulas.add_formula("QI2", "2.0"));
    model.formulas.compile_formulas();

    model.network->remove_node(3);
    REQUIRE(model.nodes().num_nodes() == 3);
    REQUIRE_THROWS_WITH(model.formulas.compile_formulas(),
                        Catch::Matchers::ContainsSubstring("3 of 4 formulas") &&
                            Catch::Matchers::ContainsSubstring("QI1") &&
                            Catch::Matchers::ContainsSubstring("GL(2,3)"));
}

TEST_CASE("Removing the broken formulas lets the formulas compile again",
          "[formulas][staleness]") {
    StalenessModel model(3);
    static_cast<void>(model.formulas.add_formula("QI1", "T3"));
    static_cast<void>(model.formulas.add_formula("QI2", "2 * T1"));
    model.network->remove_node(3);
    REQUIRE_THROWS_AS(model.formulas.validate_for_execution(),
                      std::runtime_error);

    REQUIRE(model.formulas.remove_formula("QI1"));
    model.formulas.compile_formulas();
    model.formulas.apply_compiled_formulas();
    REQUIRE(model.nodes().get_qi(2) == Catch::Approx(600.0));
}

TEST_CASE("Compiled coupling formulas survive a later matrix compression",
          "[formulas][staleness]") {
    StalenessModel model(5);
    // Single insertions out of order leave the matrices uncompressed.
    auto& couplings = model.network->conductive_couplings();
    couplings.add_coupling(4, 5, 0.1);
    couplings.add_coupling(1, 2, 0.1);
    couplings.add_coupling(3, 4, 0.1);
    couplings.add_coupling(1, 3, 0.1);
    static_cast<void>(model.formulas.add_formula("GL(4,5)", "0.25"));
    model.formulas.compile_formulas();

    // A solver initialization compresses the matrices the same way.
    model.network->conductive_matrices().compress();
    REQUIRE(model.formulas.is_compiled_current());
    model.formulas.apply_compiled_formulas();
    REQUIRE(couplings.get_coupling_value(4, 5) == Catch::Approx(0.25));
}
