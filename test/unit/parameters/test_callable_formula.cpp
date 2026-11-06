#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

// Three nodes. QI1 reads a parameter, a callable sets QI2 from QI1, and QI3
// reads QI2: the result depends on the callable running between the two.
struct CallableModel {
    std::shared_ptr<ThermalNetwork> network =
        std::make_shared<ThermalNetwork>();
    std::shared_ptr<Parameters> parameters = std::make_shared<Parameters>();
    Formulas formulas{network, parameters};
    std::shared_ptr<int> calls = std::make_shared<int>(0);

    CallableModel() {
        for (int number = 1; number <= 3; ++number) {
            Node node(number);
            network->add_node(node);
        }
        parameters->add_parameter("gain", 3.0);
        static_cast<void>(formulas.add_formula("QI1", "gain"));
        static_cast<void>(formulas.add_callable_formula(
            "QI2", [nodes_network = network, counter = calls] {
                ++*counter;
                return 10.0 * nodes_network->nodes().get_qi(1);
            }));
        static_cast<void>(formulas.add_formula("QI3", "QI2 + 1"));
    }

    [[nodiscard]] double qi(int node) const {
        return network->nodes().get_qi(node);
    }
};

}  // namespace

TEST_CASE("Callable formulas run at their insertion position",
          "[formulas][callable]") {
    CallableModel model;
    model.formulas.apply_formulas();
    REQUIRE(model.qi(2) == Catch::Approx(30.0));
    REQUIRE(model.qi(3) == Catch::Approx(31.0));

    model.formulas.validate_for_execution();
    model.formulas.compile_formulas();
    model.formulas.lock_parameters_for_execution();
    model.parameters->set_parameter("gain", 4.0);
    model.formulas.apply_formulas();
    model.formulas.unlock_parameters();
    REQUIRE(model.qi(2) == Catch::Approx(40.0));
    REQUIRE(model.qi(3) == Catch::Approx(41.0));
    REQUIRE(*model.calls == 2);
}

TEST_CASE("Callable formulas list no inputs and have no derivatives",
          "[formulas][callable]") {
    CallableModel model;
    const auto& callable = *model.formulas.formulas().at(1);
    REQUIRE(callable.kind() == FormulaKind::Callable);
    REQUIRE(callable.parameter_dependencies().empty());
    REQUIRE(callable.entity_dependencies().empty());
    REQUIRE(std::isnan(callable.get_value()));

    // Only QI1 reads parameters; the callable and QI3 are skipped.
    model.formulas.calculate_derivatives();
    REQUIRE(model.formulas.formulas().at(1)->get_derivative_values() ==
            nullptr);
    REQUIRE(model.formulas.dependencies().entities.size() == 1U);
}

TEST_CASE("Callable formulas need a function and share it with their clones",
          "[formulas][callable]") {
    CallableModel model;
    REQUIRE_THROWS_AS(
        model.formulas.add_callable_formula("QR1", CallableFormula::Function{}),
        std::invalid_argument);
    REQUIRE_FALSE(model.network->nodes().is_load_dense(NodeAttribute::QR));

    model.formulas.apply_formulas();
    const auto clone = model.formulas.formulas().at(1)->clone();
    REQUIRE(clone->get_value() == Catch::Approx(30.0));
    model.network->nodes().set_qi(1, 5.0);
    const auto& function =
        dynamic_cast<const CallableFormula&>(*clone).function();
    REQUIRE(function() == Catch::Approx(50.0));
    REQUIRE(*model.calls == 2);
}

TEST_CASE("Callable formulas run before the model's user callbacks",
          "[formulas][callable]") {
    ThermalMathematicalModel model("callable");
    model.add_node(1);
    static_cast<void>(
        model.formulas().add_callable_formula("QI1", [] { return 7.0; }));
    static_cast<void>(model.formulas().add_formula("QS1", "2 * QI1"));

    model.c_extern_callback_solver_loop = [](ThermalMathematicalModel* tmm) {
        tmm->nodes().set_qr(1, tmm->nodes().get_qs(1) + 1.0);
    };
    model.c_callbacks_active = true;
    model.callback_solver_loop();

    REQUIRE(model.nodes().get_qi(1) == Catch::Approx(7.0));
    REQUIRE(model.nodes().get_qs(1) == Catch::Approx(14.0));
    REQUIRE(model.nodes().get_qr(1) == Catch::Approx(15.0));
}
