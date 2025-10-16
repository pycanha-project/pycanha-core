#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <memory>

#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/couplings.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {
std::shared_ptr<Nodes> build_sample_nodes() {
    auto nodes = std::make_shared<Nodes>();

    Node diffusive_a(10);
    Node diffusive_b(20);
    Node boundary_c(30);
    boundary_c.set_type(NodeType::BOUNDARY_NODE);

    nodes->add_node(diffusive_a);
    nodes->add_node(diffusive_b);
    nodes->add_node(boundary_c);

    return nodes;
}
}  // namespace

TEST_CASE("Couplings stores and retrieves conductances", "[couplings]") {
    auto nodes = build_sample_nodes();
    Couplings couplings(nodes);

    SECTION("Adding new coupling creates entry and value is retrievable") {
        constexpr double coupling_value = 12.5;
        couplings.add_new_coupling(10, 20, coupling_value);

        REQUIRE(couplings.coupling_exists(10, 20));
        REQUIRE_THAT(couplings.get_coupling_value(10, 20),
                     Catch::Matchers::WithinAbs(coupling_value, 1e-12));

        auto* value_ptr = couplings.get_coupling_value_ref(10, 20);
        REQUIRE((value_ptr != nullptr));
        REQUIRE_THAT(*value_ptr,
                     Catch::Matchers::WithinAbs(coupling_value, 1e-12));

        const auto value_address = couplings.get_coupling_value_address(10, 20);
        REQUIRE((value_address != 0U));
    }

    SECTION("Sum and overwrite semantics work correctly") {
        couplings.add_new_coupling(10, 20, 5.0);
        couplings.add_sum_coupling(10, 20, 3.0);
        REQUIRE_THAT(couplings.get_coupling_value(10, 20),
                     Catch::Matchers::WithinAbs(8.0, 1e-12));

        couplings.add_sum_coupling_verbose(10, 20, 2.0);
        REQUIRE_THAT(couplings.get_coupling_value(10, 20),
                     Catch::Matchers::WithinAbs(10.0, 1e-12));

        couplings.add_ovw_coupling(10, 20, 4.0);
        REQUIRE_THAT(couplings.get_coupling_value(10, 20),
                     Catch::Matchers::WithinAbs(4.0, 1e-12));

        couplings.add_ovw_coupling_verbose(10, 20, 6.0);
        REQUIRE_THAT(couplings.get_coupling_value(10, 20),
                     Catch::Matchers::WithinAbs(6.0, 1e-12));
    }

    SECTION("Coupling objects can be used as inputs") {
        const Coupling coupling(10, 30, 9.0);
        couplings.add_coupling(coupling);
        REQUIRE(couplings.coupling_exists(10, 30));

        const Coupling increment(10, 30, 3.0);
        couplings.add_sum_coupling(increment);
        REQUIRE_THAT(couplings.get_coupling_value(10, 30),
                     Catch::Matchers::WithinAbs(12.0, 1e-12));

        const Coupling overwrite(10, 30, 2.0);
        couplings.add_ovw_coupling(overwrite);
        REQUIRE_THAT(couplings.get_coupling_value(10, 30),
                     Catch::Matchers::WithinAbs(2.0, 1e-12));
    }

    SECTION("Retrieving coupling by index reflects stored data") {
        couplings.add_new_coupling(10, 20, 5.0);
        couplings.add_new_coupling(10, 30, 3.0);

        const Coupling first_coupling =
            couplings.get_coupling_from_coupling_idx(0);
        REQUIRE_THAT(first_coupling.get_value(),
                     Catch::Matchers::WithinAbs(5.0, 1e-12));

        const Coupling second_coupling =
            couplings.get_coupling_from_coupling_idx(1);
        REQUIRE_THAT(second_coupling.get_value(),
                     Catch::Matchers::WithinAbs(3.0, 1e-12));
    }

    SECTION("Invalid nodes return safe defaults") {
        REQUIRE_FALSE(couplings.coupling_exists(99, 100));
        REQUIRE(std::isnan(couplings.get_coupling_value(99, 100)));
        REQUIRE((couplings.get_coupling_value_ref(99, 100) == nullptr));
        REQUIRE((couplings.get_coupling_value_address(99, 100) == 0U));
    }
}
