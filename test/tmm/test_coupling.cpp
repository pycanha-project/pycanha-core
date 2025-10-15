#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/tmm/coupling.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("Coupling stores nodes and value", "[coupling]") {
    const int node_1 = 2;
    const int node_2 = 7;
    const double conductor_value = 5.0;
    Coupling coupling(node_1, node_2, conductor_value);

    // NOLINTBEGIN(bugprone-chained-comparison)
    REQUIRE(node_1 == coupling.get_node_1());
    REQUIRE(node_2 == coupling.get_node_2());
    REQUIRE(conductor_value == coupling.get_value());

    SECTION("Updating first node keeps other attributes") {
        const int new_node_1 = 17;
        coupling.set_node_1(new_node_1);

        REQUIRE(new_node_1 == coupling.get_node_1());
        REQUIRE(node_2 == coupling.get_node_2());
        REQUIRE(conductor_value == coupling.get_value());
    }

    SECTION("Updating second node keeps other attributes") {
        const int new_node_2 = 29;
        coupling.set_node_2(new_node_2);

        REQUIRE(node_1 == coupling.get_node_1());
        REQUIRE(new_node_2 == coupling.get_node_2());
        REQUIRE(conductor_value == coupling.get_value());
    }

    SECTION("Updating value keeps node indices") {
        const double new_value = 7.8;
        coupling.set_value(new_value);

        REQUIRE(node_1 == coupling.get_node_1());
        REQUIRE(node_2 == coupling.get_node_2());
        REQUIRE(new_value == coupling.get_value());
    }

    SECTION("Updating all attributes reflects in getters") {
        const int new_node_1 = 17;
        const int new_node_2 = 29;
        const double new_value = 7.8;

        coupling.set_node_1(new_node_1);
        coupling.set_node_2(new_node_2);
        coupling.set_value(new_value);

        REQUIRE(new_node_1 == coupling.get_node_1());
        REQUIRE(new_node_2 == coupling.get_node_2());
        REQUIRE(new_value == coupling.get_value());
    }
    // NOLINTEND(bugprone-chained-comparison)
}
