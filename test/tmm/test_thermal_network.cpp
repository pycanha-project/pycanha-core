#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("ThermalNetwork adds diffusive nodes", "[thermalnetwork]") {
    ThermalNetwork network;

    Node node1(1);
    Node node2(5);

    network.add_node(node1);
    network.add_node(node2);

    REQUIRE(network.nodes().num_nodes() == 2);
    REQUIRE(network.nodes().get_num_diff_nodes() == 2);

    auto& conductive = network.conductive_couplings();
    conductive.add_coupling(1, 5, 42.0);

    REQUIRE(conductive.get_coupling_value(1, 5) == Catch::Approx(42.0));
}

TEST_CASE("ThermalNetwork handles boundary nodes", "[thermalnetwork]") {
    ThermalNetwork network;

    Node diffusive(1);
    Node boundary(10);
    boundary.set_type('B');

    network.add_node(diffusive);
    network.add_node(boundary);

    REQUIRE(network.nodes().num_nodes() == 2);
    REQUIRE(network.nodes().get_num_bound_nodes() == 1);

    auto& conductive = network.conductive_couplings();
    auto& radiative = network.radiative_couplings();

    conductive.add_coupling(1, 10, 5.5);
    radiative.add_coupling(1, 10, 7.5);

    REQUIRE(conductive.get_coupling_value(1, 10) == Catch::Approx(5.5));
    REQUIRE(radiative.get_coupling_value(1, 10) == Catch::Approx(7.5));
}

TEST_CASE("ThermalNetwork removes nodes and couplings", "[thermalnetwork]") {
    ThermalNetwork network;

    Node node1(1);
    Node node2(3);

    network.add_node(node1);
    network.add_node(node2);

    auto& conductive = network.conductive_couplings();
    conductive.add_coupling(1, 3, 12.0);

    REQUIRE(conductive.get_coupling_value(1, 3) == Catch::Approx(12.0));

    network.remove_node(Index{3});

    REQUIRE(network.nodes().num_nodes() == 1);
    REQUIRE(std::isnan(conductive.get_coupling_value(1, 3)));

    Node duplicate(1);
    network.add_node(duplicate);
    REQUIRE(network.nodes().num_nodes() == 1);
}
