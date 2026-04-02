#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

struct FlowLink {
    Index node_num_1;
    Index node_num_2;
    double value;
};

struct ThermalNetworkFlowFixture {
    ThermalNetwork network;
    const std::vector<Index> group_1{17, 11, 40};
    const std::vector<Index> group_2{2, 55};

    ThermalNetworkFlowFixture() {
        add_node(17, 290.0);
        add_node(11, 300.0);
        add_node(40, 315.0, 'B');
        add_node(2, 305.0);
        add_node(55, 280.0, 'B');

        for (const auto& link : conductive_links()) {
            network.conductive_couplings().add_coupling(
                link.node_num_1, link.node_num_2, link.value);
        }

        for (const auto& link : radiative_links()) {
            network.radiative_couplings().add_coupling(
                link.node_num_1, link.node_num_2, link.value);
        }
    }

    void add_node(Index node_num, double temperature, char type = 'D') {
        Node node(static_cast<int>(node_num));
        node.set_type(type);
        node.set_T(temperature);
        network.add_node(node);
    }

    [[nodiscard]] double expected_conductive_pair(Index node_num_1,
                                                  Index node_num_2) const {
        return lookup_link_value(conductive_links(), node_num_1, node_num_2) *
               (temperature(node_num_2) - temperature(node_num_1));
    }

    [[nodiscard]] double expected_radiative_pair(Index node_num_1,
                                                 Index node_num_2) const {
        return lookup_link_value(radiative_links(), node_num_1, node_num_2) *
               STF_BOLTZ *
               (std::pow(temperature(node_num_2), 4) -
                std::pow(temperature(node_num_1), 4));
    }

    [[nodiscard]] double expected_conductive_group() const {
        return expected_group_flow(
            group_1, group_2, [this](Index node_num_1, Index node_num_2) {
                return expected_conductive_pair(node_num_1, node_num_2);
            });
    }

    [[nodiscard]] double expected_radiative_group() const {
        return expected_group_flow(
            group_1, group_2, [this](Index node_num_1, Index node_num_2) {
                return expected_radiative_pair(node_num_1, node_num_2);
            });
    }

    [[nodiscard]] static double temperature(Index node_num) {
        switch (node_num) {
            case 2:
                return 305.0;
            case 11:
                return 300.0;
            case 17:
                return 290.0;
            case 40:
                return 315.0;
            case 55:
                return 280.0;
            default:
                return 0.0;
        }
    }

    [[nodiscard]] static const std::array<FlowLink, 9>& conductive_links() {
        static const std::array<FlowLink, 9> links{{
            {17, 2, 1.2},
            {17, 55, 0.8},
            {11, 2, 1.5},
            {11, 55, 2.0},
            {40, 2, 0.6},
            {40, 55, 1.1},
            {17, 11, 3.0},
            {11, 40, 0.4},
            {2, 55, 4.4},
        }};
        return links;
    }

    [[nodiscard]] static const std::array<FlowLink, 9>& radiative_links() {
        static const std::array<FlowLink, 9> links{{
            {17, 2, 0.10},
            {17, 55, 0.05},
            {11, 2, 0.07},
            {11, 55, 0.09},
            {40, 2, 0.11},
            {40, 55, 0.04},
            {17, 11, 0.08},
            {11, 40, 0.06},
            {2, 55, 0.03},
        }};
        return links;
    }

    template <typename FlowFunction>
    [[nodiscard]] static double expected_group_flow(
        const std::vector<Index>& node_nums_1,
        const std::vector<Index>& node_nums_2, FlowFunction&& flow_function) {
        double total_flow = 0.0;

        for (const Index node_num_1 : node_nums_1) {
            for (const Index node_num_2 : node_nums_2) {
                total_flow += flow_function(node_num_1, node_num_2);
            }
        }

        return total_flow;
    }

    [[nodiscard]] static double lookup_link_value(
        const std::array<FlowLink, 9>& links, Index node_num_1,
        Index node_num_2) {
        for (const auto& link : links) {
            const bool same_order =
                link.node_num_1 == node_num_1 && link.node_num_2 == node_num_2;
            const bool reversed_order =
                link.node_num_1 == node_num_2 && link.node_num_2 == node_num_1;
            if (same_order || reversed_order) {
                return link.value;
            }
        }

        return 0.0;
    }
};

}  // namespace

TEST_CASE("ThermalNetwork adds diffusive nodes", "[thermalnetwork]") {
    ThermalNetwork network;

    Node node1(1);
    Node node2(5);

    network.add_node(node1);
    network.add_node(node2);

    REQUIRE((network.nodes().num_nodes() == 2));
    REQUIRE((network.nodes().get_num_diff_nodes() == 2));

    auto& conductive = network.conductive_couplings();
    conductive.add_coupling(1, 5, 42.0);

    REQUIRE((conductive.get_coupling_value(1, 5) == Catch::Approx(42.0)));
}

TEST_CASE("ThermalNetwork handles boundary nodes", "[thermalnetwork]") {
    ThermalNetwork network;

    Node diffusive(1);
    Node boundary(10);
    boundary.set_type('B');

    network.add_node(diffusive);
    network.add_node(boundary);

    REQUIRE((network.nodes().num_nodes() == 2));
    REQUIRE((network.nodes().get_num_bound_nodes() == 1));

    auto& conductive = network.conductive_couplings();
    auto& radiative = network.radiative_couplings();

    conductive.add_coupling(1, 10, 5.5);
    radiative.add_coupling(1, 10, 7.5);

    REQUIRE((conductive.get_coupling_value(1, 10) == Catch::Approx(5.5)));
    REQUIRE((radiative.get_coupling_value(1, 10) == Catch::Approx(7.5)));
}

TEST_CASE("ThermalNetwork removes nodes and couplings", "[thermalnetwork]") {
    ThermalNetwork network;

    Node node1(1);
    Node node2(3);

    network.add_node(node1);
    network.add_node(node2);

    auto& conductive = network.conductive_couplings();
    conductive.add_coupling(1, 3, 12.0);

    REQUIRE((conductive.get_coupling_value(1, 3) == Catch::Approx(12.0)));

    network.remove_node(Index{3});

    REQUIRE((network.nodes().num_nodes() == 1));
    REQUIRE(std::isnan(conductive.get_coupling_value(1, 3)));

    Node duplicate(1);
    network.add_node(duplicate);
    REQUIRE((network.nodes().num_nodes() == 1));
}

TEST_CASE("ThermalNetwork calculates conductive flow for node groups",
          "[thermalnetwork]") {
    ThermalNetworkFlowFixture fixture;

    REQUIRE((fixture.network.flow_conductive(17, 2) ==
             Catch::Approx(fixture.expected_conductive_pair(17, 2))));
    REQUIRE((fixture.network.flow_conductive(40, 55) ==
             Catch::Approx(fixture.expected_conductive_pair(40, 55))));

    REQUIRE(
        (fixture.network.flow_conductive(fixture.group_1, fixture.group_2) ==
         Catch::Approx(fixture.expected_conductive_group())));
}

TEST_CASE("ThermalNetwork calculates radiative flow for node groups",
          "[thermalnetwork]") {
    ThermalNetworkFlowFixture fixture;

    REQUIRE((fixture.network.flow_radiative(17, 2) ==
             Catch::Approx(fixture.expected_radiative_pair(17, 2))));
    REQUIRE((fixture.network.flow_radiative(40, 55) ==
             Catch::Approx(fixture.expected_radiative_pair(40, 55))));

    REQUIRE((fixture.network.flow_radiative(fixture.group_1, fixture.group_2) ==
             Catch::Approx(fixture.expected_radiative_group())));
}
