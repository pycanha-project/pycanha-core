#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace {

void populate_network(pycanha::ThermalModel& tm, pycanha::Node& plate) {
    auto& network = tm.tmm().network();

    plate.set_T(300.0);
    plate.set_C(500.0);
    plate.set_qi(10.0);
    plate.set_eps(0.9);
    plate.set_aph(0.85);
    plate.set_a(0.85);
    plate.set_fx(0.0);
    plate.set_fy(0.0);
    plate.set_fz(1.0);

    pycanha::Node inner(2);
    inner.set_T(280.0);
    inner.set_C(200.0);

    pycanha::Node sink(100);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(250.0);

    // The lower-level network entry point is still useful when you want to
    // demonstrate node association explicitly.
    network.add_node(plate);
    network.add_node(sink);
    network.add_node(inner);

    tm.tmm().add_conductive_coupling(10, 100, 5.0);
    tm.tmm().add_conductive_coupling(2, 100, 2.0);
    tm.tmm().add_radiative_coupling(10, 100, 0.01);
}

}  // namespace

TEST_CASE(
    "ThermalModel teaches node ownership and ordering through its network",
    "[api][network]") {
    pycanha::ThermalModel tm("network_demo");
    pycanha::Node plate(10);
    populate_network(tm, plate);

    auto& tmm = tm.tmm();
    auto& network = tmm.network();
    auto& nodes = tmm.nodes();

    SECTION(
        "Nodes added through the network become associated with model "
        "storage") {
        // Once a node is associated, reading or writing through either handle
        // should tell the same story.
        REQUIRE(plate.get_T() == Catch::Approx(300.0));
        REQUIRE(nodes.set_T(10, 310.0));
        REQUIRE(plate.get_T() == Catch::Approx(310.0));
        REQUIRE(plate.get_qi() == Catch::Approx(10.0));

        REQUIRE(nodes.num_nodes() == 3);
        REQUIRE(nodes.get_num_diff_nodes() == 2);
        REQUIRE(nodes.get_num_bound_nodes() == 1);

        const auto idx_2 = nodes.get_idx_from_node_num(2);
        const auto idx_10 = nodes.get_idx_from_node_num(10);
        const auto idx_100 = nodes.get_idx_from_node_num(100);
        REQUIRE(idx_2.has_value());
        REQUIRE(idx_10.has_value());
        REQUIRE(idx_100.has_value());
        REQUIRE(*idx_2 == 0);
        REQUIRE(*idx_10 == 1);
        REQUIRE(*idx_100 == 2);
        REQUIRE(nodes.get_node_num_from_idx(*idx_2) == 2);
        REQUIRE(nodes.get_node_num_from_idx(*idx_100) == 100);
        REQUIRE_FALSE(nodes.get_idx_from_node_num(999).has_value());

        pycanha::Node duplicate_plate(10);
        network.add_node(duplicate_plate);
        REQUIRE(nodes.num_nodes() == 3);
    }

    SECTION(
        "Couplings are symmetric and flow helpers work from both tmm and "
        "network") {
        // The public API is model-bound, but the lower-level network view is
        // still the same physical model.
        REQUIRE(tmm.conductive_couplings().get_coupling_value(10, 100) ==
                Catch::Approx(5.0));
        REQUIRE(tmm.conductive_couplings().get_coupling_value(100, 10) ==
                Catch::Approx(5.0));
        REQUIRE(tmm.radiative_couplings().get_coupling_value(100, 10) ==
                Catch::Approx(0.01));

        REQUIRE(tmm.conductive_couplings().get_coupling_value(10, 2) ==
                Catch::Approx(0.0));
        REQUIRE(tmm.flow_conductive(10, 2) == Catch::Approx(0.0));

        const double expected_pair = 5.0 * (250.0 - 300.0);
        REQUIRE(network.flow_conductive(10, 100) ==
                Catch::Approx(expected_pair));
        REQUIRE(tmm.flow_conductive(10, 100) == Catch::Approx(expected_pair));

        const double expected_radiative =
            0.01 * pycanha::STF_BOLTZ *
            (std::pow(250.0, 4) - std::pow(300.0, 4));
        REQUIRE(network.flow_radiative(10, 100) ==
                Catch::Approx(expected_radiative));
        REQUIRE(tmm.flow_radiative(10, 100) ==
                Catch::Approx(expected_radiative));

        const std::vector<pycanha::Index> hot_side{10, 2};
        const std::vector<pycanha::Index> cold_side{100};
        const double expected_group = expected_pair + 2.0 * (250.0 - 280.0);
        REQUIRE(tmm.flow_conductive(hot_side, cold_side) ==
                Catch::Approx(expected_group));
    }
}