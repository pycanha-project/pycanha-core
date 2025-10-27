#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("ThermalMathematicalModel composes a ThermalNetwork",
          "[thermalmathematicalmodel]") {
    ThermalMathematicalModel model("test-model");

    auto network = model.network_ptr();
    REQUIRE(static_cast<bool>(network));

    auto nodes_storage = model.nodes_ptr();
    REQUIRE(static_cast<bool>(nodes_storage));

    const bool shares_nodes_storage = nodes_storage == network->nodes_ptr();
    REQUIRE(shares_nodes_storage);

    const bool formulas_associated = model.formulas.network() == network;
    REQUIRE(formulas_associated);

    const bool thermal_data_associated =
        model.thermal_data.network_ptr() == network;
    REQUIRE(thermal_data_associated);

    Node diffusive(1);
    diffusive.set_T(275.0);
    model.add_node(diffusive);

    model.add_node(Index{2});  // Default diffusive node

    Node boundary(3);
    boundary.set_type('B');
    model.add_node(boundary);

    const auto& nodes = model.nodes();
    const bool has_two_diff_nodes = nodes.get_num_diff_nodes() == 2;
    REQUIRE(has_two_diff_nodes);

    const bool has_one_bound_node = nodes.get_num_bound_nodes() == 1;
    REQUIRE(has_one_bound_node);

    model.add_conductive_coupling(Index{1}, Index{2}, 10.0);
    const auto conductive_value =
        model.conductive_couplings().get_coupling_value(1, 2);
    REQUIRE_THAT(conductive_value, Catch::Matchers::WithinAbs(10.0, 1e-12));

    const Coupling coupling(Index{2}, Index{3}, 5.0);
    model.add_radiative_coupling(coupling);
    const auto radiative_value =
        model.radiative_couplings().get_coupling_value(2, 3);
    REQUIRE_THAT(radiative_value, Catch::Matchers::WithinAbs(5.0, 1e-12));
}
