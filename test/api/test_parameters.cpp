#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"

namespace {

void populate_entity_demo(pycanha::ThermalModel& tm) {
    pycanha::Node plate(1);
    plate.set_T(300.0);
    plate.set_C(500.0);

    pycanha::Node sink(2);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(250.0);

    tm.tmm().add_node(plate);
    tm.tmm().add_node(sink);
    tm.tmm().add_conductive_coupling(1, 2, 5.0);
}

}  // namespace

TEST_CASE("Parameters CRUD reads like a model setup walkthrough",
          "[api][parameters]") {
    pycanha::ThermalModel tm("parameters_demo");
    auto& parameters = tm.parameters();

    // The public boundary is case-insensitive, so engineers can use the same
    // names in formulas and setup code without worrying about exact casing.
    parameters.add_parameter("Gain", 10.0);
    parameters.add_parameter("q_in", 50.0);
    parameters.add_parameter("count", std::int64_t{3});
    parameters.add_parameter("flag", true);

    REQUIRE(parameters.contains("gain"));
    REQUIRE(parameters.contains("GAIN"));
    REQUIRE(std::get<double>(parameters.get_parameter("gAiN")) ==
            Catch::Approx(10.0));

    parameters.set_parameter("GAIN", 20.0);
    parameters.rename_parameter("gain", "conductance");
    REQUIRE_FALSE(parameters.contains("gain"));
    REQUIRE(parameters.contains("conductance"));
    REQUIRE(std::get<double>(parameters.get_parameter("CONDUCTANCE")) ==
            Catch::Approx(20.0));

    parameters.remove_parameter("FLAG");
    REQUIRE_FALSE(parameters.contains("flag"));
    REQUIRE(std::get<std::int64_t>(parameters.get_parameter("count")) == 3);

    const auto size_before_reserved_names = parameters.size();
    parameters.add_parameter("time", 1.0);
    parameters.add_parameter("QI5", 2.0);
    parameters.add_parameter("GL(1,2)", 3.0);
    REQUIRE(parameters.contains("time"));
    REQUIRE(parameters.is_internal_parameter("time"));
    REQUIRE_FALSE(parameters.contains("QI5"));
    REQUIRE_FALSE(parameters.contains("GL(1,2)"));
    REQUIRE(parameters.size() == size_before_reserved_names);
}

TEST_CASE("ThermalMathematicalModel exposes model-bound entity helpers",
          "[api][parameters]") {
    pycanha::ThermalModel tm("entity_demo");
    populate_entity_demo(tm);

    auto& tmm = tm.tmm();

    // String lookup is the shortest path from the API guide to live model
    // state, while entities() gives a more structured surface when you already
    // know which kind of quantity you want.
    auto heat = tmm.entity("QI1");
    auto heat_alt = tmm.entities().attribute("qi", 1);
    auto temperature = tmm.entities().temperature(1);
    auto conductive = tmm.entities().conductive_coupling(2, 1);

    REQUIRE(heat.string_representation() == "QI1");
    REQUIRE(heat_alt.string_representation() == "QI1");
    REQUIRE(conductive.string_representation() == "GL(1,2)");
    REQUIRE(conductive.get_value() == Catch::Approx(5.0));

    REQUIRE(heat.set_value(15.0));
    REQUIRE(temperature.set_value(280.0));
    REQUIRE(tmm.nodes().get_qi(1) == Catch::Approx(15.0));
    REQUIRE(tmm.nodes().get_T(1) == Catch::Approx(280.0));

    REQUIRE_FALSE(tmm.find_entity("not_an_entity").has_value());
    REQUIRE_FALSE(tmm.find_entity("GL(1)").has_value());
}
