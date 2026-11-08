#include <Eigen/Core>
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pycanha-core/parameters/origin.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/parameters/table.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

// y = 10 * x on [1, 3], sampled at 1, 2, 3.
LookupTable1D linear_table(
    ExtrapolationMethod extrap = ExtrapolationMethod::Constant,
    InterpolationMethod interp = InterpolationMethod::Linear) {
    return {Eigen::Vector3d(1.0, 2.0, 3.0), Eigen::Vector3d(10.0, 20.0, 30.0),
            interp, extrap};
}

double published(const Parameters& parameters, const std::string& name) {
    return std::get<double>(parameters.get_parameter(name));
}

// Whether add_table refuses `name` as an argument error.
bool refuses_table_name(Parameters& parameters, const std::string& name) {
    try {
        parameters.add_table(name, linear_table());
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

TEST_CASE("Table values follow the interpolation and extrapolation",
          "[tables]") {
    const Table held("held", linear_table(), Origin::user, "");
    REQUIRE(held.evaluate(1.5) == Catch::Approx(15.0));
    REQUIRE(held.evaluate(3.0) == Catch::Approx(30.0));
    REQUIRE(held.evaluate(0.0) == Catch::Approx(10.0));
    REQUIRE(held.evaluate(5.0) == Catch::Approx(30.0));

    const Table extrapolated("extrapolated",
                             linear_table(ExtrapolationMethod::Linear),
                             Origin::user, "");
    REQUIRE(extrapolated.evaluate(5.0) == Catch::Approx(50.0));
    REQUIRE(extrapolated.evaluate(0.0) == Catch::Approx(0.0));

    const Table strict("strict", linear_table(ExtrapolationMethod::Throw),
                       Origin::user, "");
    REQUIRE_THROWS_AS(strict.evaluate(3.5), std::out_of_range);
    REQUIRE(std::isnan(held.evaluate(std::nan(""))));
}

TEST_CASE("Stepped tables hold values between breakpoints", "[tables]") {
    const Table lower(
        "lower",
        linear_table(ExtrapolationMethod::Constant, InterpolationMethod::Step),
        Origin::user, "");
    REQUIRE(lower.evaluate(1.9) == Catch::Approx(10.0));
    REQUIRE(lower.evaluate(2.0) == Catch::Approx(20.0));
    REQUIRE(lower.slope(1.5) == 0.0);

    const Table upper("upper",
                      linear_table(ExtrapolationMethod::Constant,
                                   InterpolationMethod::NearestUpper),
                      Origin::user, "");
    REQUIRE(upper.evaluate(1.1) == Catch::Approx(20.0));
    REQUIRE(upper.evaluate(1.0) == Catch::Approx(10.0));
}

TEST_CASE("Table slopes are the segment slopes", "[tables]") {
    const LookupTable1D data(Eigen::Vector3d(0.0, 1.0, 3.0),
                             Eigen::Vector3d(0.0, 2.0, 3.0));
    const Table table("k", data, Origin::user, "");
    REQUIRE(table.slope(0.5) == Catch::Approx(2.0));
    // A breakpoint takes the slope of the segment above it.
    REQUIRE(table.slope(1.0) == Catch::Approx(0.5));
    REQUIRE(table.slope(3.0) == Catch::Approx(0.5));
    REQUIRE(table.slope(4.0) == 0.0);
}

TEST_CASE("Table names share the parameters' name space", "[tables]") {
    Parameters parameters;
    parameters.add_parameter("gain", 1.0);
    const std::vector<std::string> refused{"T",  "gl",  "Qi",  "time", "sin",
                                           "pi", "max", "T10", "gain", "2k"};
    REQUIRE(std::ranges::all_of(refused, [&](const std::string& name) {
        return refuses_table_name(parameters, name);
    }));

    parameters.add_table("cp_al", linear_table());
    REQUIRE(parameters.has_table("CP_AL"));
    REQUIRE_THROWS_AS(parameters.add_table("Cp_Al", linear_table()),
                      std::invalid_argument);
    parameters.add_parameter("cp_al", 2.0);
    REQUIRE_FALSE(parameters.contains("cp_al"));
    REQUIRE(parameters.table_names() == std::vector<std::string>{"cp_al"});
}

TEST_CASE("Adding and removing tables changes the structure version",
          "[tables]") {
    Parameters parameters;
    const std::uint64_t before = parameters.get_structure_version();
    parameters.add_table("k", linear_table());
    const std::uint64_t added = parameters.get_structure_version();
    REQUIRE(added != before);

    REQUIRE(parameters.remove_table("k"));
    REQUIRE(parameters.get_structure_version() != added);
    REQUIRE_FALSE(parameters.remove_table("k"));
    REQUIRE_THROWS_AS(static_cast<void>(parameters.table("k")),
                      std::out_of_range);
}

TEST_CASE("Table values are edited in place", "[tables]") {
    Parameters parameters;
    parameters.add_table("k", linear_table());
    const double* data = parameters.table("k").lookup_table().y().data();
    const std::uint64_t version = parameters.get_structure_version();

    const std::vector<double> values{1.0, 2.0, 4.0};
    parameters.set_table_values("k", values);
    REQUIRE(parameters.table("k").evaluate(2.5) == Catch::Approx(3.0));
    REQUIRE(parameters.table("k").lookup_table().y().data() == data);
    REQUIRE(parameters.get_structure_version() == version);

    const std::vector<double> x{0.0, 1.0, 2.0};
    parameters.set_table_x("k", x);
    REQUIRE(parameters.table("k").evaluate(1.5) == Catch::Approx(3.0));

    const std::vector<double> short_values{1.0, 2.0};
    REQUIRE_THROWS_AS(parameters.set_table_values("k", short_values),
                      std::invalid_argument);
    const std::vector<double> unsorted{0.0, 2.0, 1.0};
    REQUIRE_THROWS_AS(parameters.set_table_x("k", unsorted),
                      std::invalid_argument);
    REQUIRE(parameters.table("k").lookup_table().x()(2) == Catch::Approx(2.0));
}

TEST_CASE("Generated tables never replace a user table", "[tables]") {
    Parameters parameters;
    parameters.add_table("k_user", linear_table());
    REQUIRE(parameters.add_generated_table(Origin::conduction_builder, "k_user",
                                           linear_table()) ==
            TableGeneration::kept_user);
    REQUIRE(parameters.table("k_user").origin() == Origin::user);

    REQUIRE(parameters.add_generated_table(Origin::conduction_builder, "k_gen",
                                           linear_table()) ==
            TableGeneration::added);
    REQUIRE(parameters.table("k_gen").origin() == Origin::conduction_builder);

    const std::uint64_t version = parameters.get_structure_version();
    const LookupTable1D wider(Eigen::Vector2d(0.0, 10.0),
                              Eigen::Vector2d(0.0, 1.0));
    REQUIRE(
        parameters.add_generated_table(Origin::etms_reader, "k_gen", wider) ==
        TableGeneration::overwritten);
    REQUIRE(parameters.table("k_gen").origin() == Origin::etms_reader);
    REQUIRE(parameters.table("k_gen").evaluate(5.0) == Catch::Approx(0.5));
    REQUIRE(parameters.get_structure_version() != version);

    REQUIRE_THROWS_AS(
        parameters.add_generated_table(Origin::user, "k_new", linear_table()),
        std::invalid_argument);
}

TEST_CASE("Tables cannot change structure while it is locked", "[tables]") {
    Parameters parameters;
    parameters.add_table("k", linear_table());
    parameters.lock_structure();
    REQUIRE_THROWS_AS(parameters.add_table("k2", linear_table()),
                      std::runtime_error);
    REQUIRE_THROWS_AS(parameters.remove_table("k"), std::runtime_error);

    const std::vector<double> values{1.0, 2.0, 3.0};
    parameters.set_table_values("k", values);
    REQUIRE(parameters.table("k").evaluate(2.0) == Catch::Approx(2.0));
    parameters.unlock_structure();
}

TEST_CASE("A table evaluated at a parameter publishes its value", "[tables]") {
    Parameters parameters;
    parameters.add_internal_parameter("time", 1.5);
    parameters.add_table("heater", linear_table(), "time");
    REQUIRE(published(parameters, "heater") == Catch::Approx(15.0));
    REQUIRE(parameters.is_internal_parameter("heater"));

    parameters.set_internal_parameter("time", 2.5);
    parameters.update_evaluated_tables();
    REQUIRE(published(parameters, "heater") == Catch::Approx(25.0));

    // The published value goes with the table.
    parameters.remove_internal_parameter("heater");
    REQUIRE(parameters.contains("heater"));
    REQUIRE(parameters.remove_table("heater"));
    REQUIRE_FALSE(parameters.contains("heater"));

    REQUIRE_THROWS_AS(parameters.add_table("bad", linear_table(), "missing"),
                      std::invalid_argument);
}

TEST_CASE("Out-of-range queries are recorded per table", "[tables]") {
    Parameters parameters;
    parameters.add_internal_parameter("time", 0.0);
    parameters.add_table("held", linear_table());
    parameters.add_table("extrapolated",
                         linear_table(ExtrapolationMethod::Linear));
    const double* time = parameters.time_ptr();
    REQUIRE(time != nullptr);

    const auto held = parameters.table("held").view(time);
    const auto extrapolated = parameters.table("extrapolated").view(time);
    REQUIRE(detail::table_value(held, 2.0) == Catch::Approx(20.0));
    REQUIRE(parameters.table_ranges().empty());

    parameters.set_internal_parameter("time", 60.0);
    static_cast<void>(detail::table_value(held, 4.0));
    parameters.set_internal_parameter("time", 120.0);
    static_cast<void>(detail::table_value(held, 0.5));
    static_cast<void>(detail::table_value(held, 7.0));
    static_cast<void>(detail::table_value(extrapolated, 9.0));

    const auto ranges = parameters.table_ranges();
    REQUIRE(ranges.size() == 1U);
    REQUIRE(ranges.front().table == "held");
    REQUIRE(ranges.front().first_time == Catch::Approx(60.0));
    REQUIRE(ranges.front().first_argument == Catch::Approx(4.0));
    REQUIRE(ranges.front().min_argument == Catch::Approx(0.5));
    REQUIRE(ranges.front().max_argument == Catch::Approx(7.0));
    REQUIRE(ranges.front().x_max == Catch::Approx(3.0));

    parameters.reset_table_ranges();
    REQUIRE(parameters.table_ranges().empty());
}
