#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>

#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

// NOLINTBEGIN(bugprone-chained-comparison)

TEST_CASE("ThermalData associates a thermal network", "[thermaldata]") {
    auto network = std::make_shared<ThermalNetwork>();

    ThermalData data;
    REQUIRE(data.network_ptr() == nullptr);

    data.associate(network);
    REQUIRE(data.network_ptr() == network);
    REQUIRE(data.network() == network.get());

    const auto& const_data = static_cast<const ThermalData&>(data);
    REQUIRE(const_data.network() == network.get());
}

TEST_CASE("ThermalData creates and retrieves tables", "[thermaldata]") {
    ThermalData data;

    data.create_new_table("conductive", 2, 3);
    REQUIRE(data.has_table("conductive"));
    REQUIRE(data.size() == 1);

    auto& table = data.get_table("conductive");
    REQUIRE(table.rows() == 2);
    REQUIRE(table.cols() == 3);
    REQUIRE(table.isZero(0.0));

    table(0, 1) = 4.2;
    data.create_new_table("conductive", 4, 4);
    REQUIRE(table(0, 1) == Catch::Approx(4.2));
    REQUIRE(table.rows() == 2);
    REQUIRE(table.cols() == 3);
}

TEST_CASE("ThermalData can reset and resize tables", "[thermaldata]") {
    ThermalData data;

    data.create_new_table("radiative", 1, 2);
    auto& table = data.get_table("radiative");
    table(0, 1) = 1.5;

    data.create_reset_table("radiative", 1, 2);
    REQUIRE(table.isZero(0.0));
    REQUIRE(table.rows() == 1);
    REQUIRE(table.cols() == 2);

    data.create_reset_table("radiative", 3, 1);
    auto& resized = data.get_table("radiative");
    REQUIRE(resized.rows() == 3);
    REQUIRE(resized.cols() == 1);
    REQUIRE(resized.isZero(0.0));
}

TEST_CASE("ThermalData removes tables and throws on missing ones",
          "[thermaldata]") {
    ThermalData data;

    data.create_new_table("tmp", 1, 1);
    data.create_new_table("steady", 1, 1);
    REQUIRE(data.size() == 2);

    data.remove_table("tmp");
    REQUIRE_FALSE(data.has_table("tmp"));
    REQUIRE(data.size() == 1);

    REQUIRE_THROWS_AS(data.get_table("unknown"), std::out_of_range);

    const auto& const_data = static_cast<const ThermalData&>(data);
    REQUIRE_THROWS_AS(const_data.get_table("unknown"), std::out_of_range);
}

// NOLINTEND(bugprone-chained-comparison)
