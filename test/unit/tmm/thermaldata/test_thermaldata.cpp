#include <Eigen/Core>
#include <Eigen/Sparse>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"
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

TEST_CASE("ThermalData manages dense time series", "[thermaldata]") {
    ThermalData data;

    DataModel model({10, 20, 30});
    model.T().resize(2, 3);
    data.models().add_model("solver", std::move(model));
    REQUIRE(data.models().has_model("solver"));
    REQUIRE(data.size() == 1);

    const auto& series = data.models().get_model("solver").T();
    REQUIRE(series.num_timesteps() == 2);
    REQUIRE(series.num_columns() == 3);
    REQUIRE(series.times().isZero(0.0));
    REQUIRE(series.values().isZero(0.0));

    DataModel replacement({42});
    replacement.T().resize(4, 1);
    replacement.T().times()(0) = 1.0;
    data.models().add_model("solver", std::move(replacement));

    const auto& stored = data.models().get_model("solver").T();
    REQUIRE(stored.num_timesteps() == 4);
    REQUIRE(stored.num_columns() == 1);
    REQUIRE(stored.times()(0) == Catch::Approx(1.0));
}

TEST_CASE("ThermalData manages sparse series and lookup tables",
          "[thermaldata]") {
    ThermalData data;

    Eigen::VectorXd x(2);
    x << 0.0, 1.0;

    LookupTableVec1D::MatrixType y_vec(2, 2);
    y_vec << 1.0, 2.0, 3.0, 4.0;
    const auto& table_vec =
        data.tables().add_table("vector", LookupTableVec1D(x, y_vec));

    DataModel model({1, 2});
    data.models().add_model("solver", std::move(model));

    REQUIRE(data.tables().has_table("vector"));
    REQUIRE(data.models().has_model("solver"));
    REQUIRE(data.size() == 2);

    REQUIRE(table_vec.evaluate(0.5)(1) == Catch::Approx(3.0));
}

TEST_CASE("ThermalData removes entries and throws on missing ones",
          "[thermaldata]") {
    ThermalData data;

    Eigen::VectorXd x(2);
    x << 0.0, 1.0;

    LookupTableVec1D::MatrixType y(2, 1);
    y << 1.0, 2.0;

    DataModel model({1});
    model.T().resize(1, 1);
    data.models().add_model("tmp", std::move(model));
    data.tables().add_table("steady", LookupTableVec1D(x, y));
    REQUIRE(data.size() == 2U);

    data.models().remove_model("tmp");
    REQUIRE(!data.models().has_model("tmp"));
    REQUIRE(data.size() == 1U);

    REQUIRE_THROWS_AS(data.models().get_model("unknown"), std::out_of_range);
    REQUIRE_THROWS_AS(data.tables().get_table("unknown"), std::out_of_range);

    const auto& const_data = static_cast<const ThermalData&>(data);
    REQUIRE_THROWS_AS(const_data.tables().get_table("unknown"),
                      std::out_of_range);
}

// NOLINTEND(bugprone-chained-comparison)
