#include <Eigen/Core>
#include <Eigen/Sparse>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

#include "pycanha-core/thermaldata/dense_time_series.hpp"
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

    auto& series = data.add_dense_time_series("conductive", 2, 3);
    REQUIRE(data.has_dense_time_series("conductive"));
    REQUIRE(data.size() == 1);

    REQUIRE(series.num_timesteps() == 2);
    REQUIRE(series.num_columns() == 3);
    REQUIRE(series.times().isZero(0.0));
    REQUIRE(series.values().isZero(0.0));

    auto replacement = DenseTimeSeries(4, 1);
    replacement.times()(0) = 1.0;
    data.add_dense_time_series("conductive", std::move(replacement));

    const auto& stored = data.get_dense_time_series("conductive");
    REQUIRE(stored.num_timesteps() == 4);
    REQUIRE(stored.num_columns() == 1);
    REQUIRE(stored.times()(0) == Catch::Approx(1.0));
}

TEST_CASE("ThermalData manages sparse series and lookup tables",
          "[thermaldata]") {
    ThermalData data;

    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix(2, 2);
    matrix.insert(0, 0) = 1.0;
    matrix.insert(1, 1) = 2.0;
    matrix.makeCompressed();

    auto& sparse_series = data.add_sparse_time_series("radiative");
    sparse_series.push_back(0.0, matrix);

    Eigen::VectorXd x(2);
    x << 0.0, 1.0;

    Eigen::VectorXd y(2);
    y << 10.0, 20.0;
    const auto& table = data.add_lookup_table("scalar", LookupTable1D(x, y));

    LookupTableVec1D::MatrixType y_vec(2, 2);
    y_vec << 1.0, 2.0, 3.0, 4.0;
    const auto& table_vec =
        data.add_lookup_table_vec("vector", LookupTableVec1D(x, y_vec));

    REQUIRE(data.has_sparse_time_series("radiative"));
    REQUIRE(data.has_lookup_table("scalar"));
    REQUIRE(data.has_lookup_table_vec("vector"));
    REQUIRE(data.size() == 3);

    REQUIRE(data.get_sparse_time_series("radiative").nnz() == 2);
    REQUIRE(table.evaluate(0.5) == Catch::Approx(15.0));
    REQUIRE(table_vec.evaluate(0.5)(1) == Catch::Approx(3.0));
}

TEST_CASE("ThermalData removes entries and throws on missing ones",
          "[thermaldata]") {
    ThermalData data;

    Eigen::VectorXd x(2);
    x << 0.0, 1.0;

    Eigen::VectorXd y(2);
    y << 1.0, 2.0;

    data.add_dense_time_series("tmp", 1, 1);
    data.add_lookup_table("steady", LookupTable1D(x, y));
    REQUIRE(data.size() == 2U);

    data.remove_dense_time_series("tmp");
    REQUIRE(!data.has_dense_time_series("tmp"));
    REQUIRE(data.size() == 1U);

    REQUIRE_THROWS_AS(data.get_dense_time_series("unknown"), std::out_of_range);
    REQUIRE_THROWS_AS(data.get_lookup_table("unknown"), std::out_of_range);

    const auto& const_data = static_cast<const ThermalData&>(data);
    REQUIRE_THROWS_AS(const_data.get_lookup_table_vec("unknown"),
                      std::out_of_range);
}

// NOLINTEND(bugprone-chained-comparison)
