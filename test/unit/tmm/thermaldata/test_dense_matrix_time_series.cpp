#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "pycanha-core/thermaldata/dense_matrix_time_series.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("DenseMatrixTimeSeries stores matrices and interpolates",
          "[thermaldata]") {
    DenseMatrixTimeSeries series(2, 2);

    DenseMatrixTimeSeries::MatrixType first(2, 2);
    first << 1.0, 2.0, 3.0, 4.0;
    DenseMatrixTimeSeries::MatrixType second(2, 2);
    second << 5.0, 6.0, 7.0, 8.0;

    series.push_back(0.0, first);
    series.push_back(10.0, second);

    const auto midpoint = series.interpolate(5.0);
    REQUIRE(series.num_timesteps() == 2);
    REQUIRE(series.rows() == 2);
    REQUIRE(series.cols() == 2);
    REQUIRE(midpoint(0, 0) == Catch::Approx(3.0));
    REQUIRE(midpoint(1, 1) == Catch::Approx(6.0));
}

TEST_CASE("DenseMatrixTimeSeries validates dimensions and time ordering",
          "[thermaldata]") {
    DenseMatrixTimeSeries series(2, 2);

    const DenseMatrixTimeSeries::MatrixType first =
        DenseMatrixTimeSeries::MatrixType::Identity(2, 2);
    const DenseMatrixTimeSeries::MatrixType wrong =
        DenseMatrixTimeSeries::MatrixType::Identity(3, 2);

    series.push_back(1.0, first);

    REQUIRE_THROWS_AS(series.push_back(1.0, first), std::invalid_argument);
    REQUIRE_THROWS_AS(series.push_back(2.0, wrong), std::invalid_argument);
}
