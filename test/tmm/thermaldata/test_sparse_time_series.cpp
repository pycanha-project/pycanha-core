#include <Eigen/Sparse>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <utility>

#include "pycanha-core/thermaldata/sparse_time_series.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

SparseTimeSeries::SparseMatrixType make_diagonal(double a, double b) {
    SparseTimeSeries::SparseMatrixType matrix(2, 2);
    matrix.insert(0, 0) = a;
    matrix.insert(1, 1) = b;
    matrix.makeCompressed();
    return matrix;
}

}  // namespace

TEST_CASE("SparseTimeSeries validates patterns and interpolates",
          "[thermaldata][sparse]") {
    SparseTimeSeries series;
    series.push_back(0.0, make_diagonal(1.0, 2.0));
    series.push_back(1.0, make_diagonal(3.0, 4.0));

    REQUIRE(series.num_timesteps() == 2);
    REQUIRE(series.nnz() == 2);

    const auto interpolated = series.interpolate(0.5);
    REQUIRE(interpolated.coeff(0, 0) == Catch::Approx(2.0));
    REQUIRE(interpolated.coeff(1, 1) == Catch::Approx(3.0));

    SparseTimeSeries::SparseMatrixType mismatched(2, 2);
    mismatched.insert(0, 1) = 1.0;
    mismatched.makeCompressed();
    REQUIRE_THROWS_AS(series.push_back(2.0, std::move(mismatched)),
                      std::invalid_argument);
}

TEST_CASE("SparseTimeSeries handles single-matrix interpolation",
          "[thermaldata][sparse]") {
    SparseTimeSeries series;
    series.push_back(2.0, make_diagonal(5.0, 6.0));

    const auto interpolated = series.interpolate(10.0);
    REQUIRE(interpolated.coeff(0, 0) == Catch::Approx(5.0));
    REQUIRE(interpolated.coeff(1, 1) == Catch::Approx(6.0));
}
