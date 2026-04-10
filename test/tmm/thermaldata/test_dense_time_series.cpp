#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/thermaldata/dense_time_series.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("DenseTimeSeries stores rows and interpolates",
          "[thermaldata][dense]") {
    DenseTimeSeries series(2, 2);

    const Eigen::Vector2d row0(0.0, 10.0);
    const Eigen::Vector2d row1(10.0, 20.0);
    series.set_row(0, 0.0, row0);
    series.set_row(1, 1.0, row1);

    REQUIRE(series.times()(1) == Catch::Approx(1.0));
    REQUIRE(series.values()(0, 1) == Catch::Approx(10.0));

    const Eigen::VectorXd mid = series.interpolate(0.5);
    REQUIRE(mid(0) == Catch::Approx(5.0));
    REQUIRE(mid(1) == Catch::Approx(15.0));

    const Eigen::VectorXd extrapolated = series.interpolate(-0.5);
    REQUIRE(extrapolated(0) == Catch::Approx(-5.0));
    REQUIRE(extrapolated(1) == Catch::Approx(5.0));
}

TEST_CASE("DenseTimeSeries resizes resets and handles single row",
          "[thermaldata][dense]") {
    DenseTimeSeries series(1, 3);
    const Eigen::Vector3d row(1.0, 2.0, 3.0);
    series.set_row(0, 4.0, row);

    const Eigen::VectorXd same = series.interpolate(10.0);
    REQUIRE(same(2) == Catch::Approx(3.0));

    series.reset();
    REQUIRE(series.times().isZero(0.0));
    REQUIRE(series.values().isZero(0.0));

    series.resize(2, 1);
    REQUIRE(series.num_timesteps() == 2);
    REQUIRE(series.num_columns() == 1);
}
