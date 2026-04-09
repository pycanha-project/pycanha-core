#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "pycanha-core/thermaldata/lookup_table.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("LookupTable1D interpolates and extrapolates",
          "[thermaldata][lookup]") {
    Eigen::VectorXd x(3);
    x << 0.0, 10.0, 20.0;
    Eigen::VectorXd y(3);
    y << 0.0, 100.0, 200.0;

    LookupTable1D table(x, y);
    REQUIRE(table.evaluate(5.0) == Catch::Approx(50.0));
    REQUIRE(table.evaluate(20.0) == Catch::Approx(200.0));
    REQUIRE(table.evaluate(-5.0) == Catch::Approx(0.0));

    table.set_extrapolation_method(ExtrapolationMethod::Linear);
    REQUIRE(table.evaluate(-5.0) == Catch::Approx(-50.0));

    table.set_interpolation_method(InterpolationMethod::NearestUpper);
    REQUIRE(table.evaluate(12.0) == Catch::Approx(200.0));

    const Eigen::Vector2d query(5.0, 15.0);
    const Eigen::VectorXd values = table.evaluate(query);
    REQUIRE(values(0) == Catch::Approx(100.0));
    REQUIRE(values(1) == Catch::Approx(200.0));
}

TEST_CASE("LookupTable1D validates input and throw extrapolation",
          "[thermaldata][lookup]") {
    Eigen::VectorXd x(3);
    x << 0.0, 2.0, 1.0;
    Eigen::VectorXd y(3);
    y << 0.0, 2.0, 1.0;

    REQUIRE_THROWS_AS(LookupTable1D(x, y), std::invalid_argument);

    LookupTable1D table(Eigen::Vector2d(0.0, 1.0), Eigen::Vector2d(1.0, 3.0));
    table.set_extrapolation_method(ExtrapolationMethod::Throw);
    REQUIRE_THROWS_AS(table.evaluate(2.0), std::out_of_range);
}

TEST_CASE("LookupTableVec1D evaluates scalar and batch queries",
          "[thermaldata][lookup]") {
    Eigen::VectorXd x(2);
    x << 0.0, 10.0;

    LookupTableVec1D::MatrixType y(2, 2);
    y << 0.0, 10.0, 20.0, 30.0;

    const LookupTableVec1D table(x, y);
    const Eigen::VectorXd value = table.evaluate(5.0);
    REQUIRE(value(0) == Catch::Approx(10.0));
    REQUIRE(value(1) == Catch::Approx(20.0));

    Eigen::VectorXd queries(2);
    queries << 0.0, 10.0;
    const auto batch = table.evaluate(queries);
    REQUIRE(batch.rows() == 2);
    REQUIRE(batch.cols() == 2);
    REQUIRE(batch(1, 1) == Catch::Approx(30.0));
}