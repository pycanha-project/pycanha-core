#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "pycanha-core/thermaldata/lookup_table.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

// Whether `write` throws std::invalid_argument.
template <typename Function>
bool is_refused(const Function& write) {
    try {
        write();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void periodic(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    const LookupTable1D table(x, y, InterpolationMethod::Linear,
                              ExtrapolationMethod::Periodic);
    REQUIRE(table.evaluate(x(x.size() - 1)) == y(0));
}

}  // namespace

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

TEST_CASE("LookupTable1D repeats a periodic table", "[thermaldata][lookup]") {
    // Period 4, from x = 1 to x = 5.
    const LookupTable1D table(
        Eigen::Vector3d(1.0, 2.0, 5.0), Eigen::Vector3d(10.0, 40.0, 10.0),
        InterpolationMethod::Linear, ExtrapolationMethod::Periodic);
    REQUIRE(table.evaluate(1.5) == Catch::Approx(25.0));
    REQUIRE(table.evaluate(5.0) == 10.0);
    REQUIRE(table.evaluate(6.0) == 40.0);
    REQUIRE(table.evaluate(5.5 + 12.0) == Catch::Approx(25.0));
    REQUIRE(table.evaluate(0.0) == Catch::Approx(20.0));
    REQUIRE(table.evaluate(-3.0) == 10.0);
    REQUIRE(table.evaluate(-2.5 - 8.0) == Catch::Approx(25.0));

    constexpr double infinity = std::numeric_limits<double>::infinity();
    REQUIRE(std::isnan(table.evaluate(infinity)));
    REQUIRE(std::isnan(table.evaluate(-infinity)));
    REQUIRE(std::isnan(table.evaluate(std::nan(""))));
}

TEST_CASE("LookupTable1D is not built periodic from data that does not close",
          "[thermaldata][lookup]") {
    const Eigen::Vector3d x(0.0, 1.0, 2.0);
    const Eigen::VectorXd single = Eigen::VectorXd::Constant(1, 1.0);
    REQUIRE(is_refused([&] { periodic(x, Eigen::Vector3d(1.0, 2.0, 3.0)); }));
    REQUIRE_FALSE(
        is_refused([&] { periodic(x, Eigen::Vector3d(1.0, 2.0, 1.0)); }));
    // The check is exact: one unit in the last place is refused.
    const Eigen::Vector3d almost(1.0, 2.0, std::nextafter(1.0, 2.0));
    REQUIRE(is_refused([&] { periodic(x, almost); }));
    // A single point has no period.
    REQUIRE(is_refused([&] { periodic(single, single); }));
}

TEST_CASE("LookupTable1D stays closed while it is periodic",
          "[thermaldata][lookup]") {
    const Eigen::Vector3d x(0.0, 1.0, 2.0);
    const Eigen::Vector3d open(1.0, 2.0, 3.0);
    const Eigen::Vector3d closed(1.0, 2.0, 1.0);
    LookupTable1D table(x, open);
    REQUIRE(is_refused([&] {
        table.set_extrapolation_method(ExtrapolationMethod::Periodic);
    }));
    REQUIRE(table.extrapolation_method() == ExtrapolationMethod::Constant);

    table.set_data(x, closed);
    table.set_extrapolation_method(ExtrapolationMethod::Periodic);
    const std::vector<double> open_values{1.0, 2.0, 3.0};
    REQUIRE(is_refused([&] { table.assign_y(open_values); }));
    REQUIRE(is_refused([&] { table.set_data(x, open); }));
    REQUIRE((table.y() == closed));

    const std::vector<double> closed_values{5.0, 2.0, 5.0};
    table.assign_y(closed_values);
    REQUIRE(table.evaluate(2.0) == 5.0);

    // Without data there is nothing to check yet.
    LookupTable1D empty;
    empty.set_extrapolation_method(ExtrapolationMethod::Periodic);
    REQUIRE(is_refused([&] { empty.set_data(x, open); }));
}

TEST_CASE("LookupTableVec1D repeats a periodic table",
          "[thermaldata][lookup]") {
    const Eigen::Vector3d x(0.0, 1.0, 4.0);
    LookupTableVec1D::MatrixType y(3, 2);
    y << 0.0, 10.0, 6.0, 40.0, 0.0, 10.0;
    LookupTableVec1D table(x, y, InterpolationMethod::Linear,
                           ExtrapolationMethod::Periodic);
    const Eigen::VectorXd value = table.evaluate(-3.5);
    REQUIRE(value(0) == Catch::Approx(3.0));
    REQUIRE(value(1) == Catch::Approx(25.0));
    REQUIRE((table.evaluate(4.0) == table.evaluate(0.0)));
    REQUIRE(table.evaluate(std::numeric_limits<double>::infinity())
                .array()
                .isNaN()
                .all());

    // Every column has to close.
    y(2, 1) = 11.0;
    REQUIRE_THROWS_AS(table.set_data(x, y), std::invalid_argument);
    REQUIRE(table.y()(2, 1) == 10.0);
    LookupTableVec1D open(x, y);
    REQUIRE_THROWS_AS(
        open.set_extrapolation_method(ExtrapolationMethod::Periodic),
        std::invalid_argument);
}
