#include <Eigen/Sparse>
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <stdexcept>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/dense_matrix_time_series.hpp"
#include "pycanha-core/thermaldata/sparse_time_series.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

SparseTimeSeries::SparseMatrixType make_sparse_matrix(double g12, double g13,
                                                      double g23) {
    SparseTimeSeries::SparseMatrixType matrix(3, 3);
    matrix.insert(0, 1) = g12;
    matrix.insert(1, 0) = g12;
    matrix.insert(0, 2) = g13;
    matrix.insert(2, 0) = g13;
    matrix.insert(1, 2) = g23;
    matrix.insert(2, 1) = g23;
    matrix.makeCompressed();
    return matrix;
}

DataModel make_flow_model() {
    DataModel model({10, 20, 30});
    model.T().resize(4, 3);
    model.T().set_row(0, 5.0, Eigen::Vector3d(300.0, 310.0, 320.0));
    model.T().set_row(1, 6.0, Eigen::Vector3d(301.0, 311.0, 321.0));
    model.T().set_row(2, 7.0, Eigen::Vector3d(302.0, 312.0, 322.0));
    model.T().set_row(3, 8.0, Eigen::Vector3d(303.0, 313.0, 323.0));

    model.conductive_couplings().push_back(5.0,
                                           make_sparse_matrix(2.0, 1.0, 0.5));
    model.conductive_couplings().push_back(6.0,
                                           make_sparse_matrix(3.0, 1.0, 0.5));
    model.conductive_couplings().push_back(7.0,
                                           make_sparse_matrix(4.0, 1.0, 0.5));
    model.conductive_couplings().push_back(8.0,
                                           make_sparse_matrix(5.0, 1.0, 0.5));

    model.radiative_couplings().push_back(5.0,
                                          make_sparse_matrix(0.0, 0.5, 1.5));
    model.radiative_couplings().push_back(6.0,
                                          make_sparse_matrix(0.0, 0.5, 2.5));
    model.radiative_couplings().push_back(7.0,
                                          make_sparse_matrix(0.0, 0.5, 3.5));
    model.radiative_couplings().push_back(8.0,
                                          make_sparse_matrix(0.0, 0.5, 4.5));

    return model;
}

DataModel make_mismatched_conductive_model() {
    DataModel model({10, 20});
    model.T().resize(2, 2);
    model.T().set_row(0, 0.0, Eigen::Vector2d(300.0, 310.0));
    model.T().set_row(1, 1.0, Eigen::Vector2d(301.0, 311.0));

    SparseTimeSeries::SparseMatrixType matrix(2, 2);
    matrix.insert(0, 1) = 1.0;
    matrix.insert(1, 0) = 1.0;
    matrix.makeCompressed();

    model.conductive_couplings().push_back(0.0, matrix);
    model.conductive_couplings().push_back(2.0, matrix);
    return model;
}

}  // namespace

TEST_CASE("DataModel tracks populated attributes and node numbers",
          "[thermaldata]") {
    DataModel model({10, 20});

    model.T().resize(2, 2);
    model.T().set_row(0, 0.0, Eigen::Vector2d(1.0, 2.0));
    model.T().set_row(1, 1.0, Eigen::Vector2d(3.0, 4.0));

    Eigen::SparseMatrix<double, Eigen::RowMajor> coupling(2, 2);
    coupling.insert(0, 1) = 5.0;
    coupling.makeCompressed();
    model.conductive_couplings().push_back(0.0, coupling);

    DenseMatrixTimeSeries::MatrixType jacobian(2, 1);
    jacobian << 6.0, 7.0;
    model.jacobian().push_back(0.0, jacobian);

    const auto attributes = model.populated_attributes();
    REQUIRE(model.node_numbers().size() == 2);
    REQUIRE(model.node_numbers().at(0) == 10);
    REQUIRE(std::find(attributes.begin(), attributes.end(),
                      DataModelAttribute::T) != attributes.end());
    REQUIRE(std::find(attributes.begin(), attributes.end(),
                      DataModelAttribute::KL) != attributes.end());
    REQUIRE(std::find(attributes.begin(), attributes.end(),
                      DataModelAttribute::JAC) != attributes.end());
}

TEST_CASE("DataModel rejects mismatched generic attribute access",
          "[thermaldata]") {
    DataModel model;

    REQUIRE_THROWS_AS(model.get_sparse_attribute(DataModelAttribute::T),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(model.get_matrix_attribute(DataModelAttribute::KL),
                      std::invalid_argument);
}

TEST_CASE("DataModel conductive flow uses the first stored sample by default",
          "[thermaldata]") {
    const DataModel model = make_flow_model();

    const Eigen::MatrixXd flow = model.flow_conductive(10, 20);

    REQUIRE(flow.rows() == 1);
    REQUIRE(flow.cols() == 2);
    REQUIRE(flow(0, 0) == Catch::Approx(5.0));
    REQUIRE(flow(0, 1) == Catch::Approx(20.0));
}

TEST_CASE("DataModel conductive flow snaps a single time down",
          "[thermaldata]") {
    const DataModel model = make_flow_model();

    const Eigen::MatrixXd flow = model.flow_conductive(10, 20, 7.2);

    REQUIRE(flow.rows() == 1);
    REQUIRE(flow(0, 0) == Catch::Approx(7.0));
    REQUIRE(flow(0, 1) == Catch::Approx(40.0));
}

TEST_CASE("DataModel conductive range flow snaps start down and end up",
          "[thermaldata]") {
    const DataModel model = make_flow_model();

    const Eigen::MatrixXd flow = model.flow_conductive(10, 20, 6.2, 7.3);

    REQUIRE(flow.rows() == 3);
    REQUIRE(flow.cols() == 2);
    REQUIRE(flow(0, 0) == Catch::Approx(6.0));
    REQUIRE(flow(1, 0) == Catch::Approx(7.0));
    REQUIRE(flow(2, 0) == Catch::Approx(8.0));
    REQUIRE(flow(0, 1) == Catch::Approx(30.0));
    REQUIRE(flow(1, 1) == Catch::Approx(40.0));
    REQUIRE(flow(2, 1) == Catch::Approx(50.0));
}

TEST_CASE("DataModel conductive flow sums node groups", "[thermaldata]") {
    const DataModel model = make_flow_model();

    const Eigen::MatrixXd flow = model.flow_conductive({10}, {20, 30}, 7.0);

    REQUIRE(flow.rows() == 1);
    REQUIRE(flow(0, 0) == Catch::Approx(7.0));
    REQUIRE(flow(0, 1) == Catch::Approx(60.0));
}

TEST_CASE(
    "DataModel radiative flow uses stored coupling times "
    "without interpolation",
    "[thermaldata]") {
    const DataModel model = make_flow_model();

    const Eigen::MatrixXd flow = model.flow_radiative(20, 30, 6.1);
    const double expected =
        2.5 * STF_BOLTZ * (std::pow(321.0, 4) - std::pow(311.0, 4));

    REQUIRE(flow.rows() == 1);
    REQUIRE(flow(0, 0) == Catch::Approx(6.0));
    REQUIRE(flow(0, 1) == Catch::Approx(expected));
}

TEST_CASE("DataModel radiative range flow returns time-flow rows",
          "[thermaldata]") {
    const DataModel model = make_flow_model();

    const Eigen::MatrixXd flow = model.flow_radiative({10, 20}, {30}, 5.4, 6.2);

    REQUIRE(flow.rows() == 3);
    REQUIRE(flow(0, 0) == Catch::Approx(5.0));
    REQUIRE(flow(1, 0) == Catch::Approx(6.0));
    REQUIRE(flow(2, 0) == Catch::Approx(7.0));

    const double expected_first =
        (0.5 * STF_BOLTZ * (std::pow(320.0, 4) - std::pow(300.0, 4))) +
        (1.5 * STF_BOLTZ * (std::pow(320.0, 4) - std::pow(310.0, 4)));
    const double expected_second =
        (0.5 * STF_BOLTZ * (std::pow(321.0, 4) - std::pow(301.0, 4))) +
        (2.5 * STF_BOLTZ * (std::pow(321.0, 4) - std::pow(311.0, 4)));
    const double expected_third =
        (0.5 * STF_BOLTZ * (std::pow(322.0, 4) - std::pow(302.0, 4))) +
        (3.5 * STF_BOLTZ * (std::pow(322.0, 4) - std::pow(312.0, 4)));
    REQUIRE(flow(0, 1) == Catch::Approx(expected_first));
    REQUIRE(flow(1, 1) == Catch::Approx(expected_second));
    REQUIRE(flow(2, 1) == Catch::Approx(expected_third));
}

TEST_CASE("DataModel flow rejects missing nodes and mismatched times",
          "[thermaldata]") {
    const DataModel model = make_flow_model();
    const DataModel mismatched = make_mismatched_conductive_model();

    REQUIRE_THROWS_AS(model.flow_conductive(99, 20), std::invalid_argument);
    REQUIRE_THROWS_AS(mismatched.flow_conductive(10, 20, 1.0),
                      std::runtime_error);
}
