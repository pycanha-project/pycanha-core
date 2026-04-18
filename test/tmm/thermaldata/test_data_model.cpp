#include <Eigen/Sparse>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/dense_matrix_time_series.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

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
