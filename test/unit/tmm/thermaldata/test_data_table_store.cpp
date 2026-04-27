#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "pycanha-core/thermaldata/data_table_store.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("DataTableStore manages vector lookup tables", "[thermaldata]") {
    DataTableStore store;

    Eigen::VectorXd x(2);
    x << 0.0, 1.0;
    LookupTableVec1D::MatrixType y(2, 2);
    y << 1.0, 2.0, 3.0, 4.0;

    const auto& table = store.add_table("vector", LookupTableVec1D(x, y));

    REQUIRE(store.has_table("vector"));
    REQUIRE(store.size() == 1U);
    REQUIRE(table.evaluate(0.5)(0) == Catch::Approx(2.0));

    store.remove_table("vector");
    REQUIRE_FALSE(store.has_table("vector"));
    REQUIRE_THROWS_AS(store.get_table("vector"), std::out_of_range);
}
