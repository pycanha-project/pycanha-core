#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/radiative/aggregate.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/sparse.hpp"

namespace rad = pycanha::radiative;
using pycanha::NodeNum;
using pycanha::gmm::NO_NODE;

namespace {

// 6 face slots mapped to nodes {5, 5, 7, NO_NODE, 7, 9}. One stored entry
// per row for hand-checkable sums.
constexpr std::array<NodeNum, 6> node_numbers = {5, 5, 7, NO_NODE, 7, 9};
constexpr std::array<double, 6> areas = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

[[nodiscard]] rad::SparseF64 make_face_matrix() {
    // Row r has a single entry 0.5 at column (r + 1) % 6.
    rad::SparseF64 matrix;
    matrix.rows = 6;
    matrix.cols = 6;
    matrix.indptr.resize(7);
    matrix.indices.resize(6);
    matrix.values.resize(6);
    for (Eigen::Index row = 0; row < 6; ++row) {
        matrix.indptr(row) = row;
        matrix.indices(row) = static_cast<std::int32_t>((row + 1) % 6);
        matrix.values(row) = 0.5;
    }
    matrix.indptr(6) = 6;
    return matrix;
}

[[nodiscard]] double node_value(const rad::SparseF64& matrix, Eigen::Index row,
                                std::int32_t col) {
    for (std::int64_t k = matrix.indptr(row); k < matrix.indptr(row + 1); ++k) {
        if (matrix.indices(k) == col) {
            return matrix.values(k);
        }
    }
    return 0.0;
}

}  // namespace

TEST_CASE("radiative aggregate: unique node list drops NO_NODE",
          "[radiative]") {
    const std::vector<NodeNum> nodes = rad::aggregate_nodes(node_numbers);
    REQUIRE(nodes == std::vector<NodeNum>{5, 7, 9});
}

TEST_CASE("radiative aggregate: area-weighted node matrix", "[radiative]") {
    const rad::SparseF64 face_matrix = make_face_matrix();
    const rad::SparseF64 node_matrix =
        rad::aggregate_matrix(face_matrix, node_numbers, areas);

    // Nodes {5, 7, 9} -> indices {0, 1, 2}. By hand:
    //  slot0 (n5, A=1) -> slot1 (n5):    5->5 += 1*0.5
    //  slot1 (n5, A=2) -> slot2 (n7):    5->7 += 2*0.5
    //  slot2 (n7, A=3) -> slot3 (NO_NODE): dropped
    //  slot3 (NO_NODE) -> anything:      dropped (row has no node)
    //  slot4 (n7, A=5) -> slot5 (n9):    7->9 += 5*0.5
    //  slot5 (n9, A=6) -> slot0 (n5):    9->5 += 6*0.5
    REQUIRE(node_matrix.rows == 3);
    REQUIRE(node_matrix.cols == 3);
    REQUIRE(node_matrix.nnz() == 4);
    REQUIRE(node_value(node_matrix, 0, 0) == Catch::Approx(0.5));
    REQUIRE(node_value(node_matrix, 0, 1) == Catch::Approx(1.0));
    REQUIRE(node_value(node_matrix, 1, 2) == Catch::Approx(2.5));
    REQUIRE(node_value(node_matrix, 2, 0) == Catch::Approx(3.0));
}

namespace {

// A 2-row matrix in the traced-result shape: 2 real columns plus the
// space/inactive/lost buckets. Row r: 0.25 to the other face, 0.5 to space,
// 0.25 to inactive.
[[nodiscard]] rad::SparseF64 make_bucket_matrix() {
    rad::SparseF64 matrix;
    matrix.rows = 2;
    matrix.cols = 2 + rad::num_virtual_columns;
    matrix.indptr.resize(3);
    matrix.indices.resize(6);
    matrix.values.resize(6);
    for (Eigen::Index row = 0; row < 2; ++row) {
        matrix.indptr(row) = 3 * row;
        matrix.indices(3 * row) = static_cast<std::int32_t>(1 - row);
        matrix.values(3 * row) = 0.25;
        matrix.indices((3 * row) + 1) =
            static_cast<std::int32_t>(2 + rad::space_column_offset);
        matrix.values((3 * row) + 1) = 0.5;
        matrix.indices((3 * row) + 2) =
            static_cast<std::int32_t>(2 + rad::inactive_column_offset);
        matrix.values((3 * row) + 2) = 0.25;
    }
    matrix.indptr(2) = 6;
    return matrix;
}

}  // namespace

TEST_CASE("radiative aggregate: virtual bucket columns map to nodes",
          "[radiative]") {
    const rad::SparseF64 matrix = make_bucket_matrix();
    const std::array<NodeNum, 2> row_nodes = {10, 20};
    const std::array<double, 2> face_areas = {2.0, 4.0};

    // Default overload: buckets are simply dropped.
    const rad::SparseF64 dropped =
        rad::aggregate_matrix(matrix, row_nodes, face_areas);
    REQUIRE(dropped.rows == 2);
    REQUIRE(dropped.cols == 2);
    REQUIRE(node_value(dropped, 0, 1) == Catch::Approx(0.5));
    REQUIRE(node_value(dropped, 1, 0) == Catch::Approx(1.0));

    // Column overload: the user maps space to a real node (99), keeps
    // inactive/lost unassigned.
    const std::array<NodeNum, 5> col_nodes = {10, 20, 99, NO_NODE, NO_NODE};
    const rad::SparseF64 mapped =
        rad::aggregate_matrix(matrix, row_nodes, col_nodes, face_areas);
    REQUIRE(mapped.rows == 2);
    REQUIRE(mapped.cols == 3);  // {10, 20, 99}
    REQUIRE(node_value(mapped, 0, 2) == Catch::Approx(2.0 * 0.5));
    REQUIRE(node_value(mapped, 1, 2) == Catch::Approx(4.0 * 0.5));
    REQUIRE(node_value(mapped, 0, 1) == Catch::Approx(0.5));
}

TEST_CASE("radiative aggregate: flux to watts per node", "[radiative]") {
    Eigen::VectorXd flux(6);
    flux << 10.0, 20.0, 30.0, 40.0, 50.0, 60.0;

    const Eigen::VectorXd watts =
        rad::aggregate_flux(flux, node_numbers, areas);

    REQUIRE(watts.size() == 3);
    // n5: 10*1 + 20*2; n7: 30*3 + 50*5; n9: 60*6. Slot 3 (NO_NODE) dropped.
    REQUIRE(watts(0) == Catch::Approx(50.0));
    REQUIRE(watts(1) == Catch::Approx(340.0));
    REQUIRE(watts(2) == Catch::Approx(360.0));
}
