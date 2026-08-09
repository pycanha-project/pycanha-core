#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/radiative/aggregate.hpp"
#include "pycanha-core/radiative/results.hpp"

namespace rad = pycanha::radiative;
using pycanha::NodeNum;
using pycanha::gmm::NO_NODE;

namespace {

// Row-major CSR from (row, col, value) triplets already in row-major order.
[[nodiscard]] rad::SparseMatrix make_csr(
    Eigen::Index rows, Eigen::Index cols,
    const std::vector<Eigen::Triplet<double>>& entries) {
    rad::SparseMatrix matrix(rows, cols);
    matrix.setFromTriplets(entries.begin(), entries.end());
    return matrix;
}

// 6 face slots mapped to nodes {5, 5, 7, NO_NODE, 7, 9}. One stored entry
// per row for hand-checkable sums.
constexpr std::array<NodeNum, 6> node_numbers = {5, 5, 7, NO_NODE, 7, 9};

[[nodiscard]] rad::SparseMatrix make_face_matrix() {
    // Row r has a single entry 0.5 at column (r + 1) % 6.
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(6);
    for (rad::SparseIndex row = 0; row < 6; ++row) {
        entries.emplace_back(row, (row + 1) % 6, 0.5);
    }
    return make_csr(6, 6, entries);
}

// A face pair mapping to a descending node pair has to be canonicalised,
// not written where it fell; anything below the diagonal means it was not.
void require_upper_triangular(const rad::SparseMatrix& matrix) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (rad::SparseMatrix::InnerIterator entry(matrix, row); entry;
             ++entry) {
            REQUIRE(entry.col() > row);
        }
    }
}

[[nodiscard]] double node_value(const rad::SparseMatrix& matrix,
                                Eigen::Index row, Eigen::Index col) {
    for (rad::SparseMatrix::InnerIterator entry(matrix, row); entry; ++entry) {
        if (entry.col() == col) {
            return entry.value();
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

TEST_CASE("radiative aggregate: extensive entries sum into the upper triangle",
          "[radiative]") {
    const rad::SparseMatrix face_matrix = make_face_matrix();
    const rad::AggregateResult node =
        rad::aggregate_matrix(face_matrix, node_numbers);

    // Nodes {5, 7, 9} -> indices {0, 1, 2}. The input is already extensive,
    // so every contribution is a plain 0.5. By hand:
    //  slot0 (n5) -> slot1 (n5):      5->5, intra-node: dropped
    //  slot1 (n5) -> slot2 (n7):      5->7 += 0.5
    //  slot2 (n7) -> slot3 (NO_NODE): dropped
    //  slot3 (NO_NODE) -> anything:   dropped (row has no node)
    //  slot4 (n7) -> slot5 (n9):      7->9 += 0.5
    //  slot5 (n9) -> slot0 (n5):      9->5, which is LOWER in node space and
    //                                 must land at 5->9 instead
    REQUIRE(node.matrix.rows() == 3);
    REQUIRE(node.matrix.cols() == 3);
    REQUIRE(node.intra_node_total == Catch::Approx(0.5));
    REQUIRE(node.matrix.nonZeros() == 3);
    REQUIRE(node_value(node.matrix, 0, 1) == Catch::Approx(0.5));
    REQUIRE(node_value(node.matrix, 1, 2) == Catch::Approx(0.5));
    REQUIRE(node_value(node.matrix, 0, 2) == Catch::Approx(0.5));
    require_upper_triangular(node.matrix);
}

TEST_CASE("radiative aggregate: shuffled node numbers stay upper-triangular",
          "[radiative]") {
    // Node numbers that run OPPOSITE to the face-slot order, so every face
    // pair i < j maps to a node pair m > n. A fixture whose node numbering
    // happens to follow face numbering would pass even without the
    // canonicalisation this checks.
    constexpr std::array<NodeNum, 6> descending = {60, 50, 40, 30, 20, 10};
    const rad::AggregateResult node =
        rad::aggregate_matrix(make_face_matrix(), descending);

    REQUIRE(node.matrix.rows() == 6);
    REQUIRE(node.matrix.nonZeros() == 6);
    REQUIRE(node.intra_node_total == 0.0);
    require_upper_triangular(node.matrix);
}

TEST_CASE("radiative aggregate: the reduction adds no area factor",
          "[radiative]") {
    // Scaling every stored coupling scales the node matrix by exactly the
    // same factor. An area weighting left in the reduction would break that
    // proportionality - and would quietly produce an m^4 matrix that still
    // solves and still looks plausible.
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(6);
    for (rad::SparseIndex row = 0; row < 6; ++row) {
        entries.emplace_back(row, (row + 1) % 6, 8.0 * 0.5);
    }
    const rad::AggregateResult plain =
        rad::aggregate_matrix(make_face_matrix(), node_numbers);
    const rad::AggregateResult eight =
        rad::aggregate_matrix(make_csr(6, 6, entries), node_numbers);
    REQUIRE(eight.intra_node_total ==
            Catch::Approx(8.0 * plain.intra_node_total));
    for (Eigen::Index row = 0; row < plain.matrix.rows(); ++row) {
        for (Eigen::Index col = 0; col < plain.matrix.cols(); ++col) {
            REQUIRE(node_value(eight.matrix, row, col) ==
                    Catch::Approx(8.0 * node_value(plain.matrix, row, col)));
        }
    }
}

namespace {

// A 2-row matrix in the traced-result shape: 2 real columns plus the
// space/inactive/lost buckets. Row r: 0.25 to the other face, 0.5 to space,
// 0.25 to inactive.
[[nodiscard]] rad::SparseMatrix make_bucket_matrix() {
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(6);
    for (rad::SparseIndex row = 0; row < 2; ++row) {
        entries.emplace_back(row, 1 - row, 0.25);
        entries.emplace_back(
            row, 2 + static_cast<rad::SparseIndex>(rad::space_column_offset),
            0.5);
        entries.emplace_back(
            row, 2 + static_cast<rad::SparseIndex>(rad::inactive_column_offset),
            0.25);
    }
    return make_csr(2, 2 + static_cast<Eigen::Index>(rad::num_virtual_columns),
                    entries);
}

}  // namespace

TEST_CASE("radiative aggregate: virtual bucket columns map to nodes",
          "[radiative]") {
    const rad::SparseMatrix matrix = make_bucket_matrix();
    const std::array<NodeNum, 2> row_nodes = {10, 20};

    // Default overload: buckets are simply dropped, and the two directions
    // of the face pair fold onto the single upper-triangular entry.
    const rad::AggregateResult dropped =
        rad::aggregate_matrix(matrix, row_nodes);
    REQUIRE(dropped.matrix.rows() == 2);
    REQUIRE(dropped.matrix.cols() == 2);
    REQUIRE(node_value(dropped.matrix, 0, 1) == Catch::Approx(0.5));
    REQUIRE(node_value(dropped.matrix, 1, 0) == 0.0);

    // Column overload: the user maps space to a real node (99), keeps
    // inactive/lost unassigned. Rows and columns are labelled independently
    // there, so entries stay exactly where they land.
    const std::array<NodeNum, 5> col_nodes = {10, 20, 99, NO_NODE, NO_NODE};
    const rad::AggregateResult mapped =
        rad::aggregate_matrix(matrix, row_nodes, col_nodes);
    REQUIRE(mapped.matrix.rows() == 2);
    REQUIRE(mapped.matrix.cols() == 3);  // {10, 20, 99}
    REQUIRE(node_value(mapped.matrix, 0, 2) == Catch::Approx(0.5));
    REQUIRE(node_value(mapped.matrix, 1, 2) == Catch::Approx(0.5));
    REQUIRE(node_value(mapped.matrix, 0, 1) == Catch::Approx(0.25));
}

TEST_CASE("radiative aggregate: flux to watts per node", "[radiative]") {
    Eigen::VectorXd flux(6);
    flux << 10.0, 20.0, 30.0, 40.0, 50.0, 60.0;

    constexpr std::array<double, 6> areas = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const Eigen::VectorXd watts =
        rad::aggregate_flux(flux, node_numbers, areas);

    REQUIRE(watts.size() == 3);
    // n5: 10*1 + 20*2; n7: 30*3 + 50*5; n9: 60*6. Slot 3 (NO_NODE) dropped.
    REQUIRE(watts(0) == Catch::Approx(50.0));
    REQUIRE(watts(1) == Catch::Approx(340.0));
    REQUIRE(watts(2) == Catch::Approx(360.0));
}
