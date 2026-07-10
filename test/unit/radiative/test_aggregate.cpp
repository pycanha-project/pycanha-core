#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/radiative/aggregate.hpp"
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
