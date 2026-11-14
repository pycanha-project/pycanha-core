// NOLINTBEGIN(readability-function-cognitive-complexity)
#include <Eigen/Sparse>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <memory>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "radiative/retracer_network.hpp"

namespace {

namespace rad = pycanha::radiative;
using pycanha::Index;
using rad::SparseMatrix;

using Triplet = Eigen::Triplet<double, rad::SparseIndex>;

SparseMatrix make_matrix(Index rows, Index cols,
                         const std::vector<Triplet>& entries) {
    SparseMatrix matrix(rows, cols);
    matrix.setFromTriplets(entries.begin(), entries.end());
    matrix.makeCompressed();
    return matrix;
}

// Diffusive nodes 1, 2, 3 and boundary nodes 8, 9, in that internal order.
std::shared_ptr<pycanha::ThermalMathematicalModel> make_model() {
    auto model = std::make_shared<pycanha::ThermalMathematicalModel>("split");
    for (const int number : {1, 2, 3}) {
        model->add_node(pycanha::Node(number));
    }
    for (const int number : {8, 9}) {
        pycanha::Node node(number);
        node.set_type(pycanha::BOUNDARY_NODE);
        model->add_node(node);
    }
    model->add_radiative_coupling(1, 2, 1.0);
    model->add_radiative_coupling(1, 8, 1.0);
    model->add_radiative_coupling(2, 3, 1.0);
    model->add_radiative_coupling(8, 9, 1.0);
    model->network().synchronize_structure();
    model->network().radiative_matrices().compress();
    return model;
}

Index index_of(const pycanha::ThermalMathematicalModel& model, int number) {
    const auto index = model.nodes().get_idx_from_node_num(number);
    REQUIRE(index.has_value());
    return index.value_or(-1);
}

}  // namespace

TEST_CASE("Face factors reduce to node couplings", "[radiative][retracer]") {
    // Faces 0 and 1 belong to node index 0, face 2 to index 2, face 3 has no
    // node. Columns 4, 5, 6 are space, inactive and lost.
    const std::vector<Index> face_node{0, 0, 2, -1};
    const SparseMatrix factors = make_matrix(4, 7,
                                             {{0, 1, 0.5},
                                              {0, 2, 0.25},
                                              {1, 2, 0.125},
                                              {2, 3, 4.0},
                                              {0, 4, 0.1},
                                              {2, 4, 0.2},
                                              {1, 5, 0.3},
                                              {2, 6, 0.05},
                                              {3, 4, 7.0}});
    const auto nodes = rad::detail::reduce_to_nodes(factors, face_node,
                                                    /*space_index=*/3,
                                                    /*num_nodes=*/4);
    REQUIRE(nodes.couplings.coeff(0, 2) == Catch::Approx(0.375));
    REQUIRE(nodes.couplings.coeff(0, 3) == Catch::Approx(0.1));
    REQUIRE(nodes.couplings.coeff(2, 3) == Catch::Approx(0.2));
    REQUIRE(nodes.couplings.nonZeros() == 3);
    REQUIRE(nodes.intra_node_total == Catch::Approx(0.5));
    REQUIRE(nodes.space_total == Catch::Approx(0.3));
    REQUIRE(nodes.inactive_total == Catch::Approx(0.3));
    REQUIRE(nodes.lost_total == Catch::Approx(0.05));

    SECTION("without a space node the space column is only totalled") {
        const auto dropped =
            rad::detail::reduce_to_nodes(factors, face_node, -1, 4);
        REQUIRE(dropped.couplings.nonZeros() == 1);
        REQUIRE(dropped.space_total == Catch::Approx(0.3));
    }
}

TEST_CASE("A trace is split into the coupling slots and an overflow",
          "[radiative][retracer]") {
    auto model = make_model();
    auto& matrices = model->network().radiative_matrices();
    const auto version = matrices.structure_version();
    const Index i1 = index_of(*model, 1);
    const Index i2 = index_of(*model, 2);
    const Index i3 = index_of(*model, 3);
    const Index i8 = index_of(*model, 8);
    const Index i9 = index_of(*model, 9);

    // (1, 2) and (8, 9) have slots; (1, 3), (3, 9) do not; the slots (1, 8)
    // and (2, 3) get no entry.
    const SparseMatrix trace =
        make_matrix(5, 5,
                    {{static_cast<int>(i1), static_cast<int>(i2), 0.5},
                     {static_cast<int>(i1), static_cast<int>(i3), 0.25},
                     {static_cast<int>(i3), static_cast<int>(i9), 0.125},
                     {static_cast<int>(i8), static_cast<int>(i9), 2.0}});
    const auto split = rad::detail::split_into_slots(trace, matrices);

    REQUIRE(matrices.structure_version() == version);
    REQUIRE(model->radiative_couplings().get_coupling_value(1, 2) == 0.5);
    REQUIRE(model->radiative_couplings().get_coupling_value(8, 9) == 2.0);
    REQUIRE(model->radiative_couplings().get_coupling_value(1, 8) == 0.0);
    REQUIRE(model->radiative_couplings().get_coupling_value(2, 3) == 0.0);
    REQUIRE(matrices.get_num_total_couplings() == 4);

    REQUIRE(split.overflow.nonZeros() == 2);
    REQUIRE(split.overflow.coeff(i1, i3) == 0.25);
    REQUIRE(split.overflow.coeff(i3, i9) == 0.125);
    REQUIRE(split.overflow_total == Catch::Approx(0.375));
    REQUIRE(split.trace_total == Catch::Approx(2.875));

    SECTION("a trace with a slot for every entry leaves no overflow") {
        const SparseMatrix inside = make_matrix(
            5, 5, {{static_cast<int>(i2), static_cast<int>(i3), 3.0}});
        const auto none = rad::detail::split_into_slots(inside, matrices);
        REQUIRE(none.overflow.size() == 0);
        REQUIRE(model->radiative_couplings().get_coupling_value(2, 3) == 3.0);
        REQUIRE(model->radiative_couplings().get_coupling_value(1, 2) == 0.0);
    }
}

TEST_CASE("The union of trace patterns", "[radiative][retracer]") {
    const SparseMatrix first = make_matrix(3, 3, {{0, 1, 1.0}});
    const SparseMatrix second = make_matrix(3, 3, {{0, 1, 2.0}, {1, 2, 3.0}});
    const std::array<const SparseMatrix*, 2> traces{&first, &second};
    const SparseMatrix pattern = rad::detail::pattern_union(traces, 3);
    REQUIRE(pattern.nonZeros() == 2);
    REQUIRE(pattern.coeff(0, 1) == 0.0);
    REQUIRE(pattern.coeff(1, 2) == 0.0);
}

TEST_CASE("The overflow load conserves energy", "[radiative][retracer]") {
    const SparseMatrix overflow =
        make_matrix(3, 3, {{0, 1, 0.5}, {0, 2, 0.25}});
    const std::vector<double> temperatures{300.0, 200.0, 400.0};
    std::vector<double> load;
    rad::detail::overflow_load(overflow, temperatures, load);
    const auto fourth = [](double t) { return t * t * t * t; };
    const double to_1 =
        pycanha::STF_BOLTZ * 0.5 * (fourth(200.0) - fourth(300.0));
    const double to_2 =
        pycanha::STF_BOLTZ * 0.25 * (fourth(400.0) - fourth(300.0));
    REQUIRE(load.size() == 3);
    REQUIRE(load[0] == Catch::Approx(to_1 + to_2));
    REQUIRE(load[1] == Catch::Approx(-to_1));
    REQUIRE(load[2] == Catch::Approx(-to_2));
    REQUIRE(std::abs(load[0] + load[1] + load[2]) < 1e-9);
}

// NOLINTEND(readability-function-cognitive-complexity)
