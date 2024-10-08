#include <Eigen/Sparse>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>
#include <vector>

#include "pycanha-core/tmm/nodes.hpp"

// NOLINTBEGIN(readability-function-cognitive-complexity)

// The seed is constant making the random number generator deterministic (which
// is what we want for testing)
// NOLINTBEGIN(cert-msc32-c,cert-msc51-cpp)
class DoubleRandomGenerator {
  public:
    // Constructor
    DoubleRandomGenerator(double min, double max) : _dist(min, max) {
        _rng.seed(seed);
    }
    // Generator function
    double generate_random() { return _dist(_rng); }

  private:
    std::uniform_real_distribution<double> _dist;
    std::mt19937 _rng;
    const std::mt19937::result_type seed = 100;  // same as uint
};
// NOLINTEND(cert-msc32-c,cert-msc51-cpp)

void assert_tn_has_same_values_as_tns(Node& tn, Nodes& tns,
                                      bool check_internal) {
    const int usr_num = tn.get_node_num();

    if (check_internal) {
        REQUIRE(tn.get_int_node_num() == tns.get_idx_from_node_num(usr_num));
    }
    REQUIRE(tn.get_type() == tns.get_type(usr_num));
    REQUIRE(tn.get_T() == tns.get_T(usr_num));
    REQUIRE(tn.get_C() == tns.get_C(usr_num));
    REQUIRE(tn.get_qs() == tns.get_qs(usr_num));
    REQUIRE(tn.get_qa() == tns.get_qa(usr_num));
    REQUIRE(tn.get_qe() == tns.get_qe(usr_num));
    REQUIRE(tn.get_qi() == tns.get_qi(usr_num));
    REQUIRE(tn.get_qr() == tns.get_qr(usr_num));
    REQUIRE(tn.get_a() == tns.get_a(usr_num));
    REQUIRE(tn.get_fx() == tns.get_fx(usr_num));
    REQUIRE(tn.get_fy() == tns.get_fy(usr_num));
    REQUIRE(tn.get_fz() == tns.get_fz(usr_num));
    REQUIRE(tn.get_eps() == tns.get_eps(usr_num));
    REQUIRE(tn.get_aph() == tns.get_aph(usr_num));

    // TODO: The Sparse Vector for LiteralString is not working properly. FIX
    // REQUIRE(tn.get_literal_C().compare(tns.get_literal_C(usr_num)) == 0);
}

void assert_trivial_zeros(const std::vector<Index>& non_zero_nodes,
                          Eigen::SparseVector<double>& attr_sp_vector) {
    for (Index i = 0; i < attr_sp_vector.nonZeros(); i++) {
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        REQUIRE(non_zero_nodes[static_cast<VectorIndex>(i)] ==
                attr_sp_vector.innerIndexPtr()[i]);
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }
}

void assert_trivial_zeros(const std::vector<Index>& non_zero_nodes,
                          Eigen::SparseVector<LiteralString>& attr_sp_vector) {
    for (Index i = 0; i < attr_sp_vector.nonZeros(); i++) {
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        REQUIRE(non_zero_nodes[static_cast<VectorIndex>(i)] ==
                attr_sp_vector.innerIndexPtr()[i]);
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }
}

void assert_blank_nodes_attributes_are_trivial_zeros(
    const std::vector<Index>& blank_nodes, Nodes& tns) {
    std::vector<Index> blank_internal_number;
    std::vector<Index> non_blank_internal_number;
    for (Index i = 0; i < tns.num_nodes(); i++) {
        const int usr_num = tns.get_node_num_from_idx(i);
        if (std::find(blank_nodes.begin(), blank_nodes.end(), usr_num) !=
            blank_nodes.end()) {
            blank_internal_number.push_back(i);
        } else {
            non_blank_internal_number.push_back(i);
        }
    }
    assert_trivial_zeros(non_blank_internal_number, tns.qs_vector);
    assert_trivial_zeros(non_blank_internal_number, tns.qa_vector);
    assert_trivial_zeros(non_blank_internal_number, tns.qe_vector);
    assert_trivial_zeros(non_blank_internal_number, tns.qi_vector);
    assert_trivial_zeros(non_blank_internal_number, tns.qr_vector);
    assert_trivial_zeros(non_blank_internal_number, tns.a_vector);
    assert_trivial_zeros(non_blank_internal_number, tns.fx_vector);
    assert_trivial_zeros(non_blank_internal_number, tns.fy_vector);
    assert_trivial_zeros(non_blank_internal_number, tns.fz_vector);
    assert_trivial_zeros(non_blank_internal_number, tns.eps_vector);
    assert_trivial_zeros(non_blank_internal_number, tns.aph_vector);
    assert_trivial_zeros(non_blank_internal_number, tns.literals_C);
}

TEST_CASE("Nodes Testing", "[nodes]") {
    // Random double generator for assigning values to node attributes
    DoubleRandomGenerator rand_gen(0.0, 10000.0);

    // Nodes instance.
    Nodes tns;

    // Nodes for testing
    std::vector<Index> num_nodes{1,  5,  25, 43, 48, 53, 56, 57, 58, 63,
                                 68, 73, 77, 78, 81, 83, 85, 89, 94, 98};
    std::vector<Index> bound_nodes{1, 5, 43, 63, 68, 73, 85, 94, 98};
    std::vector<Index> blank_nodes{1, 48, 53, 78, 94};
    std::vector<Index> insertion_order{63, 58, 5,  57, 43, 94, 1,  48, 89, 25,
                                       83, 98, 68, 78, 85, 81, 73, 53, 56, 77};

    // Create N nodes and store them in nodes_vector
    // NOLINTNEXTLINE(readability-identifier-naming)
    const Index N = std::ssize(num_nodes);
    std::vector<Node> nodes_vector;
    std::vector<Node> nodes_vector_copy;
    nodes_vector.reserve(static_cast<SizeType>(N));
    nodes_vector_copy.reserve(static_cast<SizeType>(N));

    // Internal order vector
    std::vector<Index> internal_order;
    internal_order.reserve(static_cast<SizeType>(N));
    Index diff_index = 0;

    for (Index i = 0; i < N; i++) {
        const int usr_num =
            static_cast<int>(num_nodes[static_cast<VectorIndex>(i)]);
        Node node(usr_num);

        // Blank/filled nodes
        if (std::find(blank_nodes.begin(), blank_nodes.end(), usr_num) ==
            blank_nodes.end()) {
            node.set_T(rand_gen.generate_random());
            node.set_C(rand_gen.generate_random());
            node.set_qs(rand_gen.generate_random());
            node.set_qa(rand_gen.generate_random());
            node.set_qe(rand_gen.generate_random());
            node.set_qi(rand_gen.generate_random());
            node.set_qr(rand_gen.generate_random());
            node.set_a(rand_gen.generate_random());
            node.set_fx(rand_gen.generate_random());
            node.set_fy(rand_gen.generate_random());
            node.set_fz(rand_gen.generate_random());
            node.set_eps(rand_gen.generate_random());
            node.set_aph(rand_gen.generate_random());
            node.set_literal_C(std::to_string(rand_gen.generate_random()));
        }

        if (std::find(bound_nodes.begin(), bound_nodes.end(), usr_num) !=
            bound_nodes.end()) {
            node.set_type('B');
            node.set_T(rand_gen.generate_random());
            internal_order.push_back(i);
        } else {
            internal_order.emplace(internal_order.begin() + diff_index, i);
            diff_index++;
        }

        nodes_vector.push_back(node);
        nodes_vector_copy.push_back(node);
    }

    // Add nodes to thermal nodes
    for (Index i = 0; i < N; i++) {
        const int usr_num =
            static_cast<int>(insertion_order[static_cast<VectorIndex>(i)]);
        auto it =
            std::find(insertion_order.begin(), insertion_order.end(), usr_num);
        if (it != insertion_order.end()) {
            const Index node_ix = it - insertion_order.begin();
            tns.add_node(nodes_vector[static_cast<VectorIndex>(node_ix)]);
        }
    }
    // Assert all nodes have been inserted
    REQUIRE(tns.num_nodes() == N);

    // Assert that tns have the values of the nodes stored
    for (SizeType i = 0; i < static_cast<SizeType>(N); i++) {
        assert_tn_has_same_values_as_tns(nodes_vector_copy[i], tns,
                                         /*check_internal=*/false);
    }

    // Assert that the nodes added to tns return the same values as tns
    for (SizeType i = 0; i < static_cast<SizeType>(N); i++) {
        assert_tn_has_same_values_as_tns(nodes_vector[i], tns,
                                         /*check_internal=*/true);
    }

    // Assert internal order is correct
    for (SizeType i = 0; i < static_cast<SizeType>(N); i++) {
        const auto int_ix = static_cast<SizeType>(internal_order[i]);
        REQUIRE(nodes_vector_copy[int_ix].get_node_num() ==
                tns.get_node_num_from_idx(static_cast<int>(i)));
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    // Assert temperatures and capacities are contiguous in memory
    double* temperatures_vector = tns.T_vector.data();
    double* capacities_vector = tns.C_vector.data();
    for (SizeType i = 0; i < static_cast<SizeType>(N); i++) {
        const auto int_ix = static_cast<SizeType>(internal_order[i]);
        const double node_temp = nodes_vector_copy[int_ix].get_T();
        const double node_capacity = nodes_vector_copy[int_ix].get_C();
        REQUIRE(node_temp == temperatures_vector[i]);
        REQUIRE(node_capacity == capacities_vector[i]);
    }

    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    // Assert that only non-zero elements are entries of the sparse vectors
    assert_blank_nodes_attributes_are_trivial_zeros(blank_nodes, tns);

    // Check that the map is updated and flagged outdated properly
    Node node_map_check_d1(1001);
    Node node_map_check_d2(1002);
    Node node_map_check_b1(1003);
    Node node_map_check_b2(1004);
    node_map_check_b1.set_type('B');
    node_map_check_b2.set_type('B');
    tns.add_node(node_map_check_d1);  // Map flagged outdated
    REQUIRE_FALSE(tns.is_mapped());
    tns.set_T(1001, 1001.0);  // Map updated
    REQUIRE(tns.is_mapped());

    tns.add_node(node_map_check_b1);  // Map flagged outdated
    REQUIRE_FALSE(tns.is_mapped());
    tns.set_T(1003, 1003.0);  // Map updated
    REQUIRE(tns.is_mapped());

    tns.add_node(node_map_check_d2);  // Map flagged outdated
    REQUIRE_FALSE(tns.is_mapped());
    tns.set_T(1002, 1002.0);  // Map updated
    REQUIRE(tns.is_mapped());

    tns.add_node(node_map_check_b2);  // Map flagged outdated
    REQUIRE_FALSE(tns.is_mapped());
    tns.set_T(1004, 1004.0);  // Map updated
    REQUIRE(tns.is_mapped());

    // Additional tests can be added here...
}

// NOLINTEND(readability-function-cognitive-complexity)
