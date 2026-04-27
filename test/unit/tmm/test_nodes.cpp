#include <Eigen/Sparse>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

// NOLINTBEGIN(readability-function-cognitive-complexity,
// bugprone-chained-comparison)

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
        REQUIRE(non_zero_nodes[to_sizet(i)] ==
                attr_sp_vector.innerIndexPtr()[i]);
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }
}

void assert_trivial_zeros(const std::vector<Index>& non_zero_nodes,
                          Eigen::SparseVector<LiteralString>& attr_sp_vector) {
    for (Index i = 0; i < attr_sp_vector.nonZeros(); i++) {
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        REQUIRE(non_zero_nodes[to_sizet(i)] ==
                attr_sp_vector.innerIndexPtr()[i]);
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }
}

void assert_blank_nodes_attributes_are_trivial_zeros(
    const std::vector<Index>& blank_nodes, Nodes& tns) {
    std::vector<Index> blank_internal_number;
    std::vector<Index> non_blank_internal_number;
    for (Index i = 0; i < tns.num_nodes(); i++) {
        const auto usr_num = tns.get_node_num_from_idx(i);
        REQUIRE(usr_num.has_value());
        if (std::find(blank_nodes.begin(), blank_nodes.end(), *usr_num) !=
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

// Test Node constructor with Nodes pointer
TEST_CASE("Node Constructor with Nodes pointer", "[node]") {
    std::shared_ptr<Nodes> tns = std::make_shared<Nodes>();

    Node stored(5);
    stored.set_T(280.0);
    stored.set_qi(12.0);
    tns->add_node(stored);

    std::weak_ptr<Nodes> weak_tns = tns;
    Node tn(5, weak_tns);

    REQUIRE_FALSE(weak_tns.expired());
    REQUIRE(tn.get_int_node_num().has_value());
    REQUIRE(tn.get_int_parent_pointer() == stored.get_int_parent_pointer());
    REQUIRE(tn.get_T() == 280.0);
    REQUIRE(tn.get_qi() == 12.0);

    tn.set_T(285.0);
    REQUIRE(tns->get_T(5) == 285.0);
    REQUIRE(tns->set_qi(5, 18.0));
    REQUIRE(tn.get_qi() == 18.0);
}

TEST_CASE("Nodes expose safe fallback access for missing nodes", "[nodes]") {
    Nodes tns;
    Node node(10);
    node.set_T(273.15);
    node.set_C(12.5);
    node.set_qi(5.0);
    tns.add_node(node);

    REQUIRE(tns.get_idx_from_node_num(10) == std::optional<Index>{0});
    REQUIRE(tns.get_node_num_from_idx(0) == std::optional<NodeNum>{10});

    double* temperature_ref = tns.get_T_value_ref(10);
    REQUIRE(temperature_ref != nullptr);
    *temperature_ref = 275.0;
    REQUIRE(tns.get_T(10) == 275.0);

    double* heat_ref = tns.get_qi_value_ref(10);
    REQUIRE(heat_ref != nullptr);
    *heat_ref = 8.0;
    REQUIRE(tns.get_qi(10) == 8.0);

    REQUIRE(std::isnan(tns.get_T(999)));
    REQUIRE_FALSE(tns.set_T(999, 1.0));
    REQUIRE(tns.get_T_value_ref(999) == nullptr);
    REQUIRE(tns.get_qi_value_ref(999) == nullptr);
    REQUIRE_FALSE(tns.get_idx_from_node_num(999).has_value());
    REQUIRE_FALSE(tns.get_node_num_from_idx(-1).has_value());
    REQUIRE_FALSE(tns.get_node_num_from_idx(5).has_value());
    REQUIRE_FALSE(tns.is_node(999));
}

TEST_CASE("Nodes removal closes mapping gaps", "[nodes]") {
    Nodes tns;

    Node diff_10(10);
    Node diff_20(20);
    Node bound_99(99);
    bound_99.set_type('B');

    tns.add_node(diff_10);
    tns.add_node(diff_20);
    tns.add_node(bound_99);

    tns.remove_node(20);

    REQUIRE(tns.num_nodes() == 2);
    REQUIRE_FALSE(tns.is_node(20));
    REQUIRE(tns.get_node_num_from_idx(0) == std::optional<NodeNum>{10});
    REQUIRE(tns.get_node_num_from_idx(1) == std::optional<NodeNum>{99});
    REQUIRE(tns.get_idx_from_node_num(99) == std::optional<Index>{1});

    tns.remove_node(999);
    REQUIRE(tns.num_nodes() == 2);
}

TEST_CASE("Nodes copy and move preserve stored values", "[nodes]") {
    Nodes original;

    Node diff_10(10);
    diff_10.set_T(300.0);
    diff_10.set_qi(4.0);

    Node bound_20(20);
    bound_20.set_type('B');
    bound_20.set_T(250.0);
    bound_20.set_a(1.5);

    original.add_node(diff_10);
    original.add_node(bound_20);

    auto require_state = [](Nodes& nodes) {
        REQUIRE(nodes.num_nodes() == 2);
        REQUIRE(nodes.get_type(10) == 'D');
        REQUIRE(nodes.get_type(20) == 'B');
        REQUIRE(nodes.get_T(10) == 300.0);
        REQUIRE(nodes.get_qi(10) == 4.0);
        REQUIRE(nodes.get_T(20) == 250.0);
        REQUIRE(nodes.get_a(20) == 1.5);
        REQUIRE(nodes.get_idx_from_node_num(10) == std::optional<Index>{0});
        REQUIRE(nodes.get_idx_from_node_num(20) == std::optional<Index>{1});
    };

    Nodes copied(original);
    require_state(copied);

    Nodes assigned;
    assigned = original;
    require_state(assigned);

    Nodes moved(std::move(copied));
    require_state(moved);

    Nodes move_assigned;
    move_assigned = std::move(assigned);
    require_state(move_assigned);
}

TEST_CASE("Nodes set_type currently reports success without reordering nodes",
          "[nodes]") {
    Nodes tns;

    Node node(10);
    tns.add_node(node);

    // TODO: Replace this with real D<->B conversion assertions once
    // diffusive_to_boundary/boundary_to_diffusive are implemented.
    REQUIRE(tns.set_type(10, 'B'));
    REQUIRE(tns.get_type(10) == 'D');
    REQUIRE(tns.get_idx_from_node_num(10) == std::optional<Index>{0});
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
    nodes_vector.reserve(to_sizet(N));
    nodes_vector_copy.reserve(to_sizet(N));

    // Internal order vector
    std::vector<Index> internal_order;
    internal_order.reserve(to_sizet(N));
    Index diff_index = 0;

    for (Index i = 0; i < N; i++) {
        const NodeNum usr_num = static_cast<NodeNum>(num_nodes[to_sizet(i)]);
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
        const NodeNum usr_num =
            static_cast<NodeNum>(insertion_order[to_sizet(i)]);
        auto it =
            std::find(insertion_order.begin(), insertion_order.end(), usr_num);
        if (it != insertion_order.end()) {
            const Index node_ix = it - insertion_order.begin();
            tns.add_node(nodes_vector[to_sizet(node_ix)]);
        }
    }
    // Assert all nodes have been inserted
    REQUIRE(tns.num_nodes() == N);

    // Assert that tns have the values of the nodes stored
    for (std::size_t i = 0; i < to_sizet(N); i++) {
        assert_tn_has_same_values_as_tns(nodes_vector_copy[i], tns,
                                         /*check_internal=*/false);
    }

    // Assert that the nodes added to tns return the same values as tns
    for (std::size_t i = 0; i < to_sizet(N); i++) {
        assert_tn_has_same_values_as_tns(nodes_vector[i], tns,
                                         /*check_internal=*/true);
    }

    // Assert internal order is correct
    for (std::size_t i = 0; i < to_sizet(N); i++) {
        const auto int_ix = to_sizet(internal_order[i]);
        const auto node_num = tns.get_node_num_from_idx(to_idx(i));
        REQUIRE(node_num.has_value());
        REQUIRE(nodes_vector_copy[int_ix].get_node_num() == *node_num);
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    // Assert temperatures and capacities are contiguous in memory
    const double* temperatures_vector = tns.T_vector.data();
    const double* capacities_vector = tns.C_vector.data();
    for (std::size_t i = 0; i < to_sizet(N); i++) {
        const auto int_ix = to_sizet(internal_order[i]);
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
    REQUIRE(!tns.is_mapped());
    tns.set_T(1001, 1001.0);  // Map updated
    REQUIRE(tns.is_mapped());

    tns.add_node(node_map_check_b1);  // Map flagged outdated
    REQUIRE(!tns.is_mapped());
    tns.set_T(1003, 1003.0);  // Map updated
    REQUIRE(tns.is_mapped());

    tns.add_node(node_map_check_d2);  // Map flagged outdated
    REQUIRE(!tns.is_mapped());
    tns.set_T(1002, 1002.0);  // Map updated
    REQUIRE(tns.is_mapped());

    tns.add_node(node_map_check_b2);  // Map flagged outdated
    REQUIRE(!tns.is_mapped());
    tns.set_T(1004, 1004.0);  // Map updated
    REQUIRE(tns.is_mapped());

    // Additional tests can be added here...
}

// NOLINTEND(readability-function-cognitive-complexity,
// bugprone-chained-comparison)
