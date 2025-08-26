#include <catch2/catch_test_macros.hpp>
#include <string>

#include "pycanha-core/tmm/node.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

// NOLINTBEGIN(readability-function-cognitive-complexity)

void assert_node_default_values(Node& tn) {
    // Suppress false positives from Catch2 expression decomposition.
    // NOLINTBEGIN(bugprone-chained-comparison)
    REQUIRE(tn.get_int_node_num() == -1);
    REQUIRE(tn.get_type() == 'D');
    REQUIRE(tn.get_T() == 0.0);
    REQUIRE(tn.get_C() == 0.0);
    REQUIRE(tn.get_qs() == 0.0);
    REQUIRE(tn.get_qa() == 0.0);
    REQUIRE(tn.get_qe() == 0.0);
    REQUIRE(tn.get_qi() == 0.0);
    REQUIRE(tn.get_qr() == 0.0);
    REQUIRE(tn.get_a() == 0.0);
    REQUIRE(tn.get_fx() == 0.0);
    REQUIRE(tn.get_fy() == 0.0);
    REQUIRE(tn.get_fz() == 0.0);
    REQUIRE(tn.get_eps() == 0.0);
    REQUIRE(tn.get_aph() == 0.0);

    // Literals
    REQUIRE(tn.get_literal_C().empty());
    // NOLINTEND(bugprone-chained-comparison)
}

void assert_nodes_have_same_attribute_values_except_usr_number(Node& tn1,
                                                               Node& tn2) {
    // NOLINTBEGIN(bugprone-chained-comparison)
    REQUIRE(tn1.get_int_node_num() == tn2.get_int_node_num());
    REQUIRE(tn1.get_type() == tn2.get_type());
    REQUIRE(tn1.get_T() == tn2.get_T());
    REQUIRE(tn1.get_C() == tn2.get_C());
    REQUIRE(tn1.get_qs() == tn2.get_qs());
    REQUIRE(tn1.get_qa() == tn2.get_qa());
    REQUIRE(tn1.get_qe() == tn2.get_qe());
    REQUIRE(tn1.get_qi() == tn2.get_qi());
    REQUIRE(tn1.get_qr() == tn2.get_qr());
    REQUIRE(tn1.get_a() == tn2.get_a());
    REQUIRE(tn1.get_fx() == tn2.get_fx());
    REQUIRE(tn1.get_fy() == tn2.get_fy());
    REQUIRE(tn1.get_fz() == tn2.get_fz());
    REQUIRE(tn1.get_eps() == tn2.get_eps());
    REQUIRE(tn1.get_aph() == tn2.get_aph());
    REQUIRE(tn1.get_literal_C() == tn2.get_literal_C());
    // NOLINTEND(bugprone-chained-comparison)
}

void assert_nodes_have_same_attribute_values(Node& tn1, Node& tn2) {
    // NOLINTBEGIN(bugprone-chained-comparison)
    REQUIRE(tn1.get_node_num() == tn2.get_node_num());
    // NOLINTEND(bugprone-chained-comparison)

    assert_nodes_have_same_attribute_values_except_usr_number(tn1, tn2);
}

TEST_CASE("Node Default Values", "[node]") {
    int usr_num = 5;
    Node tn(usr_num);

    // NOLINTBEGIN(bugprone-chained-comparison)
    REQUIRE(usr_num == tn.get_node_num());
    // NOLINTEND(bugprone-chained-comparison)

    // For Changing default values and testing the new values has been stored
    usr_num = 9;
    const char type = 'B';
    const double T = 1.3;  // NOLINT(readability-identifier-naming)
    const double C = 2.3;  // NOLINT(readability-identifier-naming)
    const double qs = 3.3;
    const double qa = 4.3;
    const double qe = 5.3;
    const double qi = 6.3;
    const double qr = 7.3;
    const double a = 8.3;
    const double fx = 9.3;
    const double fy = 10.3;
    const double fz = 11.3;
    const double eps = 12.3;
    const double aph = 13.3;
    std::string literal_c = "7e3*5.0/2+2.1";

    assert_node_default_values(tn);
    assert_node_default_values(tn);

    SECTION(
        "Copy node and test that it has the same values as the copied one") {
        {
            Node tn2 = tn;
            assert_node_default_values(tn);
            assert_nodes_have_same_attribute_values(tn, tn2);
            assert_node_default_values(tn2);

            // Change values
            tn2.set_node_num(usr_num);
            tn2.set_type(type);
            tn2.set_T(T);
            tn2.set_C(C);
            tn2.set_qs(qs);
            tn2.set_qa(qa);
            tn2.set_qe(qe);
            tn2.set_qi(qi);
            tn2.set_qr(qr);
            tn2.set_a(a);
            tn2.set_fx(fx);
            tn2.set_fy(fy);
            tn2.set_fz(fz);
            tn2.set_eps(eps);
            tn2.set_aph(aph);
            tn2.set_literal_C(literal_c);

            // NOLINTBEGIN(bugprone-chained-comparison)
            REQUIRE(usr_num == tn2.get_node_num());
            REQUIRE(tn2.get_int_node_num() == -1);
            REQUIRE(tn2.get_type() == type);
            REQUIRE(tn2.get_T() == T);
            REQUIRE(tn2.get_C() == C);
            REQUIRE(tn2.get_qs() == qs);
            REQUIRE(tn2.get_qa() == qa);
            REQUIRE(tn2.get_qe() == qe);
            REQUIRE(tn2.get_qi() == qi);
            REQUIRE(tn2.get_qr() == qr);
            REQUIRE(tn2.get_a() == a);
            REQUIRE(tn2.get_fx() == fx);
            REQUIRE(tn2.get_fy() == fy);
            REQUIRE(tn2.get_fz() == fz);
            REQUIRE(tn2.get_eps() == eps);
            REQUIRE(tn2.get_aph() == aph);
            REQUIRE(literal_c == tn2.get_literal_C());
            // NOLINTEND(bugprone-chained-comparison)

            // Check the copied node is a copy, so tn still holds old values
            assert_node_default_values(tn);

            // Check if copy assignment works
            tn = tn2;

            // NOLINTBEGIN(bugprone-chained-comparison)
            REQUIRE(usr_num == tn.get_node_num());
            REQUIRE(tn.get_int_node_num() == -1);
            REQUIRE(tn.get_type() == type);
            REQUIRE(tn.get_T() == T);
            REQUIRE(tn.get_C() == C);
            REQUIRE(tn.get_qs() == qs);
            REQUIRE(tn.get_qa() == qa);
            REQUIRE(tn.get_qe() == qe);
            REQUIRE(tn.get_qi() == qi);
            REQUIRE(tn.get_qr() == qr);
            REQUIRE(tn.get_a() == a);
            REQUIRE(tn.get_fx() == fx);
            REQUIRE(tn.get_fy() == fy);
            REQUIRE(tn.get_fz() == fz);
            REQUIRE(tn.get_eps() == eps);
            REQUIRE(tn.get_aph() == aph);
            REQUIRE(literal_c == tn.get_literal_C());
            // NOLINTEND(bugprone-chained-comparison)
        }

        // tn2 destroyed: tn should still hold the values
        // NOLINTBEGIN(bugprone-chained-comparison)
        REQUIRE(usr_num == tn.get_node_num());
        REQUIRE(tn.get_int_node_num() == -1);
        REQUIRE(tn.get_type() == type);
        REQUIRE(tn.get_T() == T);
        REQUIRE(tn.get_C() == C);
        REQUIRE(tn.get_qs() == qs);
        REQUIRE(tn.get_qa() == qa);
        REQUIRE(tn.get_qe() == qe);
        REQUIRE(tn.get_qi() == qi);
        REQUIRE(tn.get_qr() == qr);
        REQUIRE(tn.get_a() == a);
        REQUIRE(tn.get_fx() == fx);
        REQUIRE(tn.get_fy() == fy);
        REQUIRE(tn.get_fz() == fz);
        REQUIRE(tn.get_eps() == eps);
        REQUIRE(tn.get_aph() == aph);
        REQUIRE(literal_c == tn.get_literal_C());
        // NOLINTEND(bugprone-chained-comparison)
    }

    // TODO: test the rest of the methods.
}

// NOLINTEND(readability-function-cognitive-complexity)
