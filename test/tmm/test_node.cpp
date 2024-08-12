#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <string>

#include "pycanha-core/tmm/node.hpp"

void assert_node_default_values(Node &tn) {
    // Test default values for tn
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
}

void assert_nodes_have_same_attribute_values_except_usr_number(Node &tn1,
                                                               Node &tn2) {
    // Test attributes are equal
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

    // Literal assertion. string1.compare(string2) returns 0 if strings are
    // equal
    REQUIRE(tn1.get_literal_C().compare(tn2.get_literal_C()) == 0);
}

void assert_nodes_have_same_attribute_values(Node &tn1, Node &tn2) {
    // Test usr node number is the same
    REQUIRE(tn1.get_node_num() == tn2.get_node_num());

    // Test the other attributes
    assert_nodes_have_same_attribute_values_except_usr_number(tn1, tn2);
}

TEST_CASE("Node Default Values", "[node]") {
    int usr_num = 5;
    Node tn(usr_num);

    // Test usr_num supplied is the same as the stored one
    REQUIRE(usr_num == tn.get_node_num());

    // For Changing default values and testing the new values has been stored
    usr_num = 9;
    char type = 'B';
    double T = 1.3;
    double C = 2.3;
    double qs = 3.3;
    double qa = 4.3;
    double qe = 5.3;
    double qi = 6.3;
    double qr = 7.3;
    double a = 8.3;
    double fx = 9.3;
    double fy = 10.3;
    double fz = 11.3;
    double eps = 12.3;
    double aph = 13.3;
    std::string literal_C = "7e3*5.0/2+2.1";

    // Test a blank node has default values
    assert_node_default_values(tn);

    // Test that getting values from a tn does not change the attributes
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
            tn2.set_literal_C(literal_C);

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

            // Literal assertion. string1.compare(string2) returns 0 if strings
            // are equal
            REQUIRE(literal_C.compare(tn2.get_literal_C()) == 0);

            // Check the copied node is a copy, so the first one still holds the
            // old values
            assert_node_default_values(tn);

            // Check if copy assignment works
            tn = tn2;
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
            REQUIRE(literal_C.compare(tn.get_literal_C()) == 0);
        }

        // tn2 object is destroyed. check that tn still holds the values
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
        REQUIRE(literal_C.compare(tn.get_literal_C()) == 0);
    }

    // TODO test the rest of the methods.
}
