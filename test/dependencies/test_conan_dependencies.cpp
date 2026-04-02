#include <hdf5/H5public.h>
#include <symengine/add.h>
#include <symengine/basic.h>
#include <symengine/integer.h>
#include <symengine/symbol.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("HDF5 basic lifecycle works", "[deps][hdf5]") {
    unsigned major = 0;
    unsigned minor = 0;
    unsigned release = 0;

    REQUIRE(H5open() >= 0);
    REQUIRE(H5get_libversion(&major, &minor, &release) >= 0);
}

// Clang analyzer reports a use-after-free inside SymEngine's intrusive
// reference counting in unevaluated Catch2 expression paths. The smoke test is
// intentionally simple and only verifies the dependency is wired correctly.
// NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)
TEST_CASE("SymEngine builds a simple expression", "[deps][symengine]") {
    const auto x = SymEngine::symbol("x");
    const auto one = SymEngine::integer(1);
    const auto expr = SymEngine::add(x, one);
    const auto expected_expr = SymEngine::add(x, one);

    REQUIRE(SymEngine::eq(*expr, *expected_expr));
}
// NOLINTEND(clang-analyzer-cplusplus.NewDelete)
