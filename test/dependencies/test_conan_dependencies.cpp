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

TEST_CASE("SymEngine builds a simple expression", "[deps][symengine]") {
    // False positive from clang static analyzer in SymEngine reference
    // counting. NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)
    const auto x = SymEngine::symbol("x");
    const auto one = SymEngine::integer(1);
    const auto expr = SymEngine::add(x, one);

    REQUIRE(SymEngine::eq(*expr, *SymEngine::add(x, one)));
    // NOLINTEND(clang-analyzer-cplusplus.NewDelete)
}
