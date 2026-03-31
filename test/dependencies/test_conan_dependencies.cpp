#include <catch2/catch_test_macros.hpp>

#include <hdf5.h>

#include <symengine/add.h>
#include <symengine/basic.h>
#include <symengine/integer.h>
#include <symengine/symbol.h>

TEST_CASE("HDF5 basic lifecycle works", "[deps][hdf5]") {
    REQUIRE(H5open() >= 0);
    REQUIRE(H5close() >= 0);
}

TEST_CASE("SymEngine builds a simple expression", "[deps][symengine]") {
    const auto x = SymEngine::symbol("x");
    const auto one = SymEngine::integer(1);
    const auto expr = SymEngine::add(x, one);

    REQUIRE(SymEngine::eq(*expr, *SymEngine::add(x, one)));
}
