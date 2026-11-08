#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <thread>

#include "pycanha-core/utils/threads.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("The default thread count is the machine's physical cores",
          "[threads]") {
    const auto logical =
        std::max(static_cast<int>(std::thread::hardware_concurrency()), 1);
    REQUIRE(physical_cores() >= 1);
    REQUIRE(physical_cores() <= logical);
    REQUIRE(default_threads() >= 1);
    REQUIRE(threads() == default_threads());
}

TEST_CASE("The global thread count can be set and restored", "[threads]") {
    set_threads(3);
    REQUIRE(threads() == 3);
    set_threads(0);
    REQUIRE(threads() == default_threads());
    REQUIRE_THROWS_AS(set_threads(-1), std::invalid_argument);
    REQUIRE(threads() == default_threads());
}
