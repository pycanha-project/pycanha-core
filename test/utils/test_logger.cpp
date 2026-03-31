#include <sstream>

#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/utils/logger.hpp"

TEST_CASE("ostream logger emits formatted messages", "[utils][logger]") {
    std::ostringstream output;
    auto logger =
        pycanha::create_ostream_logger("logger-test", output, spdlog::level::trace);

    logger->info("logger smoke test {}", 23);
    logger->flush();

    REQUIRE(output.str().contains("logger smoke test 23"));
}