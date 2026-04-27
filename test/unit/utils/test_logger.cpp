#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <sstream>
#include <string>

#include "pycanha-core/utils/logger.hpp"

namespace {

class LoggerRegistryGuard {
  public:
    LoggerRegistryGuard() { drop_test_loggers(); }

    ~LoggerRegistryGuard() { drop_test_loggers(); }

    LoggerRegistryGuard(const LoggerRegistryGuard&) = delete;
    LoggerRegistryGuard& operator=(const LoggerRegistryGuard&) = delete;
    LoggerRegistryGuard(LoggerRegistryGuard&&) = delete;
    LoggerRegistryGuard& operator=(LoggerRegistryGuard&&) = delete;

  private:
    static void drop_test_loggers() {
        spdlog::drop("pycanha-core");
        spdlog::drop("pycanha-core.profiling");
        spdlog::drop("pycanha-python");
    }
};

}  // namespace

TEST_CASE("ostream logger emits formatted messages", "[utils][logger]") {
    std::ostringstream output;
    auto logger = pycanha::create_ostream_logger("logger-test", output);

    logger->info("logger smoke test {}", 23);
    logger->flush();

    REQUIRE(output.str().contains("logger smoke test 23"));
}

TEST_CASE("python logger shares main logger sinks", "[utils][logger]") {
    const LoggerRegistryGuard logger_registry_guard;
    std::ostringstream output;
    auto main_logger = pycanha::create_ostream_logger("pycanha-core", output);

    spdlog::register_logger(main_logger);

    auto python_logger = pycanha::get_python_logger();

    REQUIRE(python_logger->name() == "pycanha-python");
    REQUIRE(python_logger->sinks().size() == main_logger->sinks().size());
    REQUIRE_FALSE(python_logger->sinks().empty());

    for (std::size_t index = 0; index < main_logger->sinks().size(); ++index) {
        REQUIRE(python_logger->sinks()[index].get() ==
                main_logger->sinks()[index].get());
    }
}

TEST_CASE("python logger level is configurable at runtime", "[utils][logger]") {
    const LoggerRegistryGuard logger_registry_guard;
    std::ostringstream output;
    auto main_logger = pycanha::create_ostream_logger("pycanha-core", output,
                                                      spdlog::level::warn);

    spdlog::register_logger(main_logger);

    auto python_logger = pycanha::get_python_logger();

    REQUIRE(python_logger->level() == spdlog::level::warn);

    pycanha::set_python_logger_level(spdlog::level::err);

    REQUIRE(python_logger->level() == spdlog::level::err);
}

TEST_CASE("general loggers reject levels compiled away by SPDLOG_ACTIVE_LEVEL",
          "[utils][logger]") {
    const LoggerRegistryGuard logger_registry_guard;
    std::ostringstream output;
    auto main_logger = pycanha::create_ostream_logger("pycanha-core", output);

    spdlog::register_logger(main_logger);
    auto python_logger = pycanha::get_python_logger();

#if SPDLOG_ACTIVE_LEVEL > SPDLOG_LEVEL_TRACE
    REQUIRE_THROWS_AS(pycanha::set_logger_level(spdlog::level::trace),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(pycanha::set_python_logger_level(spdlog::level::trace),
                      std::invalid_argument);
#else
    REQUIRE_NOTHROW(pycanha::set_logger_level(spdlog::level::trace));
    REQUIRE_NOTHROW(pycanha::set_python_logger_level(spdlog::level::trace));
#endif

#if SPDLOG_ACTIVE_LEVEL > SPDLOG_LEVEL_DEBUG
    REQUIRE_THROWS_AS(pycanha::set_logger_level(spdlog::level::debug),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(pycanha::set_python_logger_level(spdlog::level::debug),
                      std::invalid_argument);
#else
    REQUIRE_NOTHROW(pycanha::set_logger_level(spdlog::level::debug));
    REQUIRE_NOTHROW(pycanha::set_python_logger_level(spdlog::level::debug));
#endif

    REQUIRE(python_logger != nullptr);
}

TEST_CASE("main and python loggers emit into the same ostream",
          "[utils][logger]") {
    const LoggerRegistryGuard logger_registry_guard;
    std::ostringstream output;
    auto main_logger = pycanha::create_ostream_logger("pycanha-core", output);

    spdlog::register_logger(main_logger);

    auto python_logger = pycanha::get_python_logger();

    main_logger->info("from core");
    python_logger->info("from python");
    main_logger->flush();
    python_logger->flush();

    const std::string content = output.str();
    REQUIRE(content.contains("[pycanha-core] [info] from core"));
    REQUIRE(content.contains("[pycanha-python] [info] from python"));
}
