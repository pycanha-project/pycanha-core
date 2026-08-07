#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

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
        spdlog::drop("pycanha");
    }
};

/// Saves and restores the process-wide logging configuration, so a test that
/// redirects the log file or moves a threshold cannot leak into the next one.
class LogConfigGuard {
  public:
    LogConfigGuard()
        : _directory(pycanha::log_directory()),
          _record(pycanha::record_level()),
          _display(pycanha::display_level()),
          _file_output(pycanha::file_output()) {}

    ~LogConfigGuard() {
        pycanha::set_record_level(_record);
        pycanha::set_display_level(_display);
        pycanha::set_file_output(_file_output);
        // Restoring the directory releases the file the test was writing to,
        // which has to happen before the scratch tree can be removed.
        pycanha::set_log_directory(_directory);
        if (!_scratch.empty()) {
            std::error_code ignored;
            std::filesystem::remove_all(_scratch, ignored);
        }
    }

    /// A path to delete once the configuration has been restored.
    void clean_up(std::filesystem::path scratch) {
        _scratch = std::move(scratch);
    }

    LogConfigGuard(const LogConfigGuard&) = delete;
    LogConfigGuard& operator=(const LogConfigGuard&) = delete;
    LogConfigGuard(LogConfigGuard&&) = delete;
    LogConfigGuard& operator=(LogConfigGuard&&) = delete;

  private:
    std::filesystem::path _directory;
    spdlog::level::level_enum _record;
    spdlog::level::level_enum _display;
    bool _file_output;
    std::filesystem::path _scratch;
};

/// A directory path under the system temp dir that does not exist yet, so a
/// test can assert that nothing creates it until a record is written.
[[nodiscard]] std::filesystem::path unused_log_directory(
    const std::string& tag) {
    static std::size_t counter = 0;
    ++counter;
    const auto path = std::filesystem::temp_directory_path() /
                      ("pycanha-log-" + tag + "-" + std::to_string(counter));
    std::filesystem::remove_all(path);
    return path;
}

[[nodiscard]] std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

[[nodiscard]] std::size_t count_occurrences(const std::string& haystack,
                                            const std::string& needle) {
    std::size_t count = 0;
    for (auto pos = haystack.find(needle); pos != std::string::npos;
         pos = haystack.find(needle, pos + needle.size())) {
        ++count;
    }
    return count;
}

}  // namespace

TEST_CASE("ostream logger emits formatted messages", "[utils][logger]") {
    std::ostringstream output;
    auto logger = pycanha::create_ostream_logger("logger-test", output);

    logger->info("logger smoke test {}", 23);
    logger->flush();

    REQUIRE(output.str().contains("logger smoke test 23"));
}

TEST_CASE("python logger shares main logger sinks", "[utils][logger]") {
    auto main_logger = pycanha::get_logger();
    auto python_logger = pycanha::get_python_logger();

    REQUIRE(python_logger->name() == "pycanha");
    REQUIRE(python_logger->sinks().size() == main_logger->sinks().size());
    REQUIRE_FALSE(python_logger->sinks().empty());

    for (std::size_t index = 0; index < main_logger->sinks().size(); ++index) {
        REQUIRE(python_logger->sinks()[index].get() ==
                main_logger->sinks()[index].get());
    }
}

TEST_CASE("record level drives both loggers", "[utils][logger]") {
    const LogConfigGuard log_config_guard;

    pycanha::set_record_level(spdlog::level::err);

    REQUIRE(pycanha::record_level() == spdlog::level::err);
    REQUIRE(pycanha::get_logger()->level() == spdlog::level::err);
    REQUIRE(pycanha::get_python_logger()->level() == spdlog::level::err);
}

TEST_CASE("display and record thresholds move independently",
          "[utils][logger]") {
    const LogConfigGuard log_config_guard;

    pycanha::set_record_level(spdlog::level::info);
    pycanha::set_display_level(spdlog::level::warn);

    REQUIRE(pycanha::record_level() == spdlog::level::info);
    REQUIRE(pycanha::display_level() == spdlog::level::warn);

    // Silencing the console must leave the record threshold untouched.
    pycanha::set_display_level(spdlog::level::off);

    REQUIRE(pycanha::display_level() == spdlog::level::off);
    REQUIRE(pycanha::record_level() == spdlog::level::info);
}

TEST_CASE("thresholds reject levels compiled away by SPDLOG_ACTIVE_LEVEL",
          "[utils][logger]") {
    const LogConfigGuard log_config_guard;

#if SPDLOG_ACTIVE_LEVEL > SPDLOG_LEVEL_TRACE
    REQUIRE_THROWS_AS(pycanha::set_record_level(spdlog::level::trace),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(pycanha::set_display_level(spdlog::level::trace),
                      std::invalid_argument);
#else
    REQUIRE_NOTHROW(pycanha::set_record_level(spdlog::level::trace));
    REQUIRE_NOTHROW(pycanha::set_display_level(spdlog::level::trace));
#endif

#if SPDLOG_ACTIVE_LEVEL > SPDLOG_LEVEL_DEBUG
    REQUIRE_THROWS_AS(pycanha::set_record_level(spdlog::level::debug),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(pycanha::set_display_level(spdlog::level::debug),
                      std::invalid_argument);
#else
    REQUIRE_NOTHROW(pycanha::set_record_level(spdlog::level::debug));
    REQUIRE_NOTHROW(pycanha::set_display_level(spdlog::level::debug));
#endif
}

TEST_CASE("a logger registered under the library name receives the records",
          "[utils][logger]") {
    // Tests elsewhere capture library output by putting their own logger in
    // the registry under this name, so the substitution has to keep working.
    const LoggerRegistryGuard logger_registry_guard;
    std::ostringstream output;
    auto substitute = pycanha::create_ostream_logger("pycanha-core", output);

    spdlog::register_logger(substitute);

    pycanha::get_logger()->info("captured by the substitute");
    pycanha::get_logger()->flush();

    REQUIRE(output.str().contains("captured by the substitute"));
}

TEST_CASE("log file is created by the first record, not at startup",
          "[utils][logger]") {
    LogConfigGuard log_config_guard;
    const auto directory = unused_log_directory("lazy");
    log_config_guard.clean_up(directory);

    pycanha::set_file_output(/*enabled=*/true);
    pycanha::set_record_level(spdlog::level::info);
    pycanha::set_log_directory(directory);

    // Obtaining the logger must not be enough to touch the filesystem.
    auto logger = pycanha::get_logger();
    REQUIRE(pycanha::current_log_file().empty());
    REQUIRE_FALSE(std::filesystem::exists(directory));

    logger->info("first record");
    pycanha::flush();

    const auto log_file = pycanha::current_log_file();
    REQUIRE_FALSE(log_file.empty());
    REQUIRE(std::filesystem::exists(log_file));
    REQUIRE(read_file(log_file).contains("first record"));
}

TEST_CASE("records at or above the record level reach the file",
          "[utils][logger]") {
    LogConfigGuard log_config_guard;
    const auto directory = unused_log_directory("levels");
    log_config_guard.clean_up(directory);

    pycanha::set_file_output(/*enabled=*/true);
    pycanha::set_record_level(spdlog::level::info);
    // A quiet console must not stop the file from getting the info trail.
    pycanha::set_display_level(spdlog::level::off);
    pycanha::set_log_directory(directory);

    auto logger = pycanha::get_logger();
    logger->debug("below the record threshold");
    logger->info("at the record threshold");
    logger->warn("above the record threshold");
    pycanha::flush();

    const std::string content = read_file(pycanha::current_log_file());
    REQUIRE(content.contains("at the record threshold"));
    REQUIRE(content.contains("above the record threshold"));
    REQUIRE_FALSE(content.contains("below the record threshold"));
}

TEST_CASE("file output can be switched off entirely", "[utils][logger]") {
    LogConfigGuard log_config_guard;
    const auto directory = unused_log_directory("nofile");
    log_config_guard.clean_up(directory);

    pycanha::set_record_level(spdlog::level::info);
    pycanha::set_log_directory(directory);
    pycanha::set_file_output(/*enabled=*/false);

    REQUIRE_FALSE(pycanha::file_output());

    pycanha::get_logger()->info("goes nowhere near the disk");
    pycanha::flush();

    REQUIRE(pycanha::current_log_file().empty());
    REQUIRE_FALSE(std::filesystem::exists(directory));
}

TEST_CASE("an unusable log directory degrades instead of throwing",
          "[utils][logger]") {
    LogConfigGuard log_config_guard;

    // A directory cannot be created below a regular file on any platform, so
    // this exercises the same failure path as an unwritable logs/ without
    // depending on how a platform enforces directory permissions.
    const auto blocker = unused_log_directory("blocked");
    {
        std::ofstream blocking_file(blocker);
        blocking_file << "not a directory";
    }
    log_config_guard.clean_up(blocker);

    pycanha::set_file_output(/*enabled=*/true);
    pycanha::set_record_level(spdlog::level::info);
    pycanha::set_log_directory(blocker / "logs");

    REQUIRE_NOTHROW(pycanha::get_logger()->info("nowhere to put this"));
    REQUIRE_NOTHROW(pycanha::flush());
    REQUIRE(pycanha::current_log_file().empty());
}

TEST_CASE("the run banner is written once for the file", "[utils][logger]") {
    LogConfigGuard log_config_guard;
    const auto directory = unused_log_directory("banner");
    log_config_guard.clean_up(directory);

    pycanha::set_file_output(/*enabled=*/true);
    pycanha::set_record_level(spdlog::level::info);
    pycanha::set_log_directory(directory);

    auto logger = pycanha::get_logger();
    logger->info("one");
    logger->info("two");
    logger->info("three");
    pycanha::flush();

    const std::string content = read_file(pycanha::current_log_file());
    REQUIRE(count_occurrences(content, "=== pycanha run") == 1);
    REQUIRE(content.contains("pid "));
}

TEST_CASE("both origins share one log file", "[utils][logger]") {
    LogConfigGuard log_config_guard;
    const auto directory = unused_log_directory("origins");
    log_config_guard.clean_up(directory);

    pycanha::set_file_output(/*enabled=*/true);
    pycanha::set_record_level(spdlog::level::info);
    pycanha::set_log_directory(directory);

    pycanha::get_logger()->info("from core");
    pycanha::get_python_logger()->info("from layer three");
    pycanha::flush();

    const std::string content = read_file(pycanha::current_log_file());
    REQUIRE(content.contains("[pycanha-core] [info] from core"));
    REQUIRE(content.contains("[pycanha] [info] from layer three"));
}
