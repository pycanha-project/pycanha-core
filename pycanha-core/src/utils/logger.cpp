#include "pycanha-core/utils/logger.hpp"

#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pycanha {

namespace {

constexpr auto k_unified_log_pattern = "[%H:%M:%S.%e] [%n] [%^%l%$] %v";
constexpr auto k_profiling_log_pattern = "[%H:%M:%S.%e] [profiling] %v";

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
constexpr auto k_compiled_active_log_level = spdlog::level::trace;
#elif SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
constexpr auto k_compiled_active_log_level = spdlog::level::debug;
#elif SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
constexpr auto k_compiled_active_log_level = spdlog::level::info;
#elif SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN
constexpr auto k_compiled_active_log_level = spdlog::level::warn;
#elif SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR
constexpr auto k_compiled_active_log_level = spdlog::level::err;
#elif SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_CRITICAL
constexpr auto k_compiled_active_log_level = spdlog::level::critical;
#else
constexpr auto k_compiled_active_log_level = spdlog::level::off;
#endif

std::string_view level_to_string(const spdlog::level::level_enum level) {
    switch (level) {
        case spdlog::level::trace:
            return "trace";
        case spdlog::level::debug:
            return "debug";
        case spdlog::level::info:
            return "info";
        case spdlog::level::warn:
            return "warn";
        case spdlog::level::err:
            return "error";
        case spdlog::level::critical:
            return "critical";
        case spdlog::level::off:
            return "off";
        default:
            return "unknown";
    }
}

void validate_general_logger_level(const std::string_view logger_name,
                                   const spdlog::level::level_enum level) {
    if (level < k_compiled_active_log_level) {
        throw std::invalid_argument(
            "Cannot set logger '" + std::string{logger_name} + "' to level '" +
            std::string{level_to_string(level)} +
            "' because SPDLOG_ACTIVE_LEVEL='" +
            std::string{level_to_string(k_compiled_active_log_level)} +
            "' compiled more verbose logs away");
    }
}

}  // namespace

std::shared_ptr<spdlog::logger> get_logger() {
    auto logger = spdlog::get("pycanha-core");
    if (!logger) {
        logger = spdlog::stdout_color_mt("pycanha-core");
        logger->set_pattern(k_unified_log_pattern);
        logger->set_level(spdlog::level::info);
    }

    return logger;
}

std::shared_ptr<spdlog::logger> get_profiling_logger() {
    auto logger = spdlog::get("pycanha-core.profiling");
    if (!logger) {
        logger = spdlog::stdout_color_mt("pycanha-core.profiling");
        logger->set_pattern(k_profiling_log_pattern);
        logger->set_level(spdlog::level::info);
    }

    return logger;
}

std::shared_ptr<spdlog::logger> get_python_logger() {
    auto logger = spdlog::get("pycanha-python");
    if (!logger) {
        auto main_logger = get_logger();
        logger = std::make_shared<spdlog::logger>("pycanha-python",
                                                  main_logger->sinks().begin(),
                                                  main_logger->sinks().end());
        logger->set_pattern(k_unified_log_pattern);
        logger->set_level(main_logger->level());
        spdlog::register_logger(logger);
    }

    return logger;
}

std::shared_ptr<spdlog::logger> create_ostream_logger(
    std::string_view name, std::ostream& stream,
    const spdlog::level::level_enum level) {
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(stream, true);
    auto logger =
        std::make_shared<spdlog::logger>(std::string{name}, std::move(sink));
    logger->set_pattern("%v");
    logger->set_level(level);

    return logger;
}

void set_logger_level(const spdlog::level::level_enum level) {
    validate_general_logger_level("pycanha-core", level);
    get_logger()->set_level(level);
}

void set_python_logger_level(const spdlog::level::level_enum level) {
    validate_general_logger_level("pycanha-python", level);
    get_python_logger()->set_level(level);
}

}  // namespace pycanha
