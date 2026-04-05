#pragma once

// SPDLOG_ACTIVE_LEVEL must be defined before any spdlog include so that
// compile-time level stripping macros (SPDLOG_LOGGER_TRACE, etc.) work
// correctly. The actual value is set via CMake compile definitions.
// Default behaviour is TRACE in Debug and INFO otherwise, and it can be
// overridden with PYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE.

#include <spdlog/common.h>
#include <spdlog/logger.h>

#include <memory>
#include <ostream>
#include <string_view>

namespace pycanha {

/// Main library logger ("pycanha-core").
/// Default level: info. Pattern: [HH:MM:SS.mmm] [logger-name] [level] message
std::shared_ptr<spdlog::logger> get_logger();

/// Profiling logger ("pycanha-core.profiling").
/// Pattern: [HH:MM:SS.mmm] [profiling] message
std::shared_ptr<spdlog::logger> get_profiling_logger();

/// Python-facing logger ("pycanha-python") sharing the main logger sinks.
/// Default level: inherits the main logger level at creation time.
/// Pattern: [HH:MM:SS.mmm] [logger-name] [level] message
std::shared_ptr<spdlog::logger> get_python_logger();

/// Create a logger that writes to the given ostream (useful for testing).
std::shared_ptr<spdlog::logger> create_ostream_logger(
    std::string_view name, std::ostream& stream,
    spdlog::level::level_enum level = spdlog::level::info);

/// Change the main logger level at runtime.
/// Throws std::invalid_argument if the requested level is more verbose than
/// the compile-time SPDLOG_ACTIVE_LEVEL.
void set_logger_level(spdlog::level::level_enum level);

/// Change the Python logger level at runtime.
/// Throws std::invalid_argument if the requested level is more verbose than
/// the compile-time SPDLOG_ACTIVE_LEVEL.
void set_python_logger_level(spdlog::level::level_enum level);

}  // namespace pycanha
