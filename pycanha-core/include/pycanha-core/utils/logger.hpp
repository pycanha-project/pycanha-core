#pragma once

// SPDLOG_ACTIVE_LEVEL must be defined before any spdlog include so that
// compile-time level stripping macros (SPDLOG_LOGGER_TRACE, etc.) work
// correctly. The actual value is set via CMake compile definitions.
// Default behaviour is TRACE in Debug and INFO otherwise, and it can be
// overridden with PYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE.
//
// Future Python sink integration:
//   In pycanha-core-python, create a C++ class subclassing
//   spdlog::sinks::base_sink<std::mutex> whose sink_it_() acquires the GIL
//   and calls a Python logging callback (e.g. Loguru).  Then call
//   set_main_logger_sink() / set_profiling_logger_sink() from the binding
//   init code to inject the Python-side sinks.

#include <spdlog/common.h>
#include <spdlog/logger.h>

#include <memory>
#include <ostream>
#include <string_view>

namespace pycanha {

/// Main library logger ("pycanha-core").
/// Default level: info.  Pattern: [HH:MM:SS.mmm] [level] message
std::shared_ptr<spdlog::logger> get_logger();

/// Profiling logger ("pycanha-core.profiling").
/// Default level: info when PYCANHA_PROFILING is defined, off otherwise.
/// Pattern: [HH:MM:SS.mmm] [profiling] message
std::shared_ptr<spdlog::logger> get_profiling_logger();

/// Create a logger that writes to the given ostream (useful for testing).
std::shared_ptr<spdlog::logger> create_ostream_logger(
    std::string_view name, std::ostream& stream,
    spdlog::level::level_enum level = spdlog::level::info);

/// Change the main logger level at runtime.
void set_logger_level(spdlog::level::level_enum level);

/// Change the profiling logger level at runtime.
void set_profiling_logger_level(spdlog::level::level_enum level);

}  // namespace pycanha
