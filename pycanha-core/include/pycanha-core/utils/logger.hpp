#pragma once

// SPDLOG_ACTIVE_LEVEL must be defined before any spdlog include so that
// compile-time level stripping macros (SPDLOG_LOGGER_TRACE, etc.) work
// correctly. The actual value is set via CMake compile definitions.
// Default behaviour is TRACE in Debug and INFO otherwise, and it can be
// overridden with PYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE.

#include <spdlog/common.h>
#include <spdlog/logger.h>

#include <filesystem>
#include <memory>
#include <ostream>
#include <string_view>

namespace pycanha {

/// Main library logger ("pycanha-core").
///
/// Writes to two sinks that carry independent levels: a console sink on
/// stderr, filtered by the display threshold, and a daily file sink, filtered
/// by the record threshold. That split is what lets the library keep a full
/// record on disk while staying quiet on the console.
std::shared_ptr<spdlog::logger> get_logger();

/// Profiling logger ("pycanha-core.profiling").
/// Pattern: [HH:MM:SS.mmm] [profiling] message
std::shared_ptr<spdlog::logger> get_profiling_logger();

/// Logger for records originating outside the C++ core ("pycanha").
///
/// Shares the main logger's sinks and level, so both origins land in one file
/// with one clock and one configuration. The logger name is what distinguishes
/// them in the output.
std::shared_ptr<spdlog::logger> get_python_logger();

/// Log an already-composed `message` through the main logger at `level` from a
/// context that must not throw (e.g. a noexcept function).
///
/// spdlog already wraps message formatting in try/catch, so a std::format_error
/// from formatting our arguments is swallowed. The leak is that spdlog's own
/// catch handler composes its error string with an *unguarded* std::format,
/// which is reachable through every formatting log call (SPDLOG_LOGGER_* and
/// logger->info/warn/...). clang-tidy's bugprone-exception-escape sees that
/// throw and therefore forbids any formatting log call inside a noexcept
/// function. This routes a pre-composed message through the non-formatting
/// spdlog::logger::log(level, string_view) overload, which never touches that
/// std::format path. Compose the message at the call site (e.g. with string
/// concatenation, not std::format, which would re-introduce the throw).
void log_noexcept(spdlog::level::level_enum level,
                  std::string_view message) noexcept;

/// Create a logger that writes to the given ostream (useful for testing).
std::shared_ptr<spdlog::logger> create_ostream_logger(
    std::string_view name, std::ostream& stream,
    spdlog::level::level_enum level = spdlog::level::info);

/// Set the threshold for what is produced at all, and therefore what reaches
/// the log file. Defaults to info.
///
/// Throws std::invalid_argument if the requested level is more verbose than
/// the compile-time SPDLOG_ACTIVE_LEVEL, which already stripped those calls.
void set_record_level(spdlog::level::level_enum level);

/// The current record threshold.
[[nodiscard]] spdlog::level::level_enum record_level();

/// Set the threshold for what reaches the console. Defaults to warn;
/// spdlog::level::off silences the console entirely.
///
/// Throws std::invalid_argument on the same grounds as set_record_level.
void set_display_level(spdlog::level::level_enum level);

/// The current display threshold.
[[nodiscard]] spdlog::level::level_enum display_level();

/// Master switch for writing to disk. When disabled nothing is created or
/// written under the log directory; the console still receives records.
///
/// Callers that write their own report files alongside the log are expected to
/// honour this too, so that turning it off leaves the filesystem untouched.
void set_file_output(bool enabled);

/// Whether writing to disk is enabled.
[[nodiscard]] bool file_output();

/// Directory holding the daily log file. Relative paths are resolved against
/// the current working directory every time a file is opened, so this follows
/// the process rather than the location the library was installed to.
/// Defaults to "logs".
void set_log_directory(std::filesystem::path directory);

/// The configured log directory.
[[nodiscard]] std::filesystem::path log_directory();

/// The log file currently open, or an empty path when there is none: file
/// output disabled, nothing logged yet (the file is created on the first
/// record, not at startup), or the directory could not be written to.
[[nodiscard]] std::filesystem::path current_log_file();

/// Flush both loggers.
void flush();

}  // namespace pycanha
