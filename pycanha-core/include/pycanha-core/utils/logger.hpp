#pragma once

// SPDLOG_ACTIVE_LEVEL must be defined before any spdlog include so that
// compile-time level stripping macros (SPDLOG_LOGGER_TRACE, etc.) work
// correctly. The actual value is set via CMake compile definitions.
// Default behaviour is TRACE in Debug and INFO otherwise, and it can be
// overridden with PYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE.

#include <spdlog/common.h>
#include <spdlog/logger.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <ostream>
#include <string_view>
#include <vector>

#include "pycanha-core/utils/log_record.hpp"

namespace pycanha {

/// Main library logger ("pycanha-core").
///
/// Writes to three sinks that carry independent levels: a console sink on
/// stderr, filtered by the display threshold, and a daily file sink plus an
/// in-memory buffer, both filtered by the record threshold. That split is what
/// lets the library keep a full record while staying quiet on the console.
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

/// The most verbose level this build kept. Calls below it were removed at
/// compile time, so no runtime setting can bring them back and both thresholds
/// refuse to go past it.
///
/// A caller that wants everything the build can give — a test run, a
/// development session — has to ask for this rather than name a level, because
/// naming one that a leaner build compiled away is an error.
[[nodiscard]] spdlog::level::level_enum compiled_log_level();

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

/// Take everything the in-memory buffer has accumulated since the previous
/// take, and report how much it had to discard to stay bounded.
///
/// This is the whole of the library's outbound integration: records are pulled
/// out by whoever wants them, on their own thread and at a moment of their
/// choosing. The library itself never calls out, which is what keeps it usable
/// and measurable on its own, and keeps a caller's own locking out of the
/// logging path.
///
/// A record is returned by exactly one take, so consecutive takes deliver a
/// stream without repeats or gaps — except for the discards, which the returned
/// count makes visible.
[[nodiscard]] LogDrain drain_log_records();

/// The most recent `count` records still held in memory, without consuming
/// them: taking records and inspecting them are independent.
///
/// Fewer than `count` records are returned when the buffer holds fewer, and the
/// oldest come first. This is how recent records can be examined when file
/// output is off and the console threshold hides them, which is the normal
/// configuration for an interactive session.
[[nodiscard]] std::vector<LogRecord> log_records(std::size_t count);

/// Resize the in-memory buffer, discarding the oldest records if the new
/// capacity is smaller. Clamped to at least one record.
void set_log_buffer_capacity(std::size_t capacity);

/// How many records the in-memory buffer holds before it starts discarding.
[[nodiscard]] std::size_t log_buffer_capacity();

/// Discard every buffered record and reset the discard count.
///
/// A deliberate reset, so what it throws away is not counted as a discard: the
/// caller asked for it and cannot be surprised by the resulting gap.
void clear_log_records();

}  // namespace pycanha
