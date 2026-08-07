#include "pycanha-core/utils/logger.hpp"

#include <spdlog/common.h>
#include <spdlog/details/file_helper.h>
#include <spdlog/details/log_msg.h>
#include <spdlog/details/os.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <array>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <format>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

// The banner's command line comes from a different place on each platform.
#if defined(_WIN32) || defined(__APPLE__)
#include <cstddef>
#include <span>
#endif

#ifdef _WIN32
#include <cstdlib>
#elifdef __APPLE__
#include <crt_externs.h>
#else
#include <algorithm>
#include <fstream>
#include <ios>
#endif

namespace pycanha {

namespace {

// Console records are read live, so they lead with a wall-clock time and drop
// everything a reader can infer from context. File records are read long after
// the fact, and concurrent runs share one daily file, so they carry the full
// date and the pid that keeps interleaved records attributable to a run.
constexpr auto k_console_log_pattern = "[%H:%M:%S.%e] [%n] [%^%l%$] %v";
constexpr auto k_file_log_pattern = "[%Y-%m-%d %H:%M:%S.%e] [%P] [%n] [%l] %v";
constexpr auto k_profiling_log_pattern = "[%H:%M:%S.%e] [profiling] %v";

// Everything produced is recorded; only warnings and worse are shown. A clean
// run is therefore silent on the console while the file keeps the full trail.
constexpr auto k_default_record_level = spdlog::level::info;
constexpr auto k_default_display_level = spdlog::level::warn;

constexpr auto k_default_log_directory = "logs";

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

void validate_requested_level(const std::string_view threshold_name,
                              const spdlog::level::level_enum level) {
    if (level < k_compiled_active_log_level) {
        throw std::invalid_argument(
            "Cannot set the " + std::string{threshold_name} +
            " threshold to level '" + std::string{level_to_string(level)} +
            "' because SPDLOG_ACTIVE_LEVEL='" +
            std::string{level_to_string(k_compiled_active_log_level)} +
            "' compiled more verbose logs away");
    }
}

#if defined(_WIN32) || defined(__APPLE__)
/// Joins a C-style argument vector with spaces.
std::string join_arguments(const int argc, const char* const* const argv) {
    if (argv == nullptr || argc <= 0) {
        return "unknown";
    }
    std::string joined;
    for (const auto* const argument :
         std::span{argv, static_cast<std::size_t>(argc)}) {
        if (argument == nullptr) {
            break;
        }
        if (!joined.empty()) {
            joined += ' ';
        }
        joined += argument;
    }
    return joined.empty() ? std::string{"unknown"} : joined;
}
#endif

/// The command line of the running process, for the per-run banner. Best
/// effort: a process that cannot report it still gets a banner with the pid.
std::string process_command_line() {
#ifdef _WIN32
    // The Microsoft CRT publishes the host process's arguments, which is what
    // a library loaded into an interpreter wants. Reaching for the Win32
    // GetCommandLine instead would drag in windows.h for one string.
    return join_arguments(__argc, __argv);
#elifdef __APPLE__
    return join_arguments(*_NSGetArgc(), *_NSGetArgv());
#else
    // /proc/self/cmdline separates the arguments with NUL bytes.
    std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
    if (!cmdline.is_open()) {
        return "unknown";
    }
    std::string raw((std::istreambuf_iterator<char>(cmdline)),
                    std::istreambuf_iterator<char>());
    while (!raw.empty() && raw.back() == '\0') {
        raw.pop_back();
    }
    if (raw.empty()) {
        return "unknown";
    }
    std::ranges::replace(raw, '\0', ' ');
    return raw;
#endif
}

[[nodiscard]] std::string two_digits(const int value) {
    const std::string digits = std::to_string(value);
    return digits.size() < 2U ? "0" + digits : digits;
}

/// "YYYY-MM-DD" for the given broken-down local time.
[[nodiscard]] std::string date_stamp(const std::tm& date) {
    return std::to_string(date.tm_year + 1900) + "-" +
           two_digits(date.tm_mon + 1) + "-" + two_digits(date.tm_mday);
}

[[nodiscard]] std::string time_stamp(const std::tm& moment) {
    return two_digits(moment.tm_hour) + ":" + two_digits(moment.tm_min) + ":" +
           two_digits(moment.tm_sec);
}

[[nodiscard]] spdlog::memory_buf_t to_buffer(const std::string& text) {
    spdlog::memory_buf_t buffer;
    spdlog::fmt_lib::format_to(std::back_inserter(buffer), "{}", text);
    return buffer;
}

/// Appends records to `<directory>/YYYY-MM-DD.log`.
///
/// The file is opened on the first record rather than at construction, so a
/// process that logs nothing leaves no directory and no file behind, and a
/// caller still has time to redirect or disable the output. A new day opens a
/// new file; existing files are only ever appended to.
///
/// Nothing here is allowed to abort the calling computation: if the directory
/// or the file cannot be written, the sink reports once and then stays quiet
/// for the rest of the run.
class DailyFileSink final : public spdlog::sinks::base_sink<std::mutex> {
  public:
    DailyFileSink() = default;

    void set_directory(std::filesystem::path directory) {
        const std::scoped_lock lock(mutex_);
        if (directory == _directory) {
            return;
        }
        _directory = std::move(directory);
        close_current();
        // A different directory deserves a fresh attempt even if the previous
        // one was unusable.
        _open_failed = false;
    }

    [[nodiscard]] std::filesystem::path directory() {
        const std::scoped_lock lock(mutex_);
        return _directory;
    }

    void set_enabled(const bool enabled) {
        const std::scoped_lock lock(mutex_);
        if (_enabled == enabled) {
            return;
        }
        _enabled = enabled;
        if (enabled) {
            // Switching output back on is a deliberate request to try again,
            // even if an earlier attempt gave up on this directory.
            _open_failed = false;
        } else {
            close_current();
        }
    }

    [[nodiscard]] bool enabled() {
        const std::scoped_lock lock(mutex_);
        return _enabled;
    }

    [[nodiscard]] std::filesystem::path current_file() {
        const std::scoped_lock lock(mutex_);
        return _current_file;
    }

  protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        if (!_enabled || _open_failed) {
            return;
        }

        const std::tm date = spdlog::details::os::localtime(
            std::chrono::system_clock::to_time_t(msg.time));
        if (!ensure_open(date)) {
            return;
        }

        spdlog::memory_buf_t formatted;
        formatter_->format(msg, formatted);
        try {
            _file.write(formatted);
            // Flushed per record: a thermal run that dies mid-solve should
            // still leave the records that explain why.
            _file.flush();
        } catch (const spdlog::spdlog_ex&) {
            report_failure("write to");
        }
    }

    void flush_() override {
        if (_current_file.empty()) {
            return;
        }
        try {
            _file.flush();
        } catch (const spdlog::spdlog_ex&) {
            report_failure("flush");
        }
    }

  private:
    /// Opens the file for `date` if it is not already open, rolling over when
    /// the day changes. Returns false once the sink has given up.
    bool ensure_open(const std::tm& date) {
        if (!_current_file.empty() && date.tm_yday == _current_yday &&
            date.tm_year == _current_year) {
            return true;
        }
        close_current();

        std::error_code directory_error;
        std::filesystem::create_directories(_directory, directory_error);

        // Sharing the daily file across processes keeps one timeline, which is
        // what makes concurrent runs correlatable. The pid-qualified name is
        // only a fallback for the platforms or filesystems where a second
        // opener is refused.
        const std::string stamp = date_stamp(date);
        const std::array<std::filesystem::path, 2> candidates{
            _directory / (stamp + ".log"),
            _directory / (stamp + "." +
                          std::to_string(spdlog::details::os::pid()) + ".log")};

        for (const auto& candidate : candidates) {
            try {
                _file.open(candidate.string(), /*truncate=*/false);
            } catch (const spdlog::spdlog_ex&) {
                continue;
            }
            _current_file = candidate;
            _current_yday = date.tm_yday;
            _current_year = date.tm_year;
            write_banner(date);
            return true;
        }

        report_failure("open a log file in");
        return false;
    }

    /// One line per process per file, so that records carrying only a pid can
    /// still be traced back to the run that produced them.
    void write_banner(const std::tm& date) {
        const std::string banner = "=== pycanha run | pid " +
                                   std::to_string(spdlog::details::os::pid()) +
                                   " | " + date_stamp(date) + " " +
                                   time_stamp(date) + " | " +
                                   process_command_line() + "\n";
        try {
            const spdlog::memory_buf_t buffer = to_buffer(banner);
            _file.write(buffer);
            _file.flush();
        } catch (const spdlog::spdlog_ex&) {
            report_failure("write to");
        }
    }

    void close_current() {
        _file.close();
        _current_file.clear();
    }

    /// Reports once and disables the file sink for the rest of the run.
    ///
    /// This writes straight to stderr instead of logging a warning: the sink is
    /// called from inside the logger's own sink loop, so logging from here
    /// would re-enter this sink and deadlock on its mutex.
    void report_failure(const std::string_view action) {
        close_current();
        _open_failed = true;
        const std::string message =
            "[pycanha] [warning] cannot " + std::string{action} + " '" +
            _directory.string() +
            "': continuing without file output for this run\n";
        std::cerr << message;
    }

    spdlog::details::file_helper _file;
    std::filesystem::path _directory{k_default_log_directory};
    std::filesystem::path _current_file;
    int _current_yday = -1;
    int _current_year = -1;
    bool _enabled = true;
    bool _open_failed = false;
};

/// The loggers and the sinks they share, built once on first use.
struct LogState {
    std::shared_ptr<spdlog::sinks::stderr_color_sink_mt> console;
    std::shared_ptr<DailyFileSink> file;
    std::shared_ptr<spdlog::logger> core;
    std::shared_ptr<spdlog::logger> python;
};

LogState& state() {
    static LogState instance = [] {
        LogState fresh;
        fresh.console = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        fresh.console->set_pattern(k_console_log_pattern);
        fresh.console->set_level(k_default_display_level);

        fresh.file = std::make_shared<DailyFileSink>();
        fresh.file->set_pattern(k_file_log_pattern);
        fresh.file->set_level(k_default_record_level);

        const std::vector<spdlog::sink_ptr> sinks{fresh.console, fresh.file};
        fresh.core = std::make_shared<spdlog::logger>(
            "pycanha-core", sinks.begin(), sinks.end());
        fresh.core->set_level(k_default_record_level);
        fresh.python = std::make_shared<spdlog::logger>(
            "pycanha", sinks.begin(), sinks.end());
        fresh.python->set_level(k_default_record_level);

        // Registered so that spdlog-level tooling can find them by name. A
        // name already taken is left alone: something has deliberately put its
        // own logger there and must keep receiving the records.
        if (!spdlog::get("pycanha-core")) {
            spdlog::register_logger(fresh.core);
        }
        if (!spdlog::get("pycanha")) {
            spdlog::register_logger(fresh.python);
        }
        return fresh;
    }();

    return instance;
}

/// Returns whatever is registered under `name`, falling back to `own` and
/// re-registering it.
///
/// The lookup is what lets a test drop the registered logger and put a capture
/// logger of the same name in its place. The configuration entry points below
/// deliberately do not follow that substitution: they configure this library's
/// own logger and sinks, which is what they exist for.
std::shared_ptr<spdlog::logger> registered_or_own(
    const std::string& name, const std::shared_ptr<spdlog::logger>& own) {
    if (auto registered = spdlog::get(name)) {
        return registered;
    }
    spdlog::register_logger(own);
    return own;
}

}  // namespace

std::shared_ptr<spdlog::logger> get_logger() {
    return registered_or_own("pycanha-core", state().core);
}

std::shared_ptr<spdlog::logger> get_python_logger() {
    return registered_or_own("pycanha", state().python);
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

void log_noexcept(const spdlog::level::level_enum level,
                  const std::string_view message) noexcept {
    // Non-formatting overload: builds the log_msg directly from the
    // string_view, bypassing spdlog's vformat_to / SPDLOG_LOGGER_CATCH path
    // (see logger.hpp).
    get_logger()->log(level, message);
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

void set_record_level(const spdlog::level::level_enum level) {
    validate_requested_level("record", level);
    auto& current = state();
    // The logger level bounds every sink, so it is what makes the record
    // threshold "the most verbose thing produced at all".
    current.core->set_level(level);
    current.python->set_level(level);
    current.file->set_level(level);
}

spdlog::level::level_enum record_level() { return state().core->level(); }

void set_display_level(const spdlog::level::level_enum level) {
    validate_requested_level("display", level);
    state().console->set_level(level);
}

spdlog::level::level_enum display_level() { return state().console->level(); }

void set_file_output(const bool enabled) { state().file->set_enabled(enabled); }

// Held by the sink rather than mirrored here, so readers and the sink's own
// writes cannot disagree.
bool file_output() { return state().file->enabled(); }

void set_log_directory(std::filesystem::path directory) {
    state().file->set_directory(std::move(directory));
}

std::filesystem::path log_directory() { return state().file->directory(); }

std::filesystem::path current_log_file() {
    return state().file->current_file();
}

void flush() {
    auto& current = state();
    current.core->flush();
    current.python->flush();
}

}  // namespace pycanha
