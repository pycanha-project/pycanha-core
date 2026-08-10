#pragma once

#include <spdlog/common.h>

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace pycanha {

/// A single log record, kept structured instead of pre-formatted.
///
/// A consumer that only receives a rendered line cannot filter it, re-order it
/// or hand it to its own reporting system without parsing the text back apart.
/// Keeping the fields separate is what lets one rebuild a native record on the
/// other side.
struct LogRecord {
    /// When the record was produced. The same clock for every origin, so
    /// records from different layers order correctly against each other.
    std::chrono::system_clock::time_point timestamp;

    spdlog::level::level_enum level = spdlog::level::info;

    /// Name of the logger that produced the record, which is what tells apart
    /// records made inside this library from those made by a layer above it.
    std::string origin;

    /// Already composed, because formatting happens where the arguments are.
    std::string message;

    /// Concurrent processes append to one shared log file, so a record has to
    /// carry the process it came from to stay attributable.
    int pid = 0;

    /// The thread that produced the record. Relevant because a caller may
    /// release its own concurrency guards and log from several threads at once.
    std::size_t thread_id = 0;
};

/// Everything the in-memory buffer had waiting, plus what it had to throw away
/// to make room.
///
/// The count is reported rather than kept silent so that a consumer sees a gap
/// in its stream instead of quietly receiving an incomplete one.
struct LogDrain {
    std::vector<LogRecord> records;
    std::size_t dropped = 0;
};

}  // namespace pycanha
