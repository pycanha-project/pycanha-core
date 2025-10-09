//
// Basic instrumentation profiler by Cherno (Modified)
// Refs:
// https://www.youtube.com/watch?v=xlAH4dbMVnU
// https://gist.github.com/TheCherno/31f135eea6ee729ab5f26a6908eb3a5e
//

// Usage:
//
// Enable PROFILING macro and include this header file somewhere and use it like
// this:
//
// Instrumentor::Get().BeginSession("SESSION_NAME", "PROFILE_FILE_PATH.json");
// // Begin session
//
// //Scope timing
// {
//     PROFILE_SCOPE("NAME");   // Place code like this in scopes you'd like to
//     include in profiling
//     // Code
// }
//
// //Function timing
//
// void some_function(){
//   PROFILE_FUNCTION();  //name is automatically inferred
//   // code
// }
//
// Instrumentor::Get().EndSession();                        // End Session

#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>  // For std::unique_ptr
#include <mutex>
#include <string>
#include <thread>

#include "../config.hpp"

namespace pycanha {
// Enable/Disable Profiling
using pycanha::PROFILING;

struct ProfileResult {
    std::string name;
    int64_t timestamp;
    char event_type;
    std::thread::id thread_id;
};

struct InstrumentationSession {
    std::string name;
};

class Instrumentor {
    std::unique_ptr<InstrumentationSession> _current_session = nullptr;
    std::ofstream _output_stream;
    int _profile_count = 0;
    std::mutex _lock;

  public:
    Instrumentor() = default;

    void begin_session(const std::string& name,
                       const std::string& filepath = "results.json") {
        _output_stream.open(filepath);
        write_header();
        _current_session = std::make_unique<InstrumentationSession>(
            InstrumentationSession{name});
    }

    void end_session() {
        write_footer();
        _output_stream.close();
        _current_session.reset();
        _profile_count = 0;
    }

    void write_event(const ProfileResult& result) {
        if (_current_session == nullptr) {
            return;
        }

        const std::lock_guard<std::mutex> lock(_lock);

        if (_profile_count++ > 0) {
            _output_stream << ",";
        }

        std::string name = result.name;
        std::replace(name.begin(), name.end(), '"', '\'');

        _output_stream << R"({"cat":"function",)";
        _output_stream << R"("name":")" << name << R"(",)";
        _output_stream << R"("ph":")" << result.event_type << R"(",)";
        _output_stream << R"("pid":0,)";
        _output_stream << R"("tid":)" << result.thread_id << R"(,)";
        _output_stream << R"("ts":)" << result.timestamp;
        _output_stream << "}";

        _output_stream.flush();
    }

    void write_header() {
        _output_stream << R"({"otherData": {},"traceEvents":[)";
        _output_stream.flush();
    }

    void write_footer() {
        _output_stream << "]}";
        _output_stream.flush();
    }

    static Instrumentor& get() {
        static Instrumentor instance;
        return instance;
    }
};

class InstrumentationTimer {
  public:
    explicit InstrumentationTimer(const char* name) : _name(name) { start(); }

    ~InstrumentationTimer() {
        if (!_stopped) {
            stop();
        }
    }

    InstrumentationTimer(const InstrumentationTimer&) = delete;
    InstrumentationTimer& operator=(const InstrumentationTimer&) = delete;
    InstrumentationTimer(InstrumentationTimer&&) = delete;
    InstrumentationTimer& operator=(InstrumentationTimer&&) = delete;

    void start() {
        _start_timepoint = std::chrono::high_resolution_clock::now();
        const int64_t ts =
            std::chrono::time_point_cast<std::chrono::microseconds>(
                _start_timepoint)
                .time_since_epoch()
                .count();
        const std::thread::id thread_id = std::this_thread::get_id();
        Instrumentor::get().write_event({_name, ts, 'B', thread_id});
    }

    void stop() {
        const auto end_timepoint = std::chrono::high_resolution_clock::now();
        const int64_t ts =
            std::chrono::time_point_cast<std::chrono::microseconds>(
                end_timepoint)
                .time_since_epoch()
                .count();
        const std::thread::id thread_id = std::this_thread::get_id();
        Instrumentor::get().write_event({_name, ts, 'E', thread_id});
        _stopped = true;
    }

  private:
    const char* _name;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        _start_timepoint;
    bool _stopped = false;
};

#ifdef PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#endif

}  // namespace pycanha
