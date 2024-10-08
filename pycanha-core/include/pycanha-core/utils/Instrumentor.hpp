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

// Enable/Disable Profiling
#define PROFILING 1

#include <algorithm>
#include <chrono>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

struct ProfileResult {
    std::string name;
    int64_t timestamp;
    char EventType;
    std::thread::id ThreadID;
};

struct InstrumentationSession {
    std::string name;
};

class Instrumentor {
    InstrumentationSession* m_CurrentSession;
    std::ofstream m_OutputStream;
    int m_ProfileCount;
    std::mutex m_lock;

  public:
    Instrumentor() : m_CurrentSession(nullptr), m_ProfileCount(0) {}

    void BeginSession(const std::string& name,
                      const std::string& filepath = "results.json") {
        m_OutputStream.open(filepath);
        WriteHeader();
        m_CurrentSession = new InstrumentationSession{name};
    }

    void EndSession() {
        WriteFooter();
        m_OutputStream.close();
        delete m_CurrentSession;
        m_CurrentSession = nullptr;
        m_ProfileCount = 0;
    }

    void WriteEvent(const ProfileResult& result) {
        if (!m_CurrentSession) return;

        std::lock_guard<std::mutex> lock(m_lock);

        if (m_ProfileCount++ > 0) m_OutputStream << ",";

        std::string name = result.name;
        std::replace(name.begin(), name.end(), '"', '\'');

        m_OutputStream << "{";
        m_OutputStream << "\"cat\":\"function\",";
        m_OutputStream << "\"name\":\"" << name << "\",";
        m_OutputStream << "\"ph\":\"" << result.EventType << "\",";
        m_OutputStream << "\"pid\":0,";
        m_OutputStream << "\"tid\":" << result.ThreadID << ",";
        m_OutputStream << "\"ts\":" << result.timestamp;
        m_OutputStream << "}";

        m_OutputStream.flush();
    }

    void WriteHeader() {
        m_OutputStream << "{\"otherData\": {},\"traceEvents\":[";
        m_OutputStream.flush();
    }

    void WriteFooter() {
        m_OutputStream << "]}";
        m_OutputStream.flush();
    }

    static Instrumentor& Get() {
        static Instrumentor instance;
        return instance;
    }
};

class InstrumentationTimer {
  public:
    explicit InstrumentationTimer(const char* name)
        : m_Name(name), m_Stopped(false) {
        Start();
    }

    ~InstrumentationTimer() {
        if (!m_Stopped) Stop();
    }
    void Start() {
        m_StartTimepoint = std::chrono::high_resolution_clock::now();
        int64_t ts = std::chrono::time_point_cast<std::chrono::microseconds>(
                         m_StartTimepoint)
                         .time_since_epoch()
                         .count();
        // uint32_t threadID =
        // std::hash<std::thread::id>{}(std::this_thread::get_id());
        std::thread::id threadID = std::this_thread::get_id();
        Instrumentor::Get().WriteEvent({m_Name, ts, 'B', threadID});
    }
    void Stop() {
        auto endTimepoint = std::chrono::high_resolution_clock::now();

        int64_t ts = std::chrono::time_point_cast<std::chrono::microseconds>(
                         endTimepoint)
                         .time_since_epoch()
                         .count();

        std::thread::id threadID = std::this_thread::get_id();
        Instrumentor::Get().WriteEvent({m_Name, ts, 'E', threadID});

        m_Stopped = true;
    }

  private:
    const char* m_Name;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        m_StartTimepoint;
    bool m_Stopped;
};

#ifdef PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#else
#define PROFILE_SCOPE(name)
#endif
