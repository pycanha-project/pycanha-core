#pragma once

// Forward-declare the profiling logger accessor to avoid pulling in spdlog
// headers when profiling is disabled.
#ifdef PYCANHA_PROFILING
#include <spdlog/logger.h>

#include <chrono>
#include <memory>
#include <string>
#include <string_view>

#include "pycanha-core/utils/logger.hpp"
#endif

namespace pycanha {

#ifdef PYCANHA_PROFILING

/// RAII timer that logs elapsed milliseconds to the profiling logger on
/// destruction.
class ProfileScope {
  public:
    explicit ProfileScope(std::string_view name)
        : _name(name), _start(std::chrono::steady_clock::now()) {}

    ~ProfileScope() {
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(end - _start);
        const double ms =
            static_cast<double>(elapsed.count()) / 1000.0;  // NOLINT
        get_profiling_logger()->info("[scope] {} took {:.3f} ms", _name, ms);
    }

    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
    ProfileScope(ProfileScope&&) = delete;
    ProfileScope& operator=(ProfileScope&&) = delete;

  private:
    std::string _name;
    std::chrono::steady_clock::time_point _start;
};

#endif  // PYCANHA_PROFILING

}  // namespace pycanha

// Profiling macros — zero overhead when PYCANHA_PROFILING is not defined.
#ifdef PYCANHA_PROFILING
// The __LINE__ token-paste trick requires a two-level macro expansion.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define PYCANHA_PROFILING_CONCAT_(a, b) a##b
// NOLINTNEXTLINE(bugprone-reserved-identifier)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define PYCANHA_PROFILING_CONCAT(a, b) PYCANHA_PROFILING_CONCAT_(a, b)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define PYCANHA_PROFILE_SCOPE(name)                                        \
    const ::pycanha::ProfileScope PYCANHA_PROFILING_CONCAT(_pycanha_prof_, \
                                                           __LINE__)(name)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define PYCANHA_PROFILE_FUNCTION() PYCANHA_PROFILE_SCOPE(__func__)
#else
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define PYCANHA_PROFILE_SCOPE(name)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define PYCANHA_PROFILE_FUNCTION()
#endif
