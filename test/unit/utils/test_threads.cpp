#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <stdexcept>
#include <thread>

#include "pycanha-core/utils/parallel_for.hpp"
#include "pycanha-core/utils/threads.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

[[nodiscard]] int logical_cpus() {
    return std::max(static_cast<int>(std::thread::hardware_concurrency()), 1);
}

// Leaves both counts at their defaults, also when an assertion fails.
class RestoreThreads {
  public:
    RestoreThreads() { set_threads(0); }
    ~RestoreThreads() { set_threads(0); }
    RestoreThreads(const RestoreThreads&) = delete;
    RestoreThreads& operator=(const RestoreThreads&) = delete;
    RestoreThreads(RestoreThreads&&) = delete;
    RestoreThreads& operator=(RestoreThreads&&) = delete;
};

}  // namespace

TEST_CASE("The default solver threads are at least one", "[threads]") {
    const RestoreThreads restore;
    REQUIRE(physical_cores() >= 1);
    REQUIRE(physical_cores() <= logical_cpus());
    REQUIRE(solver_threads() >= 1);
}

TEST_CASE("The default application threads are the logical CPUs", "[threads]") {
    const RestoreThreads restore;
    REQUIRE(application_threads() == logical_cpus());
}

TEST_CASE("Both thread counts can be set and restored together", "[threads]") {
    const RestoreThreads restore;
    const int default_solver = solver_threads();
    set_threads(3);
    REQUIRE(solver_threads() == 3);
    REQUIRE(application_threads() == 3);
    REQUIRE_THROWS_AS(set_threads(-1), std::invalid_argument);
    REQUIRE(solver_threads() == 3);
    REQUIRE(application_threads() == 3);
    set_threads(0);
    REQUIRE(solver_threads() == default_solver);
    REQUIRE(application_threads() == logical_cpus());
}

TEST_CASE("Each thread count can be set on its own", "[threads]") {
    const RestoreThreads restore;
    const int default_solver = solver_threads();
    set_solver_threads(3);
    REQUIRE(solver_threads() == 3);
    REQUIRE(application_threads() == logical_cpus());
    set_application_threads(5);
    REQUIRE(solver_threads() == 3);
    REQUIRE(application_threads() == 5);
    REQUIRE_THROWS_AS(set_solver_threads(-1), std::invalid_argument);
    REQUIRE_THROWS_AS(set_application_threads(-1), std::invalid_argument);
    REQUIRE(solver_threads() == 3);
    REQUIRE(application_threads() == 5);

    // The last call wins.
    set_threads(4);
    set_solver_threads(6);
    REQUIRE(solver_threads() == 6);
    REQUIRE(application_threads() == 4);
    set_solver_threads(0);
    REQUIRE(solver_threads() == default_solver);
    REQUIRE(application_threads() == 4);
    set_application_threads(0);
    REQUIRE(application_threads() == logical_cpus());
}

TEST_CASE("Independent tasks use at most the application threads",
          "[threads]") {
    const RestoreThreads restore;
    set_application_threads(4);
    REQUIRE(utils::workers_for(0U) == 0U);
    REQUIRE(utils::workers_for(1U) == 1U);
    REQUIRE(utils::workers_for(3U) == 3U);
    REQUIRE(utils::workers_for(100U) == 4U);
    // The count is used as given, also above the logical CPUs.
    const int many = logical_cpus() + 5;
    set_application_threads(many);
    REQUIRE(utils::workers_for(static_cast<std::size_t>(many) + 1U) ==
            static_cast<unsigned>(many));
}
