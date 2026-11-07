#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "pycanha-core/utils/thread_pool.hpp"
#include "pycanha-core/utils/threads.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("A thread pool runs every task once on the workers it was given",
          "[threads]") {
    detail::ThreadPool pool;
    for (const int threads : {1, 2, 4, 7}) {
        std::vector<std::atomic<int>> runs(1000);
        std::atomic<std::size_t> widest{0U};
        pool.run(runs.size(), threads,
                 [&runs, &widest](std::size_t task, std::size_t worker) {
                     runs[task].fetch_add(1);
                     std::size_t seen = widest.load();
                     while ((worker > seen) &&
                            !widest.compare_exchange_weak(seen, worker)) {
                     }
                 });
        for (const auto& count : runs) {
            REQUIRE(count.load() == 1);
        }
        REQUIRE(widest.load() < static_cast<std::size_t>(threads));
    }
}

TEST_CASE("A thread pool rethrows a task's exception and stays usable",
          "[threads]") {
    detail::ThreadPool pool;
    REQUIRE_THROWS_AS(pool.run(100U, 4,
                               [](std::size_t task, std::size_t /*worker*/) {
                                   if (task == 37U) {
                                       throw std::out_of_range("task 37");
                                   }
                               }),
                      std::out_of_range);
    std::atomic<std::size_t> done{0U};
    pool.run(100U, 4, [&done](std::size_t /*task*/, std::size_t /*worker*/) {
        done.fetch_add(1U);
    });
    REQUIRE(done.load() == 100U);
}

TEST_CASE("A job started from a task of the pool runs on its thread",
          "[threads]") {
    detail::ThreadPool pool;
    std::atomic<std::size_t> inner{0U};
    pool.run(8U, 4, [&pool, &inner](std::size_t /*task*/, std::size_t) {
        pool.run(10U, 4, [&inner](std::size_t /*task*/, std::size_t worker) {
            REQUIRE(worker == 0U);
            inner.fetch_add(1U);
        });
    });
    REQUIRE(inner.load() == 80U);
}

TEST_CASE("A thread count scope overrides the global count on its thread",
          "[threads]") {
    REQUIRE(detail::active_threads() == threads());
    {
        const detail::ThreadCountScope outer(3);
        REQUIRE(detail::active_threads() == 3);
        {
            const detail::ThreadCountScope keep(0);
            REQUIRE(detail::active_threads() == 3);
            const detail::ThreadCountScope inner(5);
            REQUIRE(detail::active_threads() == 5);
        }
        REQUIRE(detail::active_threads() == 3);
    }
    REQUIRE(detail::active_threads() == threads());
}
