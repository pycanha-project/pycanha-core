#include "pycanha-core/utils/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <ratio>
#include <thread>
#include <utility>

#include "pycanha-core/utils/threads.hpp"

namespace pycanha::detail {

thread_local int ThreadCountScope::_current = 0;

int active_threads() noexcept {
    return ThreadCountScope::_current > 0 ? ThreadCountScope::_current
                                          : pycanha::threads();
}

ThreadCountScope::ThreadCountScope(int threads) noexcept : _previous(_current) {
    if (threads > 0) {
        _current = threads;
    }
}

ThreadCountScope::~ThreadCountScope() { _current = _previous; }

ThreadPool::~ThreadPool() {
    {
        const std::scoped_lock lock(_mutex);
        _stop = true;
    }
    _start.notify_all();
    for (auto& worker : _workers) {
        worker.join();
    }
}

ThreadPool& ThreadPool::shared() {
    static ThreadPool pool;
    return pool;
}

void ThreadPool::run(std::size_t tasks, int threads, const Task& task) {
    const std::size_t helpers =
        (threads > 1) && (tasks > 1U)
            ? std::min(static_cast<std::size_t>(threads), tasks) - 1U
            : 0U;
    const std::unique_lock busy(_busy, std::try_to_lock);
    if ((helpers == 0U) || !busy.owns_lock()) {
        for (std::size_t index = 0U; index < tasks; ++index) {
            task(index, 0U);
        }
        return;
    }
    run_job(tasks, helpers, task);
}

std::optional<double> ThreadPool::overhead_ns(int threads) {
    if (threads < 2) {
        return 0.0;
    }
    const auto count = static_cast<std::size_t>(threads);
    {
        const std::scoped_lock lock(_overhead_mutex);
        if ((count < _overhead.size()) && (_overhead[count] > 0.0)) {
            return _overhead[count];
        }
    }
    const std::unique_lock busy(_busy, std::try_to_lock);
    if (!busy.owns_lock()) {
        return std::nullopt;
    }
    // The fastest of a few jobs whose tasks do nothing: what is left is
    // waking the workers and waiting for them.
    const Task nothing = [](std::size_t /*task*/, std::size_t /*worker*/) {};
    double fastest = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        const auto start = std::chrono::steady_clock::now();
        run_job(count, count - 1U, nothing);
        const std::chrono::duration<double, std::nano> elapsed =
            std::chrono::steady_clock::now() - start;
        fastest =
            trial == 0 ? elapsed.count() : std::min(fastest, elapsed.count());
    }
    // Never 0, which marks a count not measured yet.
    fastest = std::max(fastest, 1.0);
    const std::scoped_lock lock(_overhead_mutex);
    if (_overhead.size() <= count) {
        _overhead.resize(count + 1U, 0.0);
    }
    _overhead[count] = fastest;
    return fastest;
}

void ThreadPool::run_job(std::size_t tasks, std::size_t helpers,
                         const Task& task) {
    {
        const std::scoped_lock lock(_mutex);
        while (_workers.size() < helpers) {
            const std::size_t worker = _workers.size() + 1U;
            _workers.emplace_back([this, worker] { work(worker); });
        }
        _task = &task;
        _tasks = tasks;
        // Consecutive tasks go to one thread, so that a thread working
        // through neighbouring data reads it in long streams; small enough
        // pieces (about 8 per thread) that threads finishing early take over
        // the rest.
        _chunk = std::max<std::size_t>(tasks / ((helpers + 1U) * 8U), 1U);
        _next.store(0U, std::memory_order_relaxed);
        _cancel.store(false, std::memory_order_relaxed);
        _helpers = helpers;
        _running = helpers;
        _error = nullptr;
        ++_job;
    }
    _start.notify_all();
    take_tasks(0U);
    std::exception_ptr error;
    {
        std::unique_lock lock(_mutex);
        _done.wait(lock, [this] { return _running == 0U; });
        _task = nullptr;
        error = std::exchange(_error, nullptr);
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

void ThreadPool::work(std::size_t worker) {
    std::uint64_t seen = 0U;
    while (true) {
        {
            std::unique_lock lock(_mutex);
            _start.wait(lock, [this, worker, seen] {
                return _stop || ((_job != seen) && (worker <= _helpers));
            });
            if (_stop) {
                return;
            }
            seen = _job;
        }
        take_tasks(worker);
        bool last = false;
        {
            const std::scoped_lock lock(_mutex);
            last = --_running == 0U;
        }
        if (last) {
            _done.notify_one();
        }
    }
}

void ThreadPool::take_tasks(std::size_t worker) {
    while (true) {
        const std::size_t first =
            _next.fetch_add(_chunk, std::memory_order_relaxed);
        if (first >= _tasks) {
            return;
        }
        const std::size_t end = std::min(first + _chunk, _tasks);
        try {
            for (std::size_t index = first;
                 (index < end) && !_cancel.load(std::memory_order_relaxed);
                 ++index) {
                (*_task)(index, worker);
            }
        } catch (...) {
            const std::scoped_lock lock(_mutex);
            if (!_error) {
                _error = std::current_exception();
            }
            // No further chunk starts, and the running ones stop.
            _cancel.store(true, std::memory_order_relaxed);
            _next.store(_tasks, std::memory_order_relaxed);
        }
    }
}

}  // namespace pycanha::detail
