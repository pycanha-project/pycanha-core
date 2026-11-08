#include "pycanha-core/utils/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <ratio>
#include <thread>
#include <utility>
#include <vector>

#include "pycanha-core/utils/threads.hpp"

#ifndef _WIN32
#include <pthread.h>

#include <stdexcept>
#endif

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

class ThreadPool::State {
  public:
    State() = default;
    // Stops and joins the workers.
    ~State();
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;

    void run(std::size_t tasks, int threads, const Task& task);
    [[nodiscard]] std::optional<double> overhead_ns(int threads);
    [[nodiscard]] std::uint64_t jobs();

  private:
    // Runs a job with `helpers` workers besides the calling thread, which
    // holds _busy.
    void run_job(std::size_t tasks, std::size_t helpers, const Task& task);
    void work(std::size_t worker);
    // Runs tasks of the current job until none is left.
    void take_tasks(std::size_t worker);

    // Held by the job that is running.
    std::mutex _busy;
    // overhead_ns() by thread count; 0 while not measured.
    std::mutex _overhead_mutex;
    std::vector<double> _overhead;
    // Guards everything below but _next and _cancel.
    std::mutex _mutex;
    std::condition_variable _start;
    std::condition_variable _done;
    std::vector<std::thread> _workers;
    const Task* _task{nullptr};
    std::size_t _tasks{0U};
    // Tasks handed out at once.
    std::size_t _chunk{1U};
    std::atomic<std::size_t> _next{0U};
    // Set when a task threw.
    std::atomic<bool> _cancel{false};
    // Workers 1 to _helpers take part in the current job; _running of them
    // have not finished it.
    std::size_t _helpers{0U};
    std::size_t _running{0U};
    std::uint64_t _job{0U};
    std::exception_ptr _error;
    bool _stop{false};
};

#ifndef _WIN32
// fork() copies a process with only the thread that called it. A pool
// inherited by the child keeps the parent's worker list, so a job would wait
// forever for workers that do not exist; its locks and condition variables
// may also be held or waited on by those threads. The child handler gives
// every pool a new state while the child still has a single thread.
class ThreadPool::Registry {
  public:
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;
    Registry(Registry&&) = delete;
    Registry& operator=(Registry&&) = delete;
    ~Registry() = default;

    static void add(ThreadPool* pool) {
        // Registered once, for the life of the process.
        static const bool registered =
            pthread_atfork(&before_fork, &after_fork_in_parent,
                           &after_fork_in_child) == 0;
        if (!registered) {
            throw std::runtime_error(
                "The thread pool could not register its fork handlers");
        }
        auto& registry = instance();
        const std::scoped_lock lock(registry._mutex);
        registry._pools.push_back(pool);
    }

    static void remove(ThreadPool* pool) {
        auto& registry = instance();
        const std::scoped_lock lock(registry._mutex);
        std::erase(registry._pools, pool);
    }

  private:
    Registry() = default;

    static Registry& instance() {
        static Registry registry;
        return registry;
    }

    // Keeps the list from changing while the process is copied.
    static void before_fork() noexcept { instance()._mutex.lock(); }
    static void after_fork_in_parent() noexcept { instance()._mutex.unlock(); }
    static void after_fork_in_child() noexcept {
        auto& registry = instance();
        for (auto* const pool : registry._pools) {
            pool->_abandoned.push_back(pool->_state.release());
            pool->_state = std::make_unique<State>();
        }
        // Locked by this thread, the one that called fork().
        registry._mutex.unlock();
    }

    std::mutex _mutex;
    std::vector<ThreadPool*> _pools;
};
#endif

ThreadPool::ThreadPool() : _state(std::make_unique<State>()) {
#ifndef _WIN32
    Registry::add(this);
#endif
}

ThreadPool::~ThreadPool() {
#ifndef _WIN32
    Registry::remove(this);
#endif
}

ThreadPool& ThreadPool::shared() {
    static ThreadPool pool;
    return pool;
}

void ThreadPool::run(std::size_t tasks, int threads, const Task& task) {
    _state->run(tasks, threads, task);
}

std::optional<double> ThreadPool::overhead_ns(int threads) {
    return _state->overhead_ns(threads);
}

std::uint64_t ThreadPool::jobs() { return _state->jobs(); }

ThreadPool::State::~State() {
    {
        const std::scoped_lock lock(_mutex);
        _stop = true;
    }
    _start.notify_all();
    for (auto& worker : _workers) {
        worker.join();
    }
}

void ThreadPool::State::run(std::size_t tasks, int threads, const Task& task) {
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

std::optional<double> ThreadPool::State::overhead_ns(int threads) {
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

std::uint64_t ThreadPool::State::jobs() {
    const std::scoped_lock lock(_mutex);
    return _job;
}

void ThreadPool::State::run_job(std::size_t tasks, std::size_t helpers,
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

void ThreadPool::State::work(std::size_t worker) {
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

void ThreadPool::State::take_tasks(std::size_t worker) {
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
