#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace pycanha::detail {

// The thread count for work started on the calling thread: the count of the
// innermost ThreadCountScope that set one, else pycanha::threads().
[[nodiscard]] int active_threads() noexcept;

// Sets active_threads() on the calling thread while it lives; a count of 0
// keeps the enclosing one. A solver opens one around its callbacks, so the
// formulas applied there follow the solver's `threads` setting.
class ThreadCountScope {
  public:
    explicit ThreadCountScope(int threads) noexcept;
    ~ThreadCountScope();
    ThreadCountScope(const ThreadCountScope&) = delete;
    ThreadCountScope& operator=(const ThreadCountScope&) = delete;
    ThreadCountScope(ThreadCountScope&&) = delete;
    ThreadCountScope& operator=(ThreadCountScope&&) = delete;

  private:
    friend int active_threads() noexcept;

    // The count of this thread's innermost scope that set one, or 0.
    static thread_local int _current;
    int _previous;
};

// Worker threads kept from one job to the next. Between jobs they wait on a
// condition variable: they do not spin, so they take no CPU time from the
// solver's own threads.
class ThreadPool {
  public:
    // One task of a job: the task's index, and the worker running it, from
    // 0 (the calling thread) to the job's thread count - 1.
    using Task = std::function<void(std::size_t task, std::size_t worker)>;

    ThreadPool() = default;
    // Stops and joins the workers.
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Runs task(t, w) for every t in [0, tasks) on up to `threads` threads,
    // the calling thread being worker 0, and returns when all have run.
    // When a task throws, no further task starts and the first exception is
    // rethrown here. While the pool runs another job (from another thread,
    // or the one calling run() from a task), the job runs on the calling
    // thread alone.
    void run(std::size_t tasks, int threads, const Task& task);

    // What running a job on `threads` threads costs on top of its tasks:
    // waking the workers and waiting for them, in nanoseconds; 0 for one
    // thread. Measured on the first call for each count and kept. Nullopt
    // while the pool runs another job, which would distort the measurement.
    [[nodiscard]] std::optional<double> overhead_ns(int threads);

    // The pool of the whole process.
    [[nodiscard]] static ThreadPool& shared();

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

}  // namespace pycanha::detail
