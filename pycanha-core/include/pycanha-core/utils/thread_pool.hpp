#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
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
//
// A child process created by fork() inherits the pool but none of its
// threads. The child starts the pool over: new threads when a job needs
// them, new locks, and its own overhead measurements.
class ThreadPool {
  public:
    // One task of a job: the task's index, and the worker running it, from
    // 0 (the calling thread) to the job's thread count - 1.
    using Task = std::function<void(std::size_t task, std::size_t worker)>;

    ThreadPool();
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

    // The jobs run on more than one thread in this process so far, the
    // ones overhead_ns() runs to measure included.
    [[nodiscard]] std::uint64_t jobs();

    // The pool of the whole process.
    [[nodiscard]] static ThreadPool& shared();

  private:
    // The workers, the locks and the measurements: everything a child
    // process cannot use and replaces.
    class State;
    // The pools alive in the process, whose states the child replaces.
    class Registry;

    std::unique_ptr<State> _state;
    // In a child process, the states inherited from the parent. Destroying
    // one would join or destroy threads that do not exist in this process,
    // so they are left unused and never destroyed.
    std::vector<State*> _abandoned;
};

}  // namespace pycanha::detail
