#pragma once

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace pycanha::utils {

// Runs `body(index)` for every index in [0, count) across `threads` workers,
// handing indices out dynamically so uneven work items balance themselves.
//
// The caller is responsible for the only property that makes this safe and
// deterministic: every work item must write only its own output face, so which
// worker takes which index cannot change the result. `threads <= 1` runs the
// loop inline, which is what lets a test pin the worker count and show the
// result does not depend on it.
template <typename Body>
void parallel_for_index(std::size_t count, unsigned threads, const Body& body) {
    if (count == 0) {
        return;
    }
    if (threads <= 1) {
        for (std::size_t index = 0; index < count; ++index) {
            body(index);
        }
        return;
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::jthread> workers;
    workers.reserve(threads);
    for (unsigned worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&] {
            for (std::size_t index = next.fetch_add(1); index < count;
                 index = next.fetch_add(1)) {
                body(index);
            }
        });
    }
}

}  // namespace pycanha::utils
