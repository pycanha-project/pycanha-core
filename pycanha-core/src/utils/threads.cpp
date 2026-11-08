#include "pycanha-core/utils/threads.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>

#if PYCANHA_USE_MKL
#include <mkl_service.h>
#endif

#ifdef _WIN32
// Without it, <windows.h> defines min and max macros that break std::max.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>

#include <cstddef>
#endif

#ifdef __linux__
#include <sched.h>

#include <fstream>
#include <set>
#endif

namespace pycanha {

namespace {

// 0 while the default is in use.
std::atomic<int>& global_threads() {
    static std::atomic<int> count{0};
    return count;
}

// The fallback when the system does not tell the physical cores.
[[nodiscard]] int logical_cpus() noexcept {
    return std::max(static_cast<int>(std::thread::hardware_concurrency()), 1);
}

#ifdef _WIN32
// The physical cores with at least one logical CPU in the process's affinity
// mask, else the logical CPUs. The mask covers one processor group: with more
// than one group (over 64 logical CPUs) every core counts.
[[nodiscard]] int count_physical_cores() {
    DWORD_PTR process_mask = 0;
    DWORD_PTR system_mask = 0;
    const bool masked =
        (GetActiveProcessorGroupCount() == 1) &&
        (GetProcessAffinityMask(GetCurrentProcess(), &process_mask,
                                &system_mask) != FALSE) &&
        (process_mask != 0);
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (length == 0) {
        return logical_cpus();
    }
    std::string buffer(length, '\0');
    auto* const first = static_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
        static_cast<void*>(buffer.data()));
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, first,
                                         &length) == FALSE) {
        return logical_cpus();
    }
    // Variable-size records, one per core; a core's logical CPUs are all in
    // one group, so its first group mask holds them.
    int cores = 0;
    for (DWORD offset = 0; offset < length;) {
        const auto* const record =
            static_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                static_cast<const void*>(&buffer.at(offset)));
        if (!masked ||
            ((record->Processor.GroupMask[0].Mask & process_mask) != 0)) {
            ++cores;
        }
        offset += record->Size;
    }
    return std::max(cores, 1);
}
#endif

#ifdef __APPLE__
[[nodiscard]] int count_physical_cores() {
    int cores = 0;
    std::size_t size = sizeof(cores);
    if ((sysctlbyname("hw.physicalcpu", &cores, &size, nullptr, 0) != 0) ||
        (cores < 1)) {
        return logical_cpus();
    }
    return cores;
}
#endif

#ifdef __linux__
[[nodiscard]] int count_physical_cores() {
    // Logical CPUs of one core list the same siblings: count the distinct
    // lists over the CPUs this process may use.
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return logical_cpus();
    }
    std::set<std::string> cores;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed) == 0) {
            continue;
        }
        const std::string topology =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::ifstream siblings(topology + "core_cpus_list");
        if (!siblings) {
            siblings.open(topology + "thread_siblings_list");
        }
        std::string list;
        if (!std::getline(siblings, list)) {
            return logical_cpus();
        }
        cores.insert(list);
    }
    return cores.empty() ? logical_cpus() : static_cast<int>(cores.size());
}
#endif

#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__linux__)
[[nodiscard]] int count_physical_cores() { return logical_cpus(); }
#endif

}  // namespace

int physical_cores() noexcept {
    static const int cores = [] {
        try {
            return count_physical_cores();
        } catch (const std::exception&) {
            return logical_cpus();
        }
    }();
    return cores;
}

int default_threads() noexcept {
#if PYCANHA_USE_MKL
    // MKL_NUM_THREADS when set, else the physical cores.
    return std::max(mkl_get_max_threads(), 1);
#else
    return physical_cores();
#endif
}

int threads() noexcept {
    const int chosen = global_threads().load(std::memory_order_relaxed);
    return chosen > 0 ? chosen : default_threads();
}

void set_threads(int threads) {
    if (threads < 0) {
        throw std::invalid_argument("The thread count cannot be negative");
    }
    global_threads().store(threads, std::memory_order_relaxed);
}

}  // namespace pycanha
