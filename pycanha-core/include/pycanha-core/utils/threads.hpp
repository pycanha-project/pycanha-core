#pragma once

namespace pycanha {

// The number of threads pycanha uses wherever a thread count is left at 0
// (the default): the solvers (MKL and OpenMP inside them), the parallel
// formula evaluation and the radiative matrix assembly. A solver's `threads`
// setting overrides it for that solver. The boolean cuts of the geometry use
// every logical CPU instead: the cuts run faster with two threads per core.
//
// The default is the count that is usually fastest on the machine: MKL's
// own setting in builds with MKL (MKL_NUM_THREADS, else the physical cores),
// the physical cores otherwise. Logical cores beyond the physical ones share
// a core's arithmetic units and memory bandwidth, so they add little here:
// nothing for memory-bound work such as table lookups, 10-25 % for formulas
// heavy in arithmetic.
[[nodiscard]] int threads() noexcept;
// Sets the global count; 0 restores the default. Throws
// std::invalid_argument for a negative count.
void set_threads(int threads);
// The default count described above, whatever set_threads() did.
[[nodiscard]] int default_threads() noexcept;
// The physical cores the process may run on: those with at least one logical
// CPU in its CPU affinity. On Windows with more than one processor group
// (over 64 logical CPUs) every core counts. The logical CPUs when the system
// does not tell.
[[nodiscard]] int physical_cores() noexcept;

}  // namespace pycanha
