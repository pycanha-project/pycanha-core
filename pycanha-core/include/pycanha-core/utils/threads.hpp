#pragma once

namespace pycanha {

// pycanha has two process-wide thread counts. Each is at its default, the count
// that is usually fastest for its work, until it is set:
//
// - The solver threads: the solvers (MKL and OpenMP inside them) and the
//   formulas. A solver's `threads` setting overrides it for that solver. The
//   default is MKL's own setting in builds with MKL (MKL_NUM_THREADS, else the
//   physical cores), the physical cores otherwise: logical cores beyond the
//   physical ones share a core's arithmetic units and memory bandwidth, so they
//   add little here.
// - The application threads: the rest, which is the boolean cuts of the
//   geometry, the conduction network built from it, the node areas and the
//   assembly of the radiative matrices. The default is every logical CPU: the
//   cuts run faster with two threads per core.
[[nodiscard]] int solver_threads() noexcept;
[[nodiscard]] int application_threads() noexcept;
// Sets both counts; 0 restores both defaults. Throws std::invalid_argument for
// a negative count.
void set_threads(int threads);
// Set one count, whatever set_threads() did before; 0 restores its default.
// Throw std::invalid_argument for a negative count.
void set_solver_threads(int threads);
void set_application_threads(int threads);
// The physical cores the process may run on: those with at least one logical
// CPU in its CPU affinity. On Windows with more than one processor group
// (over 64 logical CPUs) every core counts. The logical CPUs when the system
// does not tell.
[[nodiscard]] int physical_cores() noexcept;

}  // namespace pycanha
