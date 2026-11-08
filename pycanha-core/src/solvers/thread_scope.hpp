#pragma once

#include <mkl_service.h>

#include "pycanha-core/utils/threads.hpp"

// The OpenMP runtime MKL is linked with (Intel's). Declared here instead of
// including <omp.h>, which may belong to the compiler's own runtime.
extern "C" {
void omp_set_num_threads(int num_threads);
int omp_get_max_threads();
}

namespace pycanha::solvers::detail {

/// Sets the threads of the calling thread for one MKL call and restores the
/// previous setting afterwards. MKL's LAPACK follows MKL's setting, but the
/// two-level PARDISO factorization follows OpenMP's, which defaults to every
/// logical CPU: both are set. 0 uses the global setting, pycanha::threads().
class ThreadScope {
  public:
    explicit ThreadScope(int threads)
        : _count(threads > 0 ? threads : pycanha::threads()),
          _previous_mkl(mkl_set_num_threads_local(_count)),
          _previous_openmp(omp_get_max_threads()) {
        omp_set_num_threads(_count);
    }
    ~ThreadScope() {
        omp_set_num_threads(_previous_openmp);
        mkl_set_num_threads_local(_previous_mkl);
    }
    ThreadScope(const ThreadScope&) = delete;
    ThreadScope& operator=(const ThreadScope&) = delete;
    ThreadScope(ThreadScope&&) = delete;
    ThreadScope& operator=(ThreadScope&&) = delete;

  private:
    int _count;
    int _previous_mkl;
    int _previous_openmp;
};

}  // namespace pycanha::solvers::detail
