#pragma once

#include <memory>

#include "pycanha-core/solvers/linear_solver.hpp"
#include "pycanha-core/solvers/ss.hpp"

namespace pycanha {

/// Steady-state solver that reuses its factors as a preconditioner (MKL
/// PARDISO only).
/**
 * Same Newton passes as SSLU. The first pass is factorised completely. In
 * the next passes the changed matrix is first solved with CGS (CG for
 * Cholesky) preconditioned by the previous factors, and only refactorised
 * when that iteration fails. It seems faster than SSLU on radiative models,
 * whose matrix changes a little at every pass.
 *
 * The iteration stops at a relative residual of 10^-L, with L the tens digit
 * of pardiso_iparm_3 (default 61: L = 6), and that residual stays in the
 * converged temperatures. Raise L, or use SSLU, for exact answers.
 *
 * - solver_type (see IterativeSolverType):
 *   - MIN_DEGREE (default): minimum degree ordering. Its analysis is slow on
 *     models where a few nodes are coupled to many.
 *   - ONE_LEVEL: nested dissection ordering (METIS). With MKL 2025.3 it can
 *     livelock (never finish) with 4 or more threads on some radiative
 *     models; limit mkl_threads to 2 if it does.
 *   PARDISO only iterates with its one-level factorisation, so the
 *   two-level factorisation of SSLU is not available here.
 * - allow_cholesky: Cholesky and CG for models without radiative couplings.
 * - pardiso_iparm_3: the stopping residual, as above.
 * - pardiso_iparm_overrides, mkl_threads, pardiso_verbose: PARDISO settings.
 *
 * In a build without MKL, initialize() throws std::invalid_argument.
 */
class SSLU_CGS : public SteadyStateSolver {
  public:
    explicit SSLU_CGS(std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~SSLU_CGS() override = default;

    SSLU_CGS(const SSLU_CGS&) = delete;
    SSLU_CGS& operator=(const SSLU_CGS&) = delete;
    SSLU_CGS(SSLU_CGS&&) = delete;
    SSLU_CGS& operator=(SSLU_CGS&&) = delete;

    void initialize() override;
    void solve() override;
    void deinitialize() override;

    IterativeSolverType solver_type = IterativeSolverType::MIN_DEGREE;
};

}  // namespace pycanha
