#pragma once

#include <memory>

#include "pycanha-core/solvers/linear_solver.hpp"
#include "pycanha-core/solvers/ss.hpp"

namespace pycanha {

/// Steady-state solver with a direct factorisation at every Newton pass.
/**
 * Radiation is linearised around the current temperatures (Newton), the
 * linear system is factorised completely and solved, and the passes repeat
 * until the largest temperature change is below abstol_temp. A model without
 * radiative couplings is linear: the first pass gives the answer and the
 * second only confirms it, reusing the factors.
 *
 * - engine: SolverEngine::MKL (PARDISO, multi-threaded) when the library is
 *   built with MKL, SolverEngine::EIGEN otherwise. Choosing MKL in a build
 *   without it makes initialize() throw std::invalid_argument.
 * - solver_type: the factorisation (see DirectSolverType). DEFAULT is
 *   TWO_LEVEL with MKL and COLAMD with Eigen. Every type is a complete
 *   factorisation, so all of them give the same temperatures; they differ in
 *   time and memory. A type of the other engine, or LDLT for a model with
 *   radiative couplings, makes initialize() throw std::invalid_argument.
 * - allow_cholesky (MKL): Cholesky for models without radiative couplings.
 * - pardiso_iparm_overrides, mkl_threads, pardiso_verbose: PARDISO settings.
 *   pardiso_iparm_3 must stay 0: the iterative variant is SSLU_CGS.
 *
 * A model whose matrix is singular (a group of diffusive nodes with no path
 * to a boundary node) is reported as an error and its temperatures are left
 * unchanged.
 */
class SSLU : public SteadyStateSolver {
  public:
    explicit SSLU(std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~SSLU() override = default;

    SSLU(const SSLU&) = delete;
    SSLU& operator=(const SSLU&) = delete;
    SSLU(SSLU&&) = delete;
    SSLU& operator=(SSLU&&) = delete;

    void initialize() override;
    void solve() override;
    void deinitialize() override;

    SolverEngine engine = default_solver_engine();
    DirectSolverType solver_type = DirectSolverType::DEFAULT;
};

}  // namespace pycanha
