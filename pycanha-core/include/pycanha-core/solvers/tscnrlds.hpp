#pragma once

#include <memory>
#include <vector>

#include "pycanha-core/solvers/linear_solver.hpp"
#include "pycanha-core/solvers/tscnrl.hpp"

namespace pycanha {

/// Transient solver: Crank-Nicolson with the radiation linearised at every
/// inner iteration, solved with a sparse direct factorisation.
/**
 * - engine and solver_type: as for SSLU (see SolverEngine and
 *   DirectSolverType). DEFAULT is the MKL two-level factorisation, or Eigen's
 *   COLAMD LU without MKL. LDLT is not available: the transient matrix is
 *   factorised with LU.
 * - pardiso_iparm_3 (MKL): 0 (default) factorises every changed matrix. 10 *
 *   L + 1 first iterates on the previous factors down to a relative residual
 *   of 10^-L, and that residual stays in every step. It seems much faster
 *   per step, since the matrix changes little between steps, and it needs
 *   ONE_LEVEL or MIN_DEGREE (see IterativeSolverType).
 * - pardiso_iparm_overrides, mkl_threads, pardiso_verbose: PARDISO settings.
 *
 * initialize() throws std::invalid_argument for a combination that is not
 * available.
 */
class TSCNRLDS : public TSCNRL {
    friend class TSCNRLDS_JACOBIAN;

  public:
    explicit TSCNRLDS(std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~TSCNRLDS() override;
    TSCNRLDS(const TSCNRLDS&) = delete;
    TSCNRLDS& operator=(const TSCNRLDS&) = delete;
    TSCNRLDS(TSCNRLDS&&) = delete;
    TSCNRLDS& operator=(TSCNRLDS&&) = delete;

    void initialize() override;
    void solve() override;
    void deinitialize() override;

    SolverEngine engine = default_solver_engine();
    DirectSolverType solver_type = DirectSolverType::DEFAULT;

  private:
    std::unique_ptr<SparseLinearSolver> _linear_solver;

    VectorXd _t3_domain;
    VectorXd _t3_boundary;
    VectorXd _t4_domain;
    VectorXd _t4_boundary;

    std::vector<int> _lower_kr_indices;
    std::vector<int> _upper_kr_indices;
    std::vector<int> _lower_kl_indices;
    std::vector<int> _upper_kl_indices;
    std::vector<int> _diagonal_indices;

    VectorXd _radiation_linear_term;
    VectorXd _kt_q_n0;
    VectorXd _ones_domain;
    VectorXd _ones_boundary;

    void build_capacities();
    void build_conductance_matrix();
    void build_heat_flux();
    void store_heat_flux_at_n0();
    void euler_step();
    void add_capacities_to_matrix();
    void solve_step();
    void release_solver_resources();
};

}  // namespace pycanha
