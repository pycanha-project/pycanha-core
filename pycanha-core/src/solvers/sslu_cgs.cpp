#include "pycanha-core/solvers/sslu_cgs.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include "pycanha-core/solvers/linear_solver.hpp"
#include "pycanha-core/solvers/ss.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace pycanha {

SSLU_CGS::SSLU_CGS(std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : SteadyStateSolver(std::move(tmm_shptr)) {
    solver_name = "SSLU_CGS";
    // Stop the iteration at a relative residual of 1e-6.
    constexpr int default_iterative_step = 61;
    pardiso_iparm_3 = default_iterative_step;
}

void SSLU_CGS::initialize() {
    release_linearised();
    constexpr int decimal = 10;
    if (pardiso_iparm_3 % decimal == 0) {
        throw std::invalid_argument(
            "SSLU_CGS needs an iterative step: pardiso_iparm_3 = 10 * L + 1 "
            "(for example 61). For a direct factorisation use SSLU.");
    }
    this->initialize_common();

    auto options = linear_solver_options(
        SolverEngine::MKL, solver_type == IterativeSolverType::ONE_LEVEL
                               ? DirectSolverType::ONE_LEVEL
                               : DirectSolverType::MIN_DEGREE);
    options.symmetric_positive_definite = allow_cholesky && !has_radiation();
    initialize_linearised(options);
}

void SSLU_CGS::solve() { solve_linearised(); }

void SSLU_CGS::deinitialize() { release_linearised(); }

}  // namespace pycanha
