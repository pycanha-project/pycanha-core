#include "pycanha-core/solvers/sslu.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include "pycanha-core/solvers/linear_solver.hpp"
#include "pycanha-core/solvers/ss.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace pycanha {

SSLU::SSLU(std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : SteadyStateSolver(std::move(tmm_shptr)) {
    solver_name = "SSLU";
}

void SSLU::initialize() {
    release_linearised();
    if (pardiso_iparm_3 != 0) {
        throw std::invalid_argument(
            "SSLU is a direct solver and needs pardiso_iparm_3 = 0. For the "
            "iterative step use SSLU_CGS.");
    }
    this->initialize_common();

    auto options = linear_solver_options(engine, solver_type);
    const auto type = resolve_solver_type(engine, solver_type);
    if (type == DirectSolverType::LDLT && has_radiation()) {
        throw std::invalid_argument(
            "LDLT needs a model without radiative couplings: radiation makes "
            "the linearised matrix unsymmetric. Use COLAMD or AMD.");
    }
    options.symmetric_positive_definite =
        type == DirectSolverType::LDLT ||
        (engine == SolverEngine::MKL && allow_cholesky && !has_radiation());
    initialize_linearised(options);
}

void SSLU::solve() { solve_linearised(); }

void SSLU::deinitialize() { release_linearised(); }

}  // namespace pycanha
