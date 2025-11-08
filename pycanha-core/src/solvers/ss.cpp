#include "pycanha-core/solvers/ss.hpp"

#include <iostream>
#include <memory>
#include <utility>

#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace pycanha {

SteadyStateSolver::SteadyStateSolver(
    std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : Solver(std::move(tmm_shptr)) {}

void SteadyStateSolver::restart_solve() {
    std::cout << "Re-starting solve..." << '\n';
    std::cout << "ERROR: Not implemented yet." << '\n';
}

}  // namespace pycanha
