#include "pycanha-core/solvers/ss.hpp"

#include <spdlog/spdlog.h>

#include <memory>
#include <utility>

#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {

SteadyStateSolver::SteadyStateSolver(
    std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : Solver(std::move(tmm_shptr)) {}

void SteadyStateSolver::restart_solve() {
    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Re-starting solve...");
    SPDLOG_LOGGER_ERROR(pycanha::get_logger(), "Not implemented yet.");
}

}  // namespace pycanha
