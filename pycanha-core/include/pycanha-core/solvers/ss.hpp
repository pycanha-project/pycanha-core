#pragma once

#include <memory>

#include "pycanha-core/solvers/solver.hpp"

namespace pycanha {

class SteadyStateSolver : public Solver {
    friend class SSLU;

  public:
    explicit SteadyStateSolver(
        std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~SteadyStateSolver() override = default;

  protected:
    void restart_solve() override;
};

}  // namespace pycanha
