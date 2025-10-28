#pragma once

#include <memory>

#include "pycanha-core/solvers/ts.hpp"

namespace pycanha {

class TSCN : public TransientSolver {
    friend class TSCNRR;
    friend class TSCNRL;

  public:
    explicit TSCN(std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~TSCN() override = default;

  protected:
    void initialize_common();

    SpMatRow _k_matrix;
    VectorXd _rhs;
    SpMatRow _boundary_matrix;

    VectorXd _heat_flux_n0;
    VectorXd _capacities;
    VectorXd _capacities_inverse;
};

}  // namespace pycanha