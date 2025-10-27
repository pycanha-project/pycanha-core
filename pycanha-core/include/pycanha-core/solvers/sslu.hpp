#pragma once

#include <Eigen/SparseLU>
#include <memory>

#include "pycanha-core/solvers/ss.hpp"

namespace pycanha {

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

  private:
    SpMatRow _k_matrix;

    VectorXd _t_cubed;
    VectorXd _t_fourth;

    WrappVectorXd _t_cubed_domain;
    WrappVectorXd _t_cubed_boundary;
    WrappVectorXd _t_fourth_domain;
    WrappVectorXd _t_fourth_boundary;

    Eigen::SparseLU<Eigen::SparseMatrix<double>> _linear_solver;

    void add_radiative_diagonal_to_matrix();
    void add_conductive_diagonal_to_matrix();
};

}  // namespace pycanha
