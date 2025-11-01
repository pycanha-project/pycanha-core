#pragma once

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <array>
#include <memory>
#include <vector>

#include "pycanha-core/solvers/tscnrl.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"

#if PYCANHA_USE_MKL
#if __has_include(<mkl/mkl.h>)
#include <mkl/mkl.h>
#endif
#endif

namespace pycanha {

class TSCNRLDS : public TSCNRL {
    friend class TSCNRLDS_JACOBIAN;

  public:
    explicit TSCNRLDS(std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~TSCNRLDS() override = default;

    void initialize() override;
    void solve() override;
    void deinitialize() override;

  private:
#if PYCANHA_USE_MKL
    std::array<void*, 64> _pardiso_pt{};
    std::array<MKL_INT, 64> _pardiso_iparm{};
    MKL_INT _pardiso_mtype = 11;
    MKL_INT _pardiso_maxfct = 1;
    MKL_INT _pardiso_mnum = 1;
    MKL_INT _pardiso_phase = 0;
    MKL_INT _pardiso_size = 0;
    std::vector<MKL_INT> _pardiso_perm;
    MKL_INT _pardiso_nrhs = 1;
    MKL_INT _pardiso_msglvl = 0;
    MKL_INT _pardiso_error = 0;
#else
  Eigen::SparseLU<SpMatRow> _eigen_solver;
#endif

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
};

}  // namespace pycanha
