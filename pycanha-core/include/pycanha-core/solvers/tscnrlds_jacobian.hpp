#pragma once

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/solvers/tscnrlds.hpp"

namespace pycanha {

class TSCNRLDS_JACOBIAN : public TSCNRLDS {
  public:
    explicit TSCNRLDS_JACOBIAN(
        std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~TSCNRLDS_JACOBIAN() override = default;

    void initialize() override;
    void solve() override;

    [[nodiscard]] const std::vector<std::string>& parameter_names()
        const noexcept {
        return _parameter_names;
    }

    [[nodiscard]] const std::vector<std::string>& derivative_parameter_names()
        const noexcept {
        return _derivative_parameter_names;
    }

  private:
    using DenseJacobian = Eigen::MatrixXd;

    void collect_parameter_names();
    void fill_matrices(const Entity& entity, Index parameter_index,
                       double derivative_value);
    void build_mk();
    void build_mq();
    void build_mc();
    void solve_jacobian_step();
    void save_jacobian_data();

    std::vector<std::string> _parameter_names;
    std::vector<std::string> _derivative_parameter_names;

    std::vector<SpMatRow> _d_kl_dd_matrices;
    std::vector<SpMatRow> _d_kl_db_matrices;
    std::vector<SpMatRow> _d_kr_dd_matrices;
    std::vector<SpMatRow> _d_kr_db_matrices;

    DenseJacobian _d_capacity_matrix;
    DenseJacobian _d_heat_flux_matrix;
    DenseJacobian _m_k;
    DenseJacobian _m_q;
    DenseJacobian _m_c;
    DenseJacobian _mt;
    DenseJacobian _mb;
    SpMatRow _sp_nd_diag;
};

}  // namespace pycanha
