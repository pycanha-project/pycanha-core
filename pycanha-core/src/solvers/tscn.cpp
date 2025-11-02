#include "pycanha-core/solvers/tscn.hpp"

#include <memory>
#include <utility>

#include "pycanha-core/solvers/ts.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"

namespace pycanha {

TSCN::TSCN(std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : TransientSolver(std::move(tmm_shptr)) {}

// TODO: Refactor initialize_common naming throughout the solver hierarchy to
// avoid duplicate inherited member warnings once the API is stabilized.
// cppcheck-suppress duplInheritedMember
void TSCN::initialize_common() {
    TransientSolver::initialize_common();

    _k_matrix.conservativeResize(nd, nd);
    _k_matrix.setIdentity();
    _k_matrix += KRdd.selfadjointView<Eigen::Upper>();
    _k_matrix += KLdd.selfadjointView<Eigen::Upper>();
    sparse_utils::set_to_zero(_k_matrix);

    _boundary_matrix.conservativeResize(nd, nb);
    _boundary_matrix = KRdb + KLdb;
    sparse_utils::set_to_zero(_boundary_matrix);

    _rhs = VectorXd::Zero(nd);
    _heat_flux_n0 = VectorXd::Zero(nd);
    _capacities = VectorXd::Zero(nd);
    _capacities_inverse = VectorXd::Zero(nd);
}

}  // namespace pycanha
