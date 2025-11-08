#include "pycanha-core/solvers/solver.hpp"

#include <iostream>
#include <memory>
#include <unordered_set>
#include <utility>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha {

Solver::Solver(std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : _tmm_shptr(std::move(tmm_shptr)),
      tmm(*_tmm_shptr),
      tnw(tmm.network()),
      ltcs(tnw.conductive_matrices()),
      rtcs(tnw.radiative_matrices()),
      KLdd(ltcs.sparse_dd),
      KLdb(ltcs.sparse_db),
      KLbb(ltcs.sparse_bb),
      KRdd(rtcs.sparse_dd),
      KRdb(rtcs.sparse_db),
      KRbb(rtcs.sparse_bb),
      QI_sp(tmm.nodes().qi_vector),
      QS_sp(tmm.nodes().qs_vector),
      QA_sp(tmm.nodes().qa_vector),
      QE_sp(tmm.nodes().qe_vector),
      QR_sp(tmm.nodes().qr_vector),
      T(nullptr, Index{0}),
      Td(nullptr, Index{0}),
      Tb(nullptr, Index{0}),
      Qd(nullptr, Index{0}),
      Qb(nullptr, Index{0}),
      C(nullptr, Index{0}),
      Cd(nullptr, Index{0}),
      Cb(nullptr, Index{0}) {}

void Solver::initialize_common() {
    if constexpr (DEBUG) {
        std::cout << solver_name << " initializing..." << '\n';
    }

    KLdd.makeCompressed();
    KLdb.makeCompressed();
    KLbb.makeCompressed();

    KRdd.makeCompressed();
    KRdb.makeCompressed();
    KRbb.makeCompressed();

    auto& nodes = tmm.nodes();
    N = nodes.num_nodes();
    nd = KLdd.cols();
    nb = KLdb.cols();

    dTd = VectorXd::Zero(nd);
    Td_solver = VectorXd::Zero(nd);
    Q = VectorXd::Zero(N);

    new (&T) WrappVectorXd(nodes.T_vector.data(), N);
    new (&Td) WrappVectorXd(nodes.T_vector.data(), nd);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    new (&Tb) WrappVectorXd(nodes.T_vector.data() + nd, nb);
    new (&C) WrappVectorXd(nodes.C_vector.data(), N);
    new (&Cd) WrappVectorXd(nodes.C_vector.data(), nd);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    new (&Cb) WrappVectorXd(nodes.C_vector.data() + nd, nb);
    new (&Qd) WrappVectorXd(Q.data(), nd);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    new (&Qb) WrappVectorXd(Q.data() + nd, nb);
}

void Solver::callback_transient_time_change() {
    SOLVER_PROFILE_SCOPE("Callback Time Change");
    _tmm_shptr->callback_transient_time_change();
}

void Solver::callback_solver_loop() {
    SOLVER_PROFILE_SCOPE("Callback Solver Loop");
    _tmm_shptr->callback_solver_loop();
}

void Solver::callback_transient_after_timestep() {
    SOLVER_PROFILE_SCOPE("Callback Timestep");
    _tmm_shptr->callback_transient_after_timestep();
}

bool Solver::temperature_convergence_check() {
    SOLVER_PROFILE_SCOPE("Convergence Check");
    dTd = Td_solver - Td;
    max_dT = dTd.cwiseAbs().maxCoeff();
    return max_dT < abstol_temp;
}

bool Solver::energy_convergence_check() {
    const double energy_norm = Q.norm();
    return energy_norm < abstol_enrgy;
}

void Solver::expand_coupling_matrices_with_zeros() {
    /*
     * Align the sparsity patterns of KL and KR by inserting explicit zeros.
     * This allows the solver to reuse the same structure for both matrices,
     * which in turn enables faster coefficient assembly (memcpy-friendly).
     *
     * The zero entries inserted here are tracked so that `restore_...` can
     * revert the matrices to their original shape after the solve finishes.
     */
    store_sparse_nonzero_indices_in_set(KRdd, _original_non_zeros_krdd);
    store_sparse_nonzero_indices_in_set(KLdd, _original_non_zeros_kldd);
    store_sparse_nonzero_indices_in_set(KRdb, _original_non_zeros_krdb);
    store_sparse_nonzero_indices_in_set(KLdb, _original_non_zeros_kldb);

    SpMatRow zero_diag;
    zero_diag.conservativeResize(nd, nd);
    zero_diag.setIdentity();
    zero_diag *= 0.0;

    KRdd += KLdd * 0.0 + zero_diag;
    KLdd += KRdd * 0.0 + zero_diag;
    KRdb += KLdb * 0.0;
    KLdb += KRdb * 0.0;

    KRdd.makeCompressed();
    KLdd.makeCompressed();
    KRdb.makeCompressed();
    KLdb.makeCompressed();
}

void Solver::restore_expanded_coupling_matrices() {
    class IsRealCoupling {
      public:
        explicit IsRealCoupling(
            std::unordered_set<std::pair<int, int>, IntPairHash>* original)
            : _original(original) {}

        bool operator()(Index row, Index col, double /*value*/) const {
            return _original->find(std::make_pair(static_cast<int>(row),
                                                  static_cast<int>(col))) !=
                   _original->end();
        }

      private:
        std::unordered_set<std::pair<int, int>, IntPairHash>* _original;
    };

    KRdd.prune(IsRealCoupling(&_original_non_zeros_krdd));
    KLdd.prune(IsRealCoupling(&_original_non_zeros_kldd));
    KRdb.prune(IsRealCoupling(&_original_non_zeros_krdb));
    KLdb.prune(IsRealCoupling(&_original_non_zeros_kldb));

    _original_non_zeros_krdd.clear();
    _original_non_zeros_kldd.clear();
    _original_non_zeros_krdb.clear();
    _original_non_zeros_kldb.clear();

    KRdd.makeCompressed();
    KLdd.makeCompressed();
    KRdb.makeCompressed();
    KLdb.makeCompressed();
}

void Solver::store_sparse_nonzero_indices_in_set(
    SpMatRow& sparse,
    std::unordered_set<std::pair<int, int>, IntPairHash>& indices_set) {
    for (Index outer = 0; outer < sparse.outerSize(); ++outer) {
        for (SpMatRow::InnerIterator it(sparse, outer); it; ++it) {
            indices_set.emplace(static_cast<int>(it.row()),
                                static_cast<int>(it.col()));
        }
    }
}

}  // namespace pycanha
