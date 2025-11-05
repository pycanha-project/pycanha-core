#pragma once

#include <Eigen/Sparse>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/Instrumentor.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"

namespace pycanha {

using SpMatRow = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SpVec = Eigen::SparseVector<double>;
using VectorXd = Eigen::VectorXd;
using WrappVectorXd = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>;
using Index = Eigen::Index;

// TODO: Move this helper to a dedicated utilities file if reused elsewhere.
struct IntPairHash {
    static_assert(sizeof(int) * 2 == sizeof(std::size_t));

    std::size_t operator()(const std::pair<int, int>& value) const noexcept {
        return (static_cast<std::size_t>(value.first) << 32U) |
               static_cast<std::size_t>(value.second);
    }
};

#ifndef SOLVER_PROFILE_SCOPE
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SOLVER_PROFILE_SCOPE(name) PROFILE_SCOPE(name)
#endif

class Solver {
    friend class SteadyStateSolver;
    friend class TransientSolver;

  public:
    // NOLINTBEGIN(readability-identifier-naming)
    static constexpr double STF_BOLTZ = 5.670374419e-8;
    // DOI: 10.1103/RevModPhys.97.025002

    explicit Solver(std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    virtual ~Solver() = default;
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver& operator=(Solver&&) = delete;

    virtual void initialize() = 0;
    virtual void solve() = 0;
    virtual void deinitialize() = 0;

    int MAX_ITERS = 5;
    int solver_iter = 0;
    double abstol_temp = 1e-3;
    double abstol_enrgy = 1e-3;
    double eps_capacity = 1.0e-7;
    double eps_time = 1.0e-6;
    double eps_coupling = 1.0e-12;
    int pardiso_iparm_3 = 31;

    bool solver_converged = false;
    bool solver_initialized = false;

    std::string solver_name;
    std::shared_ptr<ThermalMathematicalModel> _tmm_shptr;

  protected:
    ThermalMathematicalModel& tmm;
    ThermalNetwork& tnw;
    CouplingMatrices& ltcs;
    CouplingMatrices& rtcs;

    SpMatRow& KLdd;
    SpMatRow& KLdb;
    SpMatRow& KLbb;

    SpMatRow& KRdd;
    SpMatRow& KRdb;
    SpMatRow& KRbb;

    WrappVectorXd T;
    WrappVectorXd Td;
    WrappVectorXd Tb;

    WrappVectorXd Qd;
    WrappVectorXd Qb;

    WrappVectorXd C;
    WrappVectorXd Cd;
    WrappVectorXd Cb;

    const SpVec& QI_sp;
    const SpVec& QS_sp;
    const SpVec& QA_sp;
    const SpVec& QE_sp;
    const SpVec& QR_sp;

    VectorXd Q;

    Index N = 0;
    Index nd = 0;
    Index nb = 0;

    VectorXd dTd;
    double max_dT = 0.0;

    VectorXd Td_solver;

    void callback_solver_loop();
    void callback_transient_time_change();
    void callback_transient_after_timestep();

    virtual void restart_solve() = 0;

    void initialize_common();

    [[nodiscard]] bool temperature_convergence_check();
    [[nodiscard]] bool energy_convergence_check();

    void expand_coupling_matrices_with_zeros();
    void restore_expanded_coupling_matrices();

    std::unordered_set<std::pair<int, int>, IntPairHash>
        _original_non_zeros_krdd;
    std::unordered_set<std::pair<int, int>, IntPairHash>
        _original_non_zeros_kldd;
    std::unordered_set<std::pair<int, int>, IntPairHash>
        _original_non_zeros_krdb;
    std::unordered_set<std::pair<int, int>, IntPairHash>
        _original_non_zeros_kldb;
    // NOLINTEND(readability-identifier-naming)

  private:
    static void store_sparse_nonzero_indices_in_set(
        SpMatRow& sparse,
        std::unordered_set<std::pair<int, int>, IntPairHash>& indices_set);
};

}  // namespace pycanha
