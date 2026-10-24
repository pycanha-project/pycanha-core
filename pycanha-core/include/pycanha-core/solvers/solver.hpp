#pragma once

#include <Eigen/Sparse>
#include <memory>
#include <string_view>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha::solvers {

using SpMatRow = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SpVec = Eigen::SparseVector<double>;
using VectorXd = Eigen::VectorXd;
using WrappVectorXd = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>;

/// Base of every solver type.
/**
 * A solver works on one model with one Settings. initialize() resolves the
 * settings (see resolve()), warns once about the set fields that do not
 * apply, and builds everything the solver needs; solve() runs it. The INIT
 * settings are frozen while the solver is initialised; the LIVE ones are read
 * again at every solve().
 *
 * solve() throws ModelChangedError when the model's nodes or coupling
 * patterns changed since initialize(), checked at the start and after every
 * callback.
 */
class Solver {
  public:
    virtual ~Solver() = default;
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver& operator=(Solver&&) = delete;

    virtual void initialize() = 0;
    virtual void solve() = 0;
    virtual void deinitialize() = 0;

    [[nodiscard]] SolverType solver_type() const noexcept {
        return _solver_type;
    }
    /// "SteadyState", "CrankNicolson", "JacobianPropagation".
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// A copy, with solver_type set to this solver's type.
    [[nodiscard]] Settings settings() const { return _settings; }
    /// Copies `settings` and sets its solver_type to this solver's. Throws
    /// std::runtime_error when an INIT field changes while the solver is
    /// initialised.
    void set_settings(const Settings& settings);
    /// Changes one setting by name; see Settings::set and set_settings().
    void set(std::string_view name, const SettingValue& value);
    /// Unsets one setting by name, so that it takes its default.
    void unset(std::string_view name);
    /// The settings resolved for the current model, as initialize() would.
    [[nodiscard]] ResolvedSettings resolved_settings() const;

    [[nodiscard]] bool initialized() const noexcept { return _initialized; }
    [[nodiscard]] bool converged() const noexcept { return _converged; }

  protected:
    Solver(std::shared_ptr<ThermalMathematicalModel> tmm_shptr,
           SolverType type);

    class FormulaExecutionGuard {
      public:
        explicit FormulaExecutionGuard(Solver& solver);
        ~FormulaExecutionGuard();

        FormulaExecutionGuard(const FormulaExecutionGuard&) = delete;
        FormulaExecutionGuard& operator=(const FormulaExecutionGuard&) = delete;
        FormulaExecutionGuard(FormulaExecutionGuard&&) = delete;
        FormulaExecutionGuard& operator=(FormulaExecutionGuard&&) = delete;

      private:
        Solver& _solver;
    };

    std::shared_ptr<ThermalMathematicalModel> _tmm_shptr;
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
    int solver_iter = 0;

    VectorXd Td_solver;

    /// The settings resolved by initialize(); its LIVE fields are refreshed
    /// at the start of every solve().
    ResolvedSettings _resolved;
    bool _initialized = false;
    bool _converged = false;

    /// Resolves the settings for the current model, warns about the set
    /// fields that do not apply, sizes the shared vectors and records the
    /// model's structure versions. Throws std::invalid_argument for
    /// contradictory settings.
    void initialize_common();
    /// Refreshes the LIVE fields of the resolved settings; checks the model.
    void refresh_live_settings();

    void callback_solver_loop();
    void callback_transient_time_change();
    void callback_transient_after_timestep();

    // Throws ModelChangedError when nodes were added, removed, retyped or
    // reordered, or couplings inserted or removed, since initialize(): the
    // solver's sizes, patterns and index tables no longer match the model and
    // it must be initialised again. `callback` names the callback that has
    // just run, empty at the start of solve().
    void check_model_unchanged(std::string_view callback = {}) const;

    [[nodiscard]] bool temperature_convergence_check();

  private:
    void prepare_formulas_for_execution();
    void release_formulas_after_execution() noexcept;

    SolverType _solver_type;
    Settings _settings;
    // The model's structure versions when initialize() last ran.
    StructureVersions _initialized_structure;
};

}  // namespace pycanha::solvers
