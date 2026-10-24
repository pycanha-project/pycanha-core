#include "pycanha-core/solvers/solver.hpp"

#include <spdlog/spdlog.h>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/model_changed_error.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/profiling.hpp"

namespace pycanha::solvers {

namespace {

class ActiveSolverCallbackScope {
  public:
    ActiveSolverCallbackScope(ThermalMathematicalModel& tmm, Solver& solver)
        : _tmm(tmm) {
        _tmm.set_current_callback_solver(std::addressof(solver));
    }

    ~ActiveSolverCallbackScope() { _tmm.set_current_callback_solver(nullptr); }

    ActiveSolverCallbackScope(const ActiveSolverCallbackScope&) = delete;
    ActiveSolverCallbackScope& operator=(const ActiveSolverCallbackScope&) =
        delete;
    ActiveSolverCallbackScope(ActiveSolverCallbackScope&&) = delete;
    ActiveSolverCallbackScope& operator=(ActiveSolverCallbackScope&&) = delete;

  private:
    ThermalMathematicalModel& _tmm;
};

// The parts that differ, as "nodes", "nodes and conductive couplings" or
// "nodes, conductive couplings and radiative couplings".
std::string describe_changes(const StructureVersions& before,
                             const StructureVersions& now) {
    std::vector<std::string_view> parts;
    if (before.nodes != now.nodes) {
        parts.emplace_back("nodes");
    }
    if (before.conductive != now.conductive) {
        parts.emplace_back("conductive couplings");
    }
    if (before.radiative != now.radiative) {
        parts.emplace_back("radiative couplings");
    }
    std::string text;
    for (std::size_t part = 0; part < parts.size(); ++part) {
        if (part > 0) {
            text += part + 1 == parts.size() ? " and " : ", ";
        }
        text += parts[part];
    }
    return text;
}

std::string describe_not_applied(const std::vector<NotApplied>& ignored) {
    std::string text;
    for (const NotApplied& field : ignored) {
        if (!text.empty()) {
            text += "; ";
        }
        text += field.name + " (" + field.reason + ")";
    }
    return text;
}

}  // namespace

Solver::Solver(std::shared_ptr<ThermalMathematicalModel> tmm_shptr,
               SolverType type)
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
      T(nullptr, Index{0}),
      Td(nullptr, Index{0}),
      Tb(nullptr, Index{0}),
      Qd(nullptr, Index{0}),
      Qb(nullptr, Index{0}),
      C(nullptr, Index{0}),
      Cd(nullptr, Index{0}),
      Cb(nullptr, Index{0}),
      QI_sp(tmm.nodes().qi_vector),
      QS_sp(tmm.nodes().qs_vector),
      QA_sp(tmm.nodes().qa_vector),
      QE_sp(tmm.nodes().qe_vector),
      QR_sp(tmm.nodes().qr_vector),
      _solver_type(type) {
    _settings.solver_type = type;
}

void Solver::set_settings(const Settings& settings) {
    Settings replacement = settings;
    replacement.solver_type = _solver_type;
    check_settings_change(_settings, replacement,
                          resolve_context(_solver_type, tmm), _initialized);
    _settings = std::move(replacement);
}

void Solver::set(std::string_view name, const SettingValue& value) {
    Settings replacement = _settings;
    replacement.set(name, value);
    set_settings(replacement);
}

void Solver::unset(std::string_view name) {
    Settings replacement = _settings;
    replacement.unset(name);
    set_settings(replacement);
}

ResolvedSettings Solver::resolved_settings() const {
    return resolve(_settings, resolve_context(_solver_type, tmm));
}

Solver::FormulaExecutionGuard::FormulaExecutionGuard(Solver& solver)
    : _solver(solver) {
    _solver.prepare_formulas_for_execution();
}

Solver::FormulaExecutionGuard::~FormulaExecutionGuard() {
    _solver.release_formulas_after_execution();
}

void Solver::initialize_common() {
    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(), "{} initializing...", name());

    // Nodes appended since the last coupling access have not grown the
    // matrices yet.
    tnw.synchronize_structure();
    _resolved = resolve(_settings, resolve_context(_solver_type, tmm));
    if (!_resolved.not_applied.empty()) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "{}: settings not applied: {}", name(),
                           describe_not_applied(_resolved.not_applied));
    }
    for (const auto& warning : _resolved.warnings) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "{}: {}", name(), warning);
    }
    _initialized_structure = tmm.structure_versions();

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

void Solver::refresh_live_settings() {
    check_model_unchanged();
    // The INIT fields cannot have changed while the solver is initialised,
    // so resolving again only picks up the LIVE ones.
    _resolved = resolve(_settings, resolve_context(_solver_type, tmm));
}

void Solver::check_model_unchanged(std::string_view callback) const {
    const StructureVersions now = tmm.structure_versions();
    if (now == _initialized_structure) {
        return;
    }
    std::string message = std::string(name()) + ": the model's " +
                          describe_changes(_initialized_structure, now) +
                          " changed ";
    if (callback.empty()) {
        message += "after initialize()";
    } else {
        message += "in the ";
        message += callback;
        message += " callback";
    }
    message += "; call initialize() again";
    throw ModelChangedError(message);
}

void Solver::callback_transient_time_change() {
    PYCANHA_PROFILE_SCOPE("Callback Time Change");
    {
        const ActiveSolverCallbackScope active_solver_scope(tmm, *this);
        _tmm_shptr->callback_transient_time_change();
    }
    check_model_unchanged("time change");
}

void Solver::callback_solver_loop() {
    PYCANHA_PROFILE_SCOPE("Callback Solver Loop");
    {
        const ActiveSolverCallbackScope active_solver_scope(tmm, *this);
        _tmm_shptr->callback_solver_loop();
    }
    check_model_unchanged("solver loop");
}

void Solver::callback_transient_after_timestep() {
    PYCANHA_PROFILE_SCOPE("Callback Timestep");
    {
        const ActiveSolverCallbackScope active_solver_scope(tmm, *this);
        _tmm_shptr->callback_transient_after_timestep();
    }
    check_model_unchanged("after time step");
}

void Solver::prepare_formulas_for_execution() {
    tmm.formulas().validate_for_execution();
    tmm.formulas().compile_formulas();
    tmm.formulas().lock_parameters_for_execution();
}

void Solver::release_formulas_after_execution() noexcept {
    tmm.formulas().unlock_parameters();
}

bool Solver::temperature_convergence_check() {
    PYCANHA_PROFILE_SCOPE("Convergence Check");
    dTd = Td_solver - Td;
    max_dT = dTd.cwiseAbs().maxCoeff();
    return max_dT < _resolved.tolerance;
}

}  // namespace pycanha::solvers
