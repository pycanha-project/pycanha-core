#pragma once

#include <Eigen/Sparse>
#include <cstdint>
#include <span>

namespace pycanha::solvers {

/// What a Retracer did during the last solve, copied into the SolverReport.
struct RetracerCounters {
    /// Traces made since begin_solve().
    int retraces = 0;
    /// Couplings of the last trace with no slot in the model's radiative
    /// matrices, carried as the internal load.
    std::int64_t overflow_couplings = 0;
    /// Sum of those couplings over the sum of every coupling of the last
    /// trace [m^2/m^2].
    double overflow_share = 0.0;
};

/// Radiative couplings that are recomputed during a solve.
/**
 * A solver with a retracer (Solver::set_retracer) calls it at fixed points:
 * begin_solve() before its iteration loop, update() right before every
 * model callback, accepts_convergence() right after every linear solve and
 * end_solve() after the loop. The order at a callback is therefore
 * retracer, formulas, user callbacks: a formula on a radiative coupling the
 * retracer also writes wins.
 *
 * The retracer writes the couplings that have a slot in the model's radiative
 * matrices as plain values; never inserting one during a solve, it leaves
 * the solver's matrix patterns valid. The couplings without a slot (the
 * overflow) enter as an internal load, added to the node loads in every load
 * sum, and as overflow() in the radiative coupling output.
 */
class Retracer {
  public:
    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    Retracer() = default;
    virtual ~Retracer() = default;
    Retracer(const Retracer&) = delete;
    Retracer& operator=(const Retracer&) = delete;
    Retracer(Retracer&&) = delete;
    Retracer& operator=(Retracer&&) = delete;

    /// Before the iteration loop: traces again when the temperatures moved
    /// since the last trace, writes its couplings and computes the internal
    /// load. Throws when the model no longer matches the retracer.
    virtual void begin_solve() = 0;
    /// Before every model callback: traces again when due and recomputes
    /// the internal load at the current temperatures.
    virtual void update() = 0;
    /// Called right after a linear solve, whose couplings came from the
    /// latest trace, once the solver's own test passed: whether the
    /// temperatures are close enough to that trace's for the solver to
    /// accept convergence. `tolerance` [K] is the solver's.
    [[nodiscard]] virtual bool accepts_convergence(double tolerance) const = 0;
    /// One value per node [W], in the model's internal node order; empty
    /// when there is no overflow.
    [[nodiscard]] virtual std::span<const double> internal_load() const = 0;
    /// The overflow couplings [m^2], upper triangle by internal node index
    /// (diffusive nodes first), N x N; empty (0 x 0) when there is none.
    [[nodiscard]] virtual const SparseMatrix& overflow() const = 0;
    [[nodiscard]] virtual RetracerCounters counters() const = 0;
    /// After the iteration loop. Returns true when it changed the model's
    /// coupling structure, after which the solver deinitializes itself.
    virtual bool end_solve() = 0;
};

}  // namespace pycanha::solvers
