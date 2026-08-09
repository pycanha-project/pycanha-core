// Constrained-least-squares closure projection (see closure_projection.hpp
// for the derivation).
//
// The blend x = r + v * lambda is the same a + f * (b - a) shape the weight
// table has, so this file is compiled with -ffp-contract=off for the same
// reason: an FMA rounds once instead of twice and would make the result
// depend on whether the compiler chose to contract.

#include "closure_projection.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "pair_walk.hpp"

namespace pycanha::radiative::detail {

namespace {

// What each row needs from the projection: how far its raw closure missed
// the target, and how much the entries it owns are able to move (which is
// also the diagonal of C V C^T).
struct RowBudget {
    std::vector<double> capacity;
    std::vector<double> sum;
    // 1 for a row that carries a closure equation. A row with no target, or
    // whose entries cannot move at all, gets none: the second case would
    // otherwise put a zero on the diagonal and make the system singular for
    // a reason that is not the geometry's fault.
    std::vector<char> constrained;
};

// One pass over the stored entries. An entry (i, j) with j < slots belongs
// to BOTH row i's and row j's constraint — that is exactly what makes the
// system couple rows instead of decomposing into independent per-row
// renormalizations. The diagonal entry (i, i) belongs to row i once.
[[nodiscard]] RowBudget row_budget(std::span<const RowEntries> rows,
                                   std::span<const double> targets,
                                   std::size_t slots) {
    RowBudget budget;
    budget.capacity.assign(rows.size(), 0.0);
    budget.sum.assign(rows.size(), 0.0);
    budget.constrained.assign(rows.size(), 0);
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const RowEntries& entries = rows[row];
        for (std::size_t at = 0; at < entries.values.size(); ++at) {
            const auto column = static_cast<std::size_t>(entries.columns[at]);
            budget.capacity[row] += entries.variances[at];
            budget.sum[row] += entries.values[at];
            if (column < slots && column != row) {
                budget.capacity[column] += entries.variances[at];
                budget.sum[column] += entries.values[at];
            }
        }
    }
    for (std::size_t row = 0; row < rows.size(); ++row) {
        budget.constrained[row] =
            static_cast<char>(targets[row] > 0.0 && budget.capacity[row] > 0.0);
    }
    return budget;
}

struct ClosureSystem {
    Eigen::SparseMatrix<double> matrix;
    Eigen::VectorXd deficit;
};

[[nodiscard]] ClosureSystem build_system(std::span<const RowEntries> rows,
                                         std::span<const double> targets,
                                         const RowBudget& budget,
                                         std::size_t slots) {
    const std::size_t count = rows.size();
    const auto size = static_cast<Eigen::Index>(count);
    // Triplets are indexed by the matrix's own StorageIndex, which is
    // narrower than Eigen::Index; feeding them the wider type is a silent
    // truncation that MSVC rejects outright.
    using TripletIndex = Eigen::SparseMatrix<double>::StorageIndex;
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(count * 3);
    ClosureSystem system{.matrix = Eigen::SparseMatrix<double>(size, size),
                         .deficit = Eigen::VectorXd::Zero(size)};
    for (std::size_t row = 0; row < count; ++row) {
        const auto index = static_cast<TripletIndex>(row);
        if (budget.constrained[row] == 0) {
            // An identity equation with a zero right-hand side: the row's
            // multiplier comes out exactly zero, so it neither moves its own
            // entries nor leaks into anyone else's.
            triplets.emplace_back(index, index, 1.0);
            continue;
        }
        system.deficit(index) = targets[row] - budget.sum[row];
        triplets.emplace_back(index, index, budget.capacity[row]);
    }
    for (std::size_t row = 0; row < count; ++row) {
        if (budget.constrained[row] == 0) {
            continue;
        }
        const RowEntries& entries = rows[row];
        for (std::size_t at = 0; at < entries.values.size(); ++at) {
            const auto column = static_cast<std::size_t>(entries.columns[at]);
            if (column >= slots || column == row ||
                budget.constrained[column] == 0) {
                continue;
            }
            const auto index = static_cast<TripletIndex>(row);
            const auto other = static_cast<TripletIndex>(column);
            triplets.emplace_back(index, other, entries.variances[at]);
            triplets.emplace_back(other, index, entries.variances[at]);
        }
    }
    system.matrix.setFromTriplets(triplets.begin(), triplets.end());
    return system;
}

// True when no pivot has collapsed against its own row's scale. The
// factorization does not pivot, so a rank-deficient system shows up that way
// rather than as a failure code, and comparing each pivot to the
// corresponding diagonal of the system keeps the test free of the wide scale
// spread between a capacity and an identity row.
[[nodiscard]] bool pivots_are_sound(
    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver,
    const Eigen::SparseMatrix<double>& matrix) {
    constexpr double pivot_tolerance = 1e-12;
    const Eigen::VectorXd pivots = solver.vectorD();
    for (Eigen::Index index = 0; index < matrix.rows(); ++index) {
        const double scale = matrix.coeff(index, index);
        if (scale > 0.0 &&
            !(std::abs(pivots(index)) > pivot_tolerance * scale)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] Eigen::VectorXd solve_multipliers(const ClosureSystem& system) {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(system.matrix);
    Eigen::VectorXd multiplier;
    if (solver.info() == Eigen::Success &&
        pivots_are_sound(solver, system.matrix)) {
        multiplier = solver.solve(system.deficit);
        if (solver.info() == Eigen::Success &&
            multiplier.size() == system.matrix.rows() &&
            multiplier.allFinite()) {
            return multiplier;
        }
    }
    throw std::runtime_error(
        "pycanha::radiative: the closure system is singular, so the "
        "constrained least-squares triangulation has no unique solution. The "
        "row-closure equations are not independent — a closed enclosure whose "
        "couplings form a bipartite graph and whose rows have no space column "
        "is the usual case. Use TriangulationMode::RayDensity for such a "
        "model.");
}

void apply_correction(std::span<RowEntries> rows,
                      const Eigen::VectorXd& multiplier, std::size_t slots) {
    const std::size_t count = rows.size();
    const unsigned threads = worker_count({}, count, count);
    parallel_for_index(count, threads, [&](std::size_t row) {
        RowEntries& entries = rows[row];
        const double own = multiplier(static_cast<Eigen::Index>(row));
        for (std::size_t at = 0; at < entries.values.size(); ++at) {
            const auto column = static_cast<std::size_t>(entries.columns[at]);
            double total = own;
            if (column < slots && column != row) {
                total += multiplier(static_cast<Eigen::Index>(column));
            }
            entries.values[at] += entries.variances[at] * total;
        }
    });
}

}  // namespace

BlueEstimate blue_combine(double forward, double backward, double u_forward,
                          double u_backward) {
    const bool has_forward = u_forward > 0.0;
    const bool has_backward = u_backward > 0.0;
    if (has_forward && has_backward) {
        // Var(forward) is proportional to u_forward and Var(backward) to
        // u_backward, so the minimum-variance weight of the forward estimate
        // is u_backward / (u_forward + u_backward) and the combined variance
        // is their harmonic mean.
        const double total = u_forward + u_backward;
        const double weight = u_backward / total;
        const double value = (weight * forward) + ((1.0 - weight) * backward);
        // The magnitude, because the exchange lost column accumulates
        // Russian-roulette withdrawals and can come out negative; a negative
        // variance would make the dual system indefinite for a reason that
        // has nothing to do with how well that entry is known.
        return {.value = value,
                .variance = std::abs(value) * (u_forward * u_backward / total)};
    }
    // Only one direction carries samples; the other has an infinite variance
    // proxy, so the weight passes entirely to the one that does.
    if (has_forward) {
        return {.value = forward, .variance = std::abs(forward) * u_forward};
    }
    if (has_backward) {
        return {.value = backward, .variance = std::abs(backward) * u_backward};
    }
    // Nothing was sampled either way: nothing to move and nothing to prefer.
    return {.value = forward, .variance = 0.0};
}

void project_onto_closure(std::span<RowEntries> rows,
                          std::span<const double> targets, std::size_t slots) {
    if (targets.size() != rows.size()) {
        throw std::invalid_argument(
            "pycanha::radiative: the closure projection needs one target per "
            "matrix row");
    }
    if (rows.empty()) {
        return;  // nothing to close, and no system to build
    }
    const RowBudget budget = row_budget(rows, targets, slots);
    const ClosureSystem system = build_system(rows, targets, budget, slots);
    apply_correction(rows, solve_multipliers(system), slots);
}

}  // namespace pycanha::radiative::detail
