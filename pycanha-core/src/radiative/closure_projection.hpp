#pragma once

// The constrained-least-squares triangulation mode, shared by both matrix
// assemblies.
//
// Triangulation makes reciprocity structural — one stored number serves both
// directions — but it breaks ROW CLOSURE: the raw estimate of a row summed
// to exactly the energy that row emitted, and a weighted blend of two
// directions does not. The usual repair is to renormalise rows afterwards,
// which partly undoes the reciprocity just imposed. This mode instead asks
// for the matrix closest to the raw estimate, in the inverse-variance
// metric, that closes every row exactly:
//
//     minimise    sum_k (x_k - r_k)^2 / v_k
//     subject to  sum_{k in row i} x_k = T_i        for every closed row i
//
// with r_k the minimum-variance combination of the two directions, v_k its
// variance and T_i the row target (A_i for view factors, A_i eps_i for
// exchange). Reciprocity is not a constraint here: the shared entry IS the
// unknown, so it cannot be violated.
//
// Eliminating x from the KKT system leaves a much smaller dual problem
//
//     (C V C^T) lambda = T - C r ,      x = r + V C^T lambda
//
// where C is the row-membership matrix and V = diag(v). The matrix C V C^T
// is symmetric positive semi-definite, has EXACTLY the sparsity of the
// coupling matrix plus a diagonal, and is only num_slots on a side — so one
// sparse Cholesky replaces a factorization of the full unknown vector. Its
// right-hand side is the per-row closure deficit, which is also what the
// multipliers price.

#include <cstddef>
#include <span>

#include "pair_walk.hpp"

namespace pycanha::radiative::detail {

// The minimum-variance combination of the two directional estimates of one
// coupling, and the variance of that combination.
//
// `forward`/`backward` are the two estimates of the SAME extensive quantity;
// `u_forward`/`u_backward` are their variance proxies (A/N for both
// assemblies), zero for a direction that carries no samples. The returned
// variance is proportional to the estimate itself and carries only the
// factors that differ between pairs — a caller whose kernel adds a further
// per-pair factor (the exchange path's eps_i eps_j) multiplies it in.
struct BlueEstimate {
    double value = 0.0;
    double variance = 0.0;
};

[[nodiscard]] BlueEstimate blue_combine(double forward, double backward,
                                        double u_forward, double u_backward);

// Moves every stored entry the smallest inverse-variance-weighted distance
// that makes each constrained row close exactly, in place.
//
// `rows` must carry a `variances` entry per value (blue_combine's, scaled by
// whatever the caller's variance model adds) and must NOT have been
// thresholded yet: a dropped entry is one the projection cannot use, and
// closure would be imposed on a row that is missing part of itself.
//
// `targets[i] <= 0` marks a row that carries no constraint — one that
// emitted nothing, or whose emissive area is zero. Such a row is left alone
// and contributes no equation, which is also what keeps the system
// non-singular when a face never emitted.
//
// Throws if the system cannot be factorized or the solution is not finite.
void project_onto_closure(std::span<RowEntries> rows,
                          std::span<const double> targets, std::size_t slots);

}  // namespace pycanha::radiative::detail
