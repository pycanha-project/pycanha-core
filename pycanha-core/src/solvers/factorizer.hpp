#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>
#include <string>
#include <string_view>

#include "pycanha-core/solvers/settings.hpp"

namespace pycanha::solvers::detail {

/// Sparse direct factorization of A x = b for a compressed row-major matrix
/// whose pattern is fixed between analyze_pattern() and the next one.
class Factorizer {
  public:
    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    Factorizer() = default;
    virtual ~Factorizer() = default;
    Factorizer(const Factorizer&) = delete;
    Factorizer& operator=(const Factorizer&) = delete;
    Factorizer(Factorizer&&) = delete;
    Factorizer& operator=(Factorizer&&) = delete;

    virtual void analyze_pattern(const SparseMatrix& matrix) = 0;
    virtual void factorize(const SparseMatrix& matrix) = 0;
    /// With the current factors. b is not modified.
    virtual void solve(const SparseMatrix& matrix, Eigen::VectorXd& b,
                       Eigen::VectorXd& x) = 0;
    /// Several right-hand sides (columns) with the current factors.
    virtual void solve(const SparseMatrix& matrix, Eigen::MatrixXd& b,
                       Eigen::MatrixXd& x) = 0;
    /// Factorize and solve. With reuse, the previous factors may be iterated
    /// on instead (see factors_recomputed()).
    virtual void factorize_and_solve(const SparseMatrix& matrix,
                                     Eigen::VectorXd& b, Eigen::VectorXd& x);

    /// Applies the settings that may change between solves (threads,
    /// verbosity).
    virtual void apply_live_settings(const ResolvedSettings& /*settings*/) {}

    [[nodiscard]] virtual bool succeeded() const noexcept = 0;
    /// The last failure was a zero pivot (a singular matrix).
    [[nodiscard]] virtual bool zero_pivot() const noexcept = 0;
    [[nodiscard]] virtual std::string error_message() const = 0;
    /// Pivots PARDISO replaced by a small value, 0 for Eigen.
    [[nodiscard]] virtual Eigen::Index perturbed_pivots() const noexcept {
        return 0;
    }
    /// Non-zeros of the factors (PARDISO predicts it at the analysis), -1
    /// when unknown.
    [[nodiscard]] virtual Eigen::Index factor_nonzeros() const noexcept {
        return -1;
    }
    /// False after factorize_and_solve() only when it iterated on the
    /// previous factors, which then still belong to an older matrix.
    [[nodiscard]] virtual bool factors_recomputed() const noexcept {
        return true;
    }
    /// Whether factorize_and_solve() may iterate on the previous factors.
    [[nodiscard]] virtual bool reuses_factors() const noexcept { return false; }
};

/// The factorization the resolved settings select: PARDISO (two-level or
/// one-level, nested dissection or minimum degree, with or without reuse),
/// Eigen's sparse LU (COLAMD or AMD) or Eigen's LDLT. `symmetric` selects the
/// Cholesky form, which takes the upper triangle only. Throws
/// std::invalid_argument for an iparm index out of range.
[[nodiscard]] std::unique_ptr<Factorizer> make_factorizer(
    const ResolvedSettings& settings, bool symmetric);

/// For the logs: "PARDISO TWO_LEVEL NESTED_DISSECTION, Cholesky".
[[nodiscard]] std::string describe(const ResolvedSettings& settings,
                                   bool symmetric);

/// Logs a warning suggesting the minimum degree ordering when PARDISO
/// predicts a large, near-dense factor for nested dissection.
void suggest_min_degree_if_dense(const Factorizer& factorizer,
                                 const ResolvedSettings& settings,
                                 bool symmetric, Eigen::Index size,
                                 std::string_view solver_name);

}  // namespace pycanha::solvers::detail
