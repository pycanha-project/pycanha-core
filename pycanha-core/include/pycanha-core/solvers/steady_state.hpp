#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "pycanha-core/solvers/solver.hpp"

namespace pycanha::solvers::detail {
class Factorizer;
}  // namespace pycanha::solvers::detail

namespace pycanha::solvers::steady {

/// Steady state of the network.
/**
 * Newton linearisation of the radiation, assembled in place into a matrix
 * whose pattern is built once in initialize(), and solved with a sparse
 * direct factorization (see Settings: backend, factorization, reuse_factors,
 * pardiso.*, eigen.*). Each iteration refactorizes only when the matrix
 * values differ from those of the last factorization, so the second
 * iteration of a linear model, or a new solve of an unchanged model, costs
 * one solve with the existing factors.
 *
 * A singular matrix (a group of diffusive nodes without a path to a boundary
 * node) is reported as an ERROR and leaves the temperatures of that solve
 * untouched; a solve that does not converge logs an ERROR.
 */
class SteadyState : public Solver {
  public:
    explicit SteadyState(std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~SteadyState() override;

    SteadyState(const SteadyState&) = delete;
    SteadyState& operator=(const SteadyState&) = delete;
    SteadyState(SteadyState&&) = delete;
    SteadyState& operator=(SteadyState&&) = delete;

    void initialize() override;
    void solve() override;
    void deinitialize() override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "SteadyState";
    }

    /// The last initialize() chose a Cholesky factorization.
    [[nodiscard]] bool uses_cholesky() const noexcept { return _symmetric; }
    /// Numerical factorizations done by the last solve().
    [[nodiscard]] int num_factorizations() const noexcept {
        return _num_factorizations;
    }
    /// Iterations done by the last solve().
    [[nodiscard]] int iterations() const noexcept { return _iterations; }

  private:
    // A = -K and b = -Qd of the linearised system K T = Qd, so the diagonal
    // is positive, which Cholesky needs.
    SpMatRow _a_matrix;
    VectorXd _b;
    VectorXd _factorized_values;
    bool _has_factors = false;

    VectorXd _t_cubed_domain;
    VectorXd _t_fourth_boundary;
    VectorXd _radiation_linear_term;
    VectorXd _row_sums;
    VectorXd _ones_domain;
    VectorXd _ones_boundary;

    std::vector<int> _upper_kl_indices;
    std::vector<int> _lower_kl_indices;
    std::vector<int> _upper_kr_indices;
    std::vector<int> _lower_kr_indices;
    std::vector<int> _diagonal_indices;

    int _num_factorizations = 0;
    int _iterations = 0;
    bool _symmetric = false;
    bool _radiation = false;

    std::unique_ptr<detail::Factorizer> _factorizer;

    [[nodiscard]] bool has_radiation() const noexcept;
    void build_pattern();
    void assemble_matrix();
    void assemble_rhs();
    [[nodiscard]] bool factorize_and_solve_step();
    void release() noexcept;
};

}  // namespace pycanha::solvers::steady
