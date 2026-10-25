#include "pycanha-core/solvers/steady_state.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "factorizer.hpp"
#include "memory.hpp"
#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/profiling.hpp"

namespace pycanha::solvers::steady {

namespace {

// Position in the value array of the stored entry (row, col) of a compressed
// row-major matrix.
int value_index(const SpMatRow& matrix, Index row, Index col) {
    const std::span<const int> outer(matrix.outerIndexPtr(),
                                     to_sizet(matrix.outerSize() + 1));
    const std::span<const int> inner(matrix.innerIndexPtr(),
                                     to_sizet(matrix.nonZeros()));
    const auto first = inner.begin() + outer[to_sizet(row)];
    const auto last = inner.begin() + outer[to_sizet(row + 1)];
    const auto found = std::lower_bound(first, last, static_cast<int>(col));
    PYCANHA_ASSERT(found != last && *found == col,
                   "Entry missing from the solver matrix pattern");
    return static_cast<int>(found - inner.begin());
}

// For every stored value of source, in storage order, the position in
// target's value array of the same entry, or of its transpose.
std::vector<int> map_values(const SpMatRow& source, const SpMatRow& target,
                            bool transposed) {
    std::vector<int> indices;
    indices.reserve(to_sizet(source.nonZeros()));
    for (Index row = 0; row < source.outerSize(); ++row) {
        for (SpMatRow::InnerIterator it(source, row); it; ++it) {
            indices.push_back(transposed ? value_index(target, it.col(), row)
                                         : value_index(target, row, it.col()));
        }
    }
    return indices;
}

// dest[indices[k]] += scale * source[k]
void scatter_add(std::span<double> dest, std::span<const double> source,
                 const std::vector<int>& indices, double scale) {
    for (std::size_t k = 0; k < indices.size(); ++k) {
        dest[to_sizet(indices[k])] += scale * source[k];
    }
}

std::span<double> mutable_values_of(SpMatRow& matrix) {
    return {matrix.valuePtr(), to_sizet(matrix.nonZeros())};
}

std::span<const double> values_of(const SpMatRow& matrix) {
    return {matrix.valuePtr(), to_sizet(matrix.nonZeros())};
}

}  // namespace

SteadyState::SteadyState(std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : Solver(std::move(tmm_shptr), SolverType::STEADY_STATE) {}

SteadyState::~SteadyState() = default;

bool SteadyState::has_radiation() const noexcept {
    return KRdd.nonZeros() != 0 || KRdb.nonZeros() != 0;
}

void SteadyState::initialize() {
    const auto start = Clock::now();
    begin_initialize_report();
    if (preparation_matches()) {
        // estimate_memory() built the pattern and ran the analysis.
        initialize_common();
    } else {
        prepare(/*log_warnings=*/true);
        if (!_prepared) {
            return;
        }
    }
    _has_factors = false;
    if (const auto hint =
            detail::min_degree_hint(*_factorizer, _resolved, _symmetric, nd)) {
        warn(*hint);
    }
    SPDLOG_LOGGER_INFO(get_logger(), "{} initializing ({})", name(),
                       detail::describe(_resolved, _symmetric));
    _prepared = false;
    _initialized = true;
    end_initialize_report(start);
}

void SteadyState::prepare(bool log_warnings) {
    release();
    // Throws before anything is built when the settings contradict.
    initialize_common(log_warnings);
    _symmetric = _resolved.factorization == Factorization::CHOLESKY;
    _radiation = has_radiation();
    _factorizer = detail::make_factorizer(_resolved, _symmetric);

    build_pattern();

    _b = VectorXd::Zero(nd);
    _factorized_values = VectorXd::Zero(_a_matrix.nonZeros());
    _t_cubed_domain = VectorXd::Zero(nd);
    _t_fourth_boundary = VectorXd::Zero(nb);
    _radiation_linear_term = VectorXd::Zero(nd);
    _row_sums = VectorXd::Zero(nd);
    _ones_domain = VectorXd::Ones(nd);
    _ones_boundary = VectorXd::Ones(nb);
    _has_factors = false;

    _factorizer->analyze_pattern(_a_matrix);
    if (!_factorizer->succeeded()) {
        SPDLOG_LOGGER_ERROR(get_logger(), "{}: analysis failed ({}).", name(),
                            _factorizer->error_message());
        release();
        return;
    }
    mark_prepared();
}

SolverMemory SteadyState::memory_now() const {
    using detail::bytes_of;
    SolverMemory memory;
    if (_factorizer) {
        memory.factors = _factorizer->factor_bytes();
        memory.exact = _factorizer->factor_bytes_exact();
        memory.matrix = _factorizer->matrix_copy_bytes();
    }
    memory.matrix += bytes_of(_a_matrix) + bytes_of(_b) +
                     bytes_of(_factorized_values) + bytes_of(_t_cubed_domain) +
                     bytes_of(_t_fourth_boundary) +
                     bytes_of(_radiation_linear_term) + bytes_of(_row_sums) +
                     bytes_of(_ones_domain) + bytes_of(_ones_boundary) +
                     bytes_of(_upper_kl_indices) + bytes_of(_lower_kl_indices) +
                     bytes_of(_upper_kr_indices) + bytes_of(_lower_kr_indices) +
                     bytes_of(_diagonal_indices) + common_bytes();
    return memory;
}

void SteadyState::build_pattern() {
    if (_symmetric) {
        // Upper triangle with the diagonal: KLdd is already stored that way.
        _a_matrix = KLdd;
        sparse_utils::add_zero_diag_square(_a_matrix);
        _upper_kl_indices = map_values(KLdd, _a_matrix, /*transposed=*/false);
        _lower_kl_indices.clear();
        _upper_kr_indices.clear();
        _lower_kr_indices.clear();
    } else {
        _a_matrix = KLdd.selfadjointView<Eigen::Upper>();
        if (_radiation) {
            _a_matrix += KRdd.selfadjointView<Eigen::Upper>();
        }
        sparse_utils::add_zero_diag_square(_a_matrix);
        _upper_kl_indices = map_values(KLdd, _a_matrix, /*transposed=*/false);
        _lower_kl_indices = map_values(KLdd, _a_matrix, /*transposed=*/true);
        _upper_kr_indices.clear();
        _lower_kr_indices.clear();
        if (_radiation) {
            _upper_kr_indices =
                map_values(KRdd, _a_matrix, /*transposed=*/false);
            _lower_kr_indices =
                map_values(KRdd, _a_matrix, /*transposed=*/true);
        }
    }

    _diagonal_indices.resize(to_sizet(nd));
    for (Index row = 0; row < nd; ++row) {
        _diagonal_indices[to_sizet(row)] = value_index(_a_matrix, row, row);
    }
    sparse_utils::set_to_zero(_a_matrix);
}

void SteadyState::assemble_matrix() {
    PYCANHA_PROFILE_SCOPE("Steady state assemble matrix");
    sparse_utils::set_to_zero(_a_matrix);
    const auto values = mutable_values_of(_a_matrix);
    const std::span<const double> diagonal_sums(_row_sums.data(), to_sizet(nd));

    if (_radiation) {
        const auto kr_values = values_of(std::as_const(KRdd));
        scatter_add(values, kr_values, _upper_kr_indices, -1.0);
        scatter_add(values, kr_values, _lower_kr_indices, -1.0);
        _row_sums.noalias() =
            KRdd.selfadjointView<Eigen::Upper>() * _ones_domain;
        _row_sums.noalias() += KRdb * _ones_boundary;
        for (std::size_t row = 0; row < _diagonal_indices.size(); ++row) {
            values[to_sizet(_diagonal_indices[row])] += diagonal_sums[row];
        }

        // Newton linearisation of sigma T^4: column j carries 4 sigma Tj^3.
        const Eigen::Map<const Eigen::VectorXi> columns(
            _a_matrix.innerIndexPtr(), _a_matrix.nonZeros());
        _a_matrix.coeffs() *= _t_cubed_domain(columns).array();
        _radiation_linear_term.noalias() = _a_matrix * Td;
    }

    const auto kl_values = values_of(std::as_const(KLdd));
    scatter_add(values, kl_values, _upper_kl_indices, -1.0);
    scatter_add(values, kl_values, _lower_kl_indices, -1.0);
    _row_sums.noalias() = KLdd.selfadjointView<Eigen::Upper>() * _ones_domain;
    _row_sums.noalias() += KLdb * _ones_boundary;
    for (std::size_t row = 0; row < _diagonal_indices.size(); ++row) {
        values[to_sizet(_diagonal_indices[row])] += diagonal_sums[row];
    }
}

void SteadyState::assemble_rhs() {
    PYCANHA_PROFILE_SCOPE("Steady state assemble rhs");
    Q.setZero();
    Q += QI_sp;
    Q += QS_sp;
    Q += QA_sp;
    Q += QE_sp;
    Q += QR_sp;

    _b = Q.head(nd);
    _b.noalias() += KLdb * Tb;
    if (_radiation) {
        _t_fourth_boundary = STF_BOLTZ * Tb.array().square().square();
        _b.noalias() += KRdb * _t_fourth_boundary;
        _b += 0.75 * _radiation_linear_term;
    }
}

bool SteadyState::factorize_and_solve_step() {
    PYCANHA_PROFILE_SCOPE("Steady state factorize and solve");
    auto& factorizer = *_factorizer;
    const auto start = Clock::now();
    const auto values = values_of(std::as_const(_a_matrix));
    const std::span<const double> factorized(_factorized_values.data(),
                                             values.size());
    bool factors_recomputed = true;

    if (_has_factors && std::ranges::equal(values, factorized)) {
        factorizer.solve(_a_matrix, _b, Td_solver);
        factors_recomputed = false;
        SPDLOG_LOGGER_DEBUG(get_logger(),
                            "{} iteration {}: matrix unchanged, solved with "
                            "the existing factors.",
                            name(), solver_iter + 1);
    } else if (_has_factors && factorizer.reuses_factors()) {
        factorizer.factorize_and_solve(_a_matrix, _b, Td_solver);
        factors_recomputed = factorizer.factors_recomputed();
        SPDLOG_LOGGER_DEBUG(get_logger(),
                            "{} iteration {}: iterative step, factors {}.",
                            name(), solver_iter + 1,
                            factors_recomputed ? "recomputed" : "reused");
    } else {
        factorizer.factorize(_a_matrix);
        if (factorizer.succeeded()) {
            factorizer.solve(_a_matrix, _b, Td_solver);
        }
        SPDLOG_LOGGER_DEBUG(get_logger(), "{} iteration {}: factorized.",
                            name(), solver_iter + 1);
    }

    if (!factorizer.succeeded()) {
        if (factorizer.zero_pivot()) {
            SPDLOG_LOGGER_ERROR(
                get_logger(),
                "{}: zero pivot in the {} factorization ({}). The matrix is "
                "singular: a group of diffusive nodes has no conductive or "
                "radiative path to a boundary node.",
                name(), _symmetric ? "Cholesky" : "LU",
                factorizer.error_message());
        } else {
            SPDLOG_LOGGER_ERROR(get_logger(), "{}: {}.", name(),
                                factorizer.error_message());
        }
        _has_factors = false;
        return false;
    }

    // PARDISO replaces a pivot below 1e-13 relative to the matrix by that
    // value. A matrix diagonally dominant by columns never needs it unless
    // it is singular, and the solution is then meaningless.
    if (factors_recomputed && factorizer.perturbed_pivots() > 0) {
        SPDLOG_LOGGER_ERROR(
            get_logger(),
            "{}: {} pivots perturbed in the LU factorization. The matrix is "
            "singular: a group of diffusive nodes has no conductive or "
            "radiative path to a boundary node.",
            name(), factorizer.perturbed_pivots());
        _has_factors = false;
        return false;
    }

    if (factors_recomputed) {
        _factorized_values = Eigen::Map<const VectorXd>(_a_matrix.valuePtr(),
                                                        _a_matrix.nonZeros());
        _has_factors = true;
    }
    record_linear_solve(start, factors_recomputed);
    return true;
}

void SteadyState::solve() {
    if (!_initialized) {
        SPDLOG_LOGGER_ERROR(
            get_logger(),
            "{} has not been initialized. Call initialize() before solve().",
            name());
        return;
    }
    const auto start = Clock::now();
    begin_solve_report();
    refresh_live_settings();
    _factorizer->apply_live_settings(_resolved);
    SPDLOG_LOGGER_INFO(get_logger(), "{} solving...", name());

    const FormulaExecutionGuard formula_execution(*this);

    _converged = false;
    int iterations_done = 0;
    bool converged = false;

    for (solver_iter = 0; solver_iter < _resolved.max_iterations;
         ++solver_iter) {
        PYCANHA_PROFILE_SCOPE("Steady state iteration");
        iterations_done = solver_iter + 1;

        if (_radiation) {
            _t_cubed_domain = (4.0 * STF_BOLTZ) * Td.array().cube();
        }
        assemble_matrix();
        assemble_rhs();

        if (!factorize_and_solve_step()) {
            finish_solve_report(start, /*failed=*/true);
            return;
        }

        dTd = Td_solver - Td;
        Td = Td_solver;
        max_dT = dTd.cwiseAbs().maxCoeff();

        this->callback_solver_loop();

        if (max_dT < _resolved.tolerance) {
            SPDLOG_LOGGER_INFO(get_logger(),
                               "{} converged. Num. iters: {}. Max. dT = {} K. "
                               "Factorizations: {}.",
                               name(), iterations_done, max_dT,
                               _report.factorizations);
            converged = true;
            break;
        }
    }

    record_steady_iterations(iterations_done, converged);
    finish_solve_report(start);
}

void SteadyState::deinitialize() { release(); }

void SteadyState::release() noexcept {
    _factorizer.reset();
    _prepared = false;
    _has_factors = false;
    _symmetric = false;
    _a_matrix = SpMatRow();
    _factorized_values.resize(0);
    _initialized = false;
}

}  // namespace pycanha::solvers::steady
