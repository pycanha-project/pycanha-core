#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "factorizer.hpp"
#include "linearized_network.hpp"
#include "memory.hpp"
#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"

namespace pycanha::solvers::detail {

namespace {

// Position in the value array of the stored entry (row, col) of a compressed
// row-major matrix.
int value_index(const SparseMatrix& matrix, Index row, Index col) {
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
std::vector<int> map_values(const SparseMatrix& source,
                            const SparseMatrix& target, bool transposed) {
    std::vector<int> indices;
    indices.reserve(to_sizet(source.nonZeros()));
    for (Index row = 0; row < source.outerSize(); ++row) {
        for (SparseMatrix::InnerIterator it(source, row); it; ++it) {
            indices.push_back(transposed ? value_index(target, it.col(), row)
                                         : value_index(target, row, it.col()));
        }
    }
    return indices;
}

std::span<const double> values_of(const SparseMatrix& matrix) {
    return {matrix.valuePtr(), to_sizet(matrix.nonZeros())};
}

// The matrix in compressed row-major storage, with a pattern fixed at
// construction: both triangles, or the upper one in the Cholesky form. Every
// linearize() writes the values in place through index tables, and the
// factorizer reads them without a copy.
class SparseNetwork final : public LinearizedNetwork {
  public:
    SparseNetwork(const ResolvedSettings& settings,
                  const NetworkCouplings& couplings,
                  const NetworkOptions& options)
        : _couplings(couplings),
          _options(options),
          _factorizer(make_factorizer(settings, options.symmetric)) {
        build_pattern();
        const Index nd = _matrix.rows();
        _row_sums = Eigen::VectorXd::Zero(nd);
        _ones_domain = Eigen::VectorXd::Ones(nd);
        _ones_boundary =
            Eigen::VectorXd::Ones(couplings.conductive_boundary->cols());
    }

    bool analyze() override {
        _factorizer->analyze_pattern(_matrix);
        _has_factors = false;
        return _factorizer->succeeded();
    }

    void allocate() override {}

    void linearize(const Eigen::VectorXd& t3, const Eigen::VectorXd& t,
                   Eigen::VectorXd& radiation) override {
        sparse_utils::set_to_zero(_matrix);
        const std::span<double> values(_matrix.valuePtr(),
                                       to_sizet(_matrix.nonZeros()));
        if (_options.radiation) {
            const auto radiative = values_of(*_couplings.radiative);
            for (std::size_t k = 0; k < radiative.size(); ++k) {
                values[to_sizet(_upper_kr[k])] = -radiative[k];
                values[to_sizet(_lower_kr[k])] = -radiative[k];
            }
            _row_sums.noalias() =
                _couplings.radiative->selfadjointView<Eigen::Upper>() *
                _ones_domain;
            _row_sums.noalias() +=
                *_couplings.radiative_boundary * _ones_boundary;
            for (std::size_t row = 0; row < _diagonal.size(); ++row) {
                values[to_sizet(_diagonal[row])] = _row_sums[to_idx(row)];
            }
            // Newton linearisation of sigma T^4: column j carries
            // 4 sigma Tj^3.
            const Eigen::Map<const Eigen::VectorXi> columns(
                _matrix.innerIndexPtr(), _matrix.nonZeros());
            _matrix.coeffs() *= t3(columns).array();
            radiation.noalias() = _matrix * t;
            radiation *= 0.75;
        } else {
            radiation.setZero();
        }

        const auto conductive = values_of(*_couplings.conductive);
        for (std::size_t k = 0; k < conductive.size(); ++k) {
            values[to_sizet(_upper_kl[k])] -= conductive[k];
        }
        for (std::size_t k = 0; k < _lower_kl.size(); ++k) {
            values[to_sizet(_lower_kl[k])] -= conductive[k];
        }
        _row_sums.noalias() =
            _couplings.conductive->selfadjointView<Eigen::Upper>() *
            _ones_domain;
        _row_sums.noalias() += *_couplings.conductive_boundary * _ones_boundary;
        add_to_diagonal(_row_sums);
    }

    void add_to_diagonal(const Eigen::VectorXd& values) override {
        const std::span<double> stored(_matrix.valuePtr(),
                                       to_sizet(_matrix.nonZeros()));
        for (std::size_t row = 0; row < _diagonal.size(); ++row) {
            stored[to_sizet(_diagonal[row])] += values[to_idx(row)];
        }
    }

    void multiply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override {
        if (_options.symmetric) {
            y.noalias() = _matrix.selfadjointView<Eigen::Upper>() * x;
        } else {
            y.noalias() = _matrix * x;
        }
    }

    void multiply(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) const override {
        if (_options.symmetric) {
            y.noalias() = _matrix.selfadjointView<Eigen::Upper>() * x;
        } else {
            y.noalias() = _matrix * x;
        }
    }

    SolveOutcome solve(Eigen::VectorXd& b, Eigen::VectorXd& x) override {
        auto& factorizer = *_factorizer;
        SolveOutcome outcome{.succeeded = false, .factorized = true};
        if (_options.skip_unchanged && _has_factors &&
            std::ranges::equal(values_of(_matrix), _factorized_values)) {
            factorizer.solve(_matrix, b, x);
            outcome.factorized = false;
        } else if (_has_factors && factorizer.reuses_factors()) {
            factorizer.factorize_and_solve(_matrix, b, x);
            outcome.factorized = factorizer.factors_recomputed();
        } else {
            factorizer.factorize(_matrix);
            if (factorizer.succeeded()) {
                factorizer.solve(_matrix, b, x);
            }
        }
        outcome.succeeded = factorizer.succeeded();
        if (!outcome.succeeded) {
            _has_factors = false;
        } else if (outcome.factorized) {
            _has_factors = true;
            if (_options.skip_unchanged) {
                const auto current = values_of(_matrix);
                _factorized_values.assign(current.begin(), current.end());
            }
        }
        return outcome;
    }

    bool solve(Eigen::MatrixXd& b, Eigen::MatrixXd& x) override {
        _factorizer->solve(_matrix, b, x);
        return _factorizer->succeeded();
    }

    void discard_factors() noexcept override { _has_factors = false; }

    void apply_live_settings(const ResolvedSettings& settings) override {
        _factorizer->apply_live_settings(settings);
    }

    [[nodiscard]] bool zero_pivot() const noexcept override {
        return _factorizer->zero_pivot();
    }
    [[nodiscard]] std::string error_message() const override {
        return _factorizer->error_message();
    }
    [[nodiscard]] Index perturbed_pivots() const noexcept override {
        return _factorizer->perturbed_pivots();
    }
    [[nodiscard]] std::optional<std::string> hint(
        const ResolvedSettings& settings) const override {
        return min_degree_hint(*_factorizer, settings, _options.symmetric,
                               _matrix.rows());
    }

    [[nodiscard]] std::int64_t factor_bytes() const noexcept override {
        return _factorizer->factor_bytes();
    }
    [[nodiscard]] bool factor_bytes_exact() const noexcept override {
        return _factorizer->factor_bytes_exact();
    }
    [[nodiscard]] std::int64_t matrix_bytes() const noexcept override {
        return _factorizer->matrix_copy_bytes() + bytes_of(_matrix) +
               bytes_of(_upper_kl) + bytes_of(_lower_kl) + bytes_of(_upper_kr) +
               bytes_of(_lower_kr) + bytes_of(_diagonal) +
               bytes_of(_factorized_values) + bytes_of(_row_sums) +
               bytes_of(_ones_domain) + bytes_of(_ones_boundary);
    }

  private:
    void build_pattern() {
        const SparseMatrix& conductive = *_couplings.conductive;
        if (_options.symmetric) {
            // Upper triangle with the diagonal: the conductive couplings are
            // already stored that way.
            _matrix = conductive;
            sparse_utils::add_zero_diag_square(_matrix);
            _upper_kl = map_values(conductive, _matrix, /*transposed=*/false);
        } else {
            _matrix = conductive.selfadjointView<Eigen::Upper>();
            if (_options.radiation) {
                _matrix +=
                    _couplings.radiative->selfadjointView<Eigen::Upper>();
            }
            sparse_utils::add_zero_diag_square(_matrix);
            _upper_kl = map_values(conductive, _matrix, /*transposed=*/false);
            _lower_kl = map_values(conductive, _matrix, /*transposed=*/true);
            if (_options.radiation) {
                _upper_kr = map_values(*_couplings.radiative, _matrix,
                                       /*transposed=*/false);
                _lower_kr = map_values(*_couplings.radiative, _matrix,
                                       /*transposed=*/true);
            }
        }
        _diagonal.resize(to_sizet(_matrix.rows()));
        for (Index row = 0; row < _matrix.rows(); ++row) {
            _diagonal[to_sizet(row)] = value_index(_matrix, row, row);
        }
        sparse_utils::set_to_zero(_matrix);
    }

    NetworkCouplings _couplings;
    NetworkOptions _options;
    std::unique_ptr<Factorizer> _factorizer;
    SparseMatrix _matrix;
    std::vector<int> _upper_kl;
    std::vector<int> _lower_kl;
    std::vector<int> _upper_kr;
    std::vector<int> _lower_kr;
    std::vector<int> _diagonal;
    std::vector<double> _factorized_values;
    Eigen::VectorXd _row_sums;
    Eigen::VectorXd _ones_domain;
    Eigen::VectorXd _ones_boundary;
    bool _has_factors = false;
};

}  // namespace

std::unique_ptr<LinearizedNetwork> make_sparse_network(
    const ResolvedSettings& settings, const NetworkCouplings& couplings,
    const NetworkOptions& options) {
    return std::make_unique<SparseNetwork>(settings, couplings, options);
}

std::unique_ptr<LinearizedNetwork> make_network(
    const ResolvedSettings& settings, const NetworkCouplings& couplings,
    const NetworkOptions& options) {
    if (settings.storage == Storage::DENSE) {
        return make_dense_network(settings, couplings, options);
    }
    return make_sparse_network(settings, couplings, options);
}

}  // namespace pycanha::solvers::detail
