#include "factorizer.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <cstddef>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/utils/logger.hpp"

#if PYCANHA_USE_MKL
#include "pardiso.hpp"
#endif

namespace pycanha::solvers::detail {

namespace {

using SparseMatrix = Factorizer::SparseMatrix;
using ColumnMajorMatrix = Eigen::SparseMatrix<double>;

// Eigen's sparse solvers read column-major matrices. The factorizer keeps a
// column-major copy of the pattern and, for every one of its values, the
// position of the same entry in the row-major matrix, so each factorization
// only gathers the values.
template <typename EigenFactorization>
class EigenFactorizer final : public Factorizer {
  public:
    explicit EigenFactorizer(std::string_view name) : _name(name) {}

    void analyze_pattern(const SparseMatrix& matrix) override {
        // Number the row-major values, convert, and read the numbers back.
        SparseMatrix numbered = matrix;
        const std::span<double> numbers(numbered.valuePtr(),
                                        to_sizet(numbered.nonZeros()));
        for (std::size_t k = 0; k < numbers.size(); ++k) {
            numbers[k] = static_cast<double>(k);
        }
        _column_major = numbered;
        const std::span<const double> sources(
            _column_major.valuePtr(), to_sizet(_column_major.nonZeros()));
        _source.resize(sources.size());
        for (std::size_t k = 0; k < sources.size(); ++k) {
            _source[k] = static_cast<std::size_t>(sources[k]);
        }
        _factorization.analyzePattern(_column_major);
        _info = Eigen::Success;
    }

    void factorize(const SparseMatrix& matrix) override {
        const std::span<const double> values(matrix.valuePtr(),
                                             to_sizet(matrix.nonZeros()));
        const std::span<double> gathered(_column_major.valuePtr(),
                                         to_sizet(_column_major.nonZeros()));
        for (std::size_t k = 0; k < gathered.size(); ++k) {
            gathered[k] = values[_source[k]];
        }
        _factorization.factorize(_column_major);
        _info = _factorization.info();
    }

    void solve(const SparseMatrix& /*matrix*/, Eigen::VectorXd& b,
               Eigen::VectorXd& x) override {
        x = _factorization.solve(b);
        _info = _factorization.info();
    }

    void solve(const SparseMatrix& /*matrix*/, Eigen::MatrixXd& b,
               Eigen::MatrixXd& x) override {
        x = _factorization.solve(b);
        _info = _factorization.info();
    }

    [[nodiscard]] bool succeeded() const noexcept override {
        return _info == Eigen::Success;
    }
    [[nodiscard]] bool zero_pivot() const noexcept override {
        return _info == Eigen::NumericalIssue;
    }
    [[nodiscard]] std::string error_message() const override {
        return std::format("Eigen {} failed (ComputationInfo {})", _name,
                           static_cast<int>(_info));
    }

  private:
    std::string_view _name;
    EigenFactorization _factorization;
    ColumnMajorMatrix _column_major;
    std::vector<std::size_t> _source;
    Eigen::ComputationInfo _info = Eigen::Success;
};

using EigenLuColamd = EigenFactorizer<
    Eigen::SparseLU<ColumnMajorMatrix, Eigen::COLAMDOrdering<int>>>;
using EigenLuAmd = EigenFactorizer<
    Eigen::SparseLU<ColumnMajorMatrix, Eigen::AMDOrdering<int>>>;
using EigenLdlt =
    EigenFactorizer<Eigen::SimplicialLDLT<ColumnMajorMatrix, Eigen::Upper,
                                          Eigen::AMDOrdering<int>>>;

std::unique_ptr<Factorizer> make_eigen(const ResolvedSettings& settings,
                                       bool symmetric) {
    if (symmetric) {
        return std::make_unique<EigenLdlt>("SimplicialLDLT");
    }
    if (settings.eigen.ordering == EigenOrdering::AMD) {
        return std::make_unique<EigenLuAmd>("SparseLU (AMD)");
    }
    return std::make_unique<EigenLuColamd>("SparseLU (COLAMD)");
}

#if PYCANHA_USE_MKL
std::unique_ptr<Factorizer> make_pardiso(const ResolvedSettings& settings,
                                         bool symmetric) {
    // iparm[3] = 61: iterate on the previous factors (CGS, or CG for
    // Cholesky) down to a relative residual of 1e-6 before refactorizing.
    constexpr int reuse_iterative_step = 61;
    constexpr int iterative_step = 3;
    constexpr std::size_t ordering = 1;
    constexpr std::size_t two_level = 23;
    constexpr int minimum_degree = 0;

    const auto& iparm = settings.pardiso.iparm;
    const auto found = iparm.find(iterative_step);
    int iparm_3 = settings.reuse_factors ? reuse_iterative_step : 0;
    if (found != iparm.end()) {
        iparm_3 = found->second;
    }
    auto pardiso = std::make_unique<PardisoSolver>(
        symmetric ? PardisoSolver::MatrixType::SYMMETRIC_POSITIVE_DEFINITE
                  : PardisoSolver::MatrixType::STRUCTURALLY_SYMMETRIC,
        iparm_3);
    if (settings.pardiso.level == PardisoLevel::ONE_LEVEL) {
        pardiso->set_iparm(two_level, 0);
    }
    if (settings.pardiso.ordering == PardisoOrdering::MIN_DEGREE) {
        pardiso->set_iparm(ordering, minimum_degree);
    }
    for (const auto& [index, value] : iparm) {
        if (index == iterative_step) {
            continue;  // passed to the constructor
        }
        if (index < 0 ||
            std::cmp_greater_equal(index, PardisoSolver::iparm_size)) {
            throw std::invalid_argument(
                std::format("PARDISO iparm index {} is out of range (0 to {})",
                            index, PardisoSolver::iparm_size - 1));
        }
        pardiso->set_iparm(to_sizet(index), value);
    }
    pardiso->apply_live_settings(settings);
    return pardiso;
}
#endif

}  // namespace

void Factorizer::factorize_and_solve(const SparseMatrix& matrix,
                                     Eigen::VectorXd& b, Eigen::VectorXd& x) {
    factorize(matrix);
    if (succeeded()) {
        solve(matrix, b, x);
    }
}

std::unique_ptr<Factorizer> make_factorizer(const ResolvedSettings& settings,
                                            bool symmetric) {
    if (settings.backend == Backend::EIGEN) {
        return make_eigen(settings, symmetric);
    }
#if PYCANHA_USE_MKL
    return make_pardiso(settings, symmetric);
#else
    throw std::invalid_argument(
        "backend = PARDISO needs a build with MKL; use backend = EIGEN");
#endif
}

std::string describe(const ResolvedSettings& settings, bool symmetric) {
    const std::string_view form = symmetric ? "Cholesky" : "LU";
    if (settings.backend == Backend::EIGEN) {
        return std::format(
            "EIGEN {}, {}",
            symmetric ? "AMD" : to_string(settings.eigen.ordering), form);
    }
    return std::format("PARDISO {} {}{}, {}", to_string(settings.pardiso.level),
                       to_string(settings.pardiso.ordering),
                       settings.reuse_factors ? ", reusing factors" : "", form);
}

void suggest_min_degree_if_dense(const Factorizer& factorizer,
                                 const ResolvedSettings& settings,
                                 bool symmetric, Eigen::Index size,
                                 std::string_view solver_name) {
    if (settings.backend != Backend::PARDISO ||
        settings.pardiso.ordering == PardisoOrdering::MIN_DEGREE || size <= 0) {
        return;
    }
    // A factor of more than 5e7 non-zeros filling more than a quarter of the
    // dense matrix: many radiative couplings per node, where the minimum
    // degree ordering seems to factorize much faster than nested dissection.
    constexpr double large_factor = 5.0e7;
    constexpr double near_dense = 0.25;
    const auto factor = static_cast<double>(factorizer.factor_nonzeros());
    const auto n = static_cast<double>(size);
    const double dense = symmetric ? n * (n + 1.0) / 2.0 : n * n;
    const double fraction = factor / dense;
    if (factor > large_factor && fraction > near_dense) {
        SPDLOG_LOGGER_WARN(
            get_logger(),
            "{}: the factor will have {:.3g} non-zeros, {:.0f} % of a dense "
            "matrix. For such nearly dense models (many radiative couplings "
            "per node) pardiso.ordering = MIN_DEGREE can factorize several "
            "times faster.",
            solver_name, factor, 100.0 * fraction);
    }
}

}  // namespace pycanha::solvers::detail
