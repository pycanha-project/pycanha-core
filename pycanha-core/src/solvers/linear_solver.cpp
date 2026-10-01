#include "pycanha-core/solvers/linear_solver.hpp"

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

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/pardiso.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {

namespace {

using SparseMatrix = SparseLinearSolver::SparseMatrix;
using ColumnMajorMatrix = Eigen::SparseMatrix<double>;

[[nodiscard]] bool is_pardiso_type(DirectSolverType type) noexcept {
    return type == DirectSolverType::TWO_LEVEL ||
           type == DirectSolverType::ONE_LEVEL ||
           type == DirectSolverType::MIN_DEGREE;
}

[[nodiscard]] bool iterates(int iparm_3) noexcept {
    constexpr int decimal = 10;
    return iparm_3 % decimal != 0;
}

// Eigen's sparse solvers read column-major matrices. The solver keeps a
// column-major copy of the pattern and, for every one of its values, the
// position of the same entry in the row-major matrix, so each factorisation
// only gathers the values.
template <typename Factorization>
class EigenSolver final : public SparseLinearSolver {
  public:
    explicit EigenSolver(std::string_view name) : _name(name) {}

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
    Factorization _factorization;
    ColumnMajorMatrix _column_major;
    std::vector<std::size_t> _source;
    Eigen::ComputationInfo _info = Eigen::Success;
};

#if PYCANHA_USE_MKL
std::unique_ptr<SparseLinearSolver> make_pardiso(
    const LinearSolverOptions& options, DirectSolverType type, int iparm_3) {
    constexpr std::size_t ordering = 1;
    constexpr std::size_t two_level = 23;
    constexpr int minimum_degree = 0;
    auto solver = std::make_unique<PardisoSolver>(
        options.symmetric_positive_definite
            ? PardisoSolver::MatrixType::SYMMETRIC_POSITIVE_DEFINITE
            : PardisoSolver::MatrixType::STRUCTURALLY_SYMMETRIC,
        iparm_3);
    if (type == DirectSolverType::ONE_LEVEL) {
        solver->set_iparm(two_level, 0);
    } else if (type == DirectSolverType::MIN_DEGREE) {
        solver->set_iparm(ordering, minimum_degree);
        solver->set_iparm(two_level, 0);
    }
    for (const auto& [index, value] : options.iparm_overrides) {
        if (index == 3) {
            continue;  // passed to the constructor
        }
        if (index < 0 ||
            std::cmp_greater_equal(index, PardisoSolver::iparm_size)) {
            throw std::invalid_argument(
                std::format("PARDISO iparm index {} is out of range (0 to {})",
                            index, PardisoSolver::iparm_size - 1));
        }
        solver->set_iparm(to_sizet(index), value);
    }
    solver->set_threads(options.threads);
    solver->set_verbose(options.verbose);
    return solver;
}
#endif

}  // namespace

SolverEngine default_solver_engine() noexcept {
    return MKL_ENABLED ? SolverEngine::MKL : SolverEngine::EIGEN;
}

std::string_view to_string(SolverEngine engine) noexcept {
    return engine == SolverEngine::MKL ? "MKL" : "EIGEN";
}

std::string_view to_string(DirectSolverType type) noexcept {
    switch (type) {
        case DirectSolverType::DEFAULT:
            return "DEFAULT";
        case DirectSolverType::TWO_LEVEL:
            return "TWO_LEVEL";
        case DirectSolverType::ONE_LEVEL:
            return "ONE_LEVEL";
        case DirectSolverType::MIN_DEGREE:
            return "MIN_DEGREE";
        case DirectSolverType::COLAMD:
            return "COLAMD";
        case DirectSolverType::AMD:
            return "AMD";
        case DirectSolverType::LDLT:
            return "LDLT";
    }
    return "UNKNOWN";
}

DirectSolverType resolve_solver_type(SolverEngine engine,
                                     DirectSolverType type) noexcept {
    if (type != DirectSolverType::DEFAULT) {
        return type;
    }
    return engine == SolverEngine::MKL ? DirectSolverType::TWO_LEVEL
                                       : DirectSolverType::COLAMD;
}

void SparseLinearSolver::factorize_and_solve(const SparseMatrix& matrix,
                                             Eigen::VectorXd& b,
                                             Eigen::VectorXd& x) {
    factorize(matrix);
    if (succeeded()) {
        solve(matrix, b, x);
    }
}

std::unique_ptr<SparseLinearSolver> make_linear_solver(
    const LinearSolverOptions& options) {
    const auto type = resolve_solver_type(options.engine, options.type);
    const auto found = options.iparm_overrides.find(3);
    const int iparm_3 = found == options.iparm_overrides.end() ? options.iparm_3
                                                               : found->second;

    if (options.engine == SolverEngine::MKL) {
        if (!MKL_ENABLED) {
            throw std::invalid_argument(
                "The MKL engine is not available: pycanha-core was built "
                "without MKL. Use the EIGEN engine.");
        }
        if (!is_pardiso_type(type)) {
            throw std::invalid_argument(std::format(
                "Solver type {} belongs to the EIGEN engine. With the MKL "
                "engine use TWO_LEVEL, ONE_LEVEL or MIN_DEGREE.",
                to_string(type)));
        }
        if (iterates(iparm_3) && type == DirectSolverType::TWO_LEVEL) {
            throw std::invalid_argument(std::format(
                "The iterative step (iparm[3] = {}) needs PARDISO's one-level "
                "factorisation: use ONE_LEVEL or MIN_DEGREE, not TWO_LEVEL.",
                iparm_3));
        }
#if PYCANHA_USE_MKL
        return make_pardiso(options, type, iparm_3);
#endif
    }

    if (is_pardiso_type(type)) {
        throw std::invalid_argument(std::format(
            "Solver type {} belongs to the MKL engine. With the EIGEN engine "
            "use COLAMD, AMD or LDLT.",
            to_string(type)));
    }
    if (iterates(iparm_3)) {
        throw std::invalid_argument(
            "The iterative step (pardiso_iparm_3) is only available with the "
            "MKL engine.");
    }
    if ((type == DirectSolverType::LDLT) !=
        options.symmetric_positive_definite) {
        throw std::invalid_argument(
            "LDLT needs a symmetric matrix, and only LDLT takes one: it is "
            "available to SSLU for models without radiative couplings.");
    }
    switch (type) {
        case DirectSolverType::AMD:
            return std::make_unique<EigenSolver<
                Eigen::SparseLU<ColumnMajorMatrix, Eigen::AMDOrdering<int>>>>(
                "SparseLU (AMD)");
        case DirectSolverType::LDLT:
            return std::make_unique<EigenSolver<Eigen::SimplicialLDLT<
                ColumnMajorMatrix, Eigen::Upper, Eigen::AMDOrdering<int>>>>(
                "SimplicialLDLT");
        default:
            return std::make_unique<EigenSolver<Eigen::SparseLU<
                ColumnMajorMatrix, Eigen::COLAMDOrdering<int>>>>(
                "SparseLU (COLAMD)");
    }
}

void suggest_min_degree_if_dense(const SparseLinearSolver& solver,
                                 const LinearSolverOptions& options,
                                 Eigen::Index size,
                                 std::string_view solver_name) {
    const auto type = resolve_solver_type(options.engine, options.type);
    if (options.engine != SolverEngine::MKL ||
        type == DirectSolverType::MIN_DEGREE || size <= 0) {
        return;
    }
    // A factor of more than 5e7 non-zeros filling more than a quarter of the
    // dense matrix: many radiative couplings per node, where minimum degree
    // ordering seems to factorise much faster than nested dissection.
    constexpr double large_factor = 5.0e7;
    constexpr double near_dense = 0.25;
    const auto factor = static_cast<double>(solver.factor_nonzeros());
    const auto n = static_cast<double>(size);
    const double dense =
        options.symmetric_positive_definite ? n * (n + 1.0) / 2.0 : n * n;
    const double fraction = factor / dense;
    if (factor > large_factor && fraction > near_dense) {
        SPDLOG_LOGGER_WARN(
            get_logger(),
            "{}: the factor will have {:.3g} non-zeros, {:.0f} % of a dense "
            "matrix. For such nearly dense models (many radiative couplings "
            "per node) solver_type MIN_DEGREE can factorise several times "
            "faster.",
            solver_name, factor, 100.0 * fraction);
    }
}

}  // namespace pycanha
