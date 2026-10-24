#pragma once

#include <memory>

#include "pycanha-core/solvers/linear_solver.hpp"

namespace pycanha {

/// Whether the ACCELERATE engine can be used: the build has Apple Accelerate
/// and the running macOS has its sparse LU (macOS 15.5 or newer).
[[nodiscard]] bool accelerate_available() noexcept;

#if PYCANHA_USE_ACCELERATE
/// Sparse direct solver of the Apple Accelerate framework.
/**
 * Multi-threaded supernodal factorisation: LU (threshold partial pivoting)
 * for a matrix holding both triangles of a structurally symmetric pattern,
 * Cholesky when symmetric_positive_definite (upper triangle only). type is
 * the fill-reducing ordering: METIS (nested dissection) or AMD.
 *
 * The matrix is passed without copying its values: the row-major (CSR)
 * arrays are read as the transpose of a CSC matrix for LU, and as the lower
 * triangle of a CSC matrix for Cholesky. The index arrays are copied at
 * analyze_pattern(), and refactorisations reuse the symbolic analysis and
 * the factor storage. Requires accelerate_available().
 */
[[nodiscard]] std::unique_ptr<SparseLinearSolver> make_accelerate_solver(
    DirectSolverType type, bool symmetric_positive_definite);
#endif

}  // namespace pycanha
