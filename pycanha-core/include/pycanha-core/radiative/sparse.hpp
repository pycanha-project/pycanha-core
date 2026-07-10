#pragma once

#include <Eigen/Dense>
#include <cstdint>

namespace pycanha::radiative {

// Minimal CSR container used by every matrix result; maps 1:1 (zero-copy
// through the bindings) to scipy.sparse.csr_matrix. Rows/cols are face slots
// (global, both sides); "to space" is implicit as the row deficit.
struct SparseF64 {
    Eigen::VectorX<std::int64_t> indptr;   // rows + 1
    Eigen::VectorX<std::int32_t> indices;  // column of each stored value
    Eigen::VectorXd values;
    std::int64_t rows = 0;
    std::int64_t cols = 0;

    [[nodiscard]] std::int64_t nnz() const noexcept { return values.rows(); }
};

}  // namespace pycanha::radiative
