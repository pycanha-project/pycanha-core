#pragma once

// src-private CSR assembly: turns the row-major (row_starts, columns,
// values) arrays every radiative producer builds anyway into a compressed
// Eigen sparse matrix with one copy per array.
//
// Deliberately NOT setFromTriplets: that sorts O(nnz log nnz) and allocates
// per entry, which would dominate an assembly pass whose whole point is to
// stream. Every producer here already emits ascending column order within a
// row, so the arrays are a valid CSR image as they stand.

#include <Eigen/Sparse>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

#include "pycanha-core/radiative/results.hpp"

namespace pycanha::radiative::detail {

// Largest entry count the CSR index type can address. Exceeding it would
// silently wrap the row pointers, so it is checked rather than assumed.
[[nodiscard]] inline std::size_t max_sparse_entries() noexcept {
    return static_cast<std::size_t>(std::numeric_limits<SparseIndex>::max());
}

inline void check_sparse_capacity(std::size_t entries) {
    if (entries > max_sparse_entries()) {
        throw std::length_error(
            "pycanha::radiative: the result has " + std::to_string(entries) +
            " stored entries, more than the sparse index type can address; "
            "raise sparse_threshold or aggregate to nodes first");
    }
}

// `row_starts` holds rows + 1 ascending offsets into `columns`/`values`;
// columns must be ascending within each row. The arrays are copied, so the
// caller's buffers can be reused or freed straight away.
[[nodiscard]] inline SparseMatrix make_csr(
    Eigen::Index rows, Eigen::Index cols,
    std::span<const SparseIndex> row_starts,
    std::span<const SparseIndex> columns, std::span<const double> values) {
    SparseMatrix matrix(rows, cols);
    if (values.empty()) {
        return matrix;  // default-constructed is compressed and empty
    }
    return Eigen::Map<const SparseMatrix>(
        rows, cols, static_cast<Eigen::Index>(values.size()), row_starts.data(),
        columns.data(), values.data());
}

}  // namespace pycanha::radiative::detail
