
#include "pycanha-core/utils/SparseUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/utils/Instrumentor.hpp"
#include "pycanha-core/utils/RandomGenerators.hpp"

// USE MKL FUNCTION IF AVAILABLE
#if PYCANHA_USE_MKL
#include <mkl_cblas.h>
#include <mkl_types.h>

#include <limits>
#endif

// This file use a lot of pointer arithmetic for performance, the warning from
// clang-tidy is suppressed
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)

using namespace pycanha;  // NOLINT(build/namespaces)

namespace pycanha::sparse_utils {

// Internal functions
// TODO: After fix in add_zero_col_row_fun, refactor the internal function and
// put the code inside the public functions
namespace {

void add_zero_row_fun(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                      Index new_row_idx) {
    if (sparse.isCompressed()) {
        // Copy the outer indices to the new positions
        auto* src_ptr_outer = sparse.outerIndexPtr() + new_row_idx;
        memmove(
            src_ptr_outer + 1, src_ptr_outer,
            (sparse.outerSize() - new_row_idx - 1) * sizeof(*src_ptr_outer));
    } else {
        // Copy the outer and nnz indices to the new positions
        auto* src_ptr_outer = sparse.outerIndexPtr() + new_row_idx;
        auto* src_ptr_nnz = sparse.innerNonZeroPtr() + new_row_idx;
        const size_t elems_to_copy = (sparse.outerSize() - new_row_idx - 1);
        memmove(src_ptr_outer + 1, src_ptr_outer,
                elems_to_copy * sizeof(*src_ptr_outer));
        memmove(src_ptr_nnz + 1, src_ptr_nnz,
                elems_to_copy * sizeof(*src_ptr_nnz));

        sparse.innerNonZeroPtr()[new_row_idx] = 0;
    }
}

void add_zero_col_fun(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                      Index new_col_idx) {
    if (sparse.isCompressed()) {
        // The column index (innerIndexPtr) of any coefficient equal or bigger
        // than new_index should be increased in one.
        for (Index col = 0; col < sparse.outerIndexPtr()[sparse.outerSize()];
             col++) {
            if (sparse.innerIndexPtr()[col] >= new_col_idx) {
                sparse.innerIndexPtr()[col]++;
            }
        }
    } else {
        // The column index (innerIndexPtr) of any coefficient equal or bigger
        // than new_index should be increased in one.
        for (Index row = 0; row < sparse.outerSize(); row++) {
            const Index ival_start = sparse.outerIndexPtr()[row];
            const Index nnz = sparse.innerNonZeroPtr()[row];
            for (Index ival = ival_start; ival < ival_start + nnz; ival++) {
                if (sparse.innerIndexPtr()[ival] >= new_col_idx) {
                    sparse.innerIndexPtr()[ival]++;
                }
            }
        }
    }
}

// TODO: FIX. This function should be faster than calling first row and then col
// TOOD: After fix, refactor internal functions ..._row  ..._col ...row_col
void add_zero_row_col_fun(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                          Index new_row_idx, Index new_col_idx) {
    // Change the size of the matrix without deleting the values
    sparse.conservativeResize(sparse.rows() + 1, sparse.cols() + 1);

    if (sparse.isCompressed()) {
        // The column index (innerIndexPtr) of any coefficient equal or bigger
        // than new_index should be increased in one.
        for (Index col = 0; col < sparse.outerIndexPtr()[sparse.outerSize()];
             col++) {
            if (sparse.innerIndexPtr()[col] >= new_col_idx) {
                sparse.innerIndexPtr()[col]++;
            }
        }

        // Copy the outer indices to the new positions
        auto* src_ptr_outer = sparse.outerIndexPtr() + new_row_idx;
        memmove(
            src_ptr_outer + 1, src_ptr_outer,
            (sparse.outerSize() - new_row_idx - 1) * sizeof(*src_ptr_outer));
    } else {
        // The column index (innerIndexPtr) of any coefficient equal or bigger
        // than new_index should be increased in one.
        for (Index row = 0; row < sparse.outerSize(); row++) {
            const Index ival_start = sparse.outerIndexPtr()[row];
            const Index nnz = sparse.innerNonZeroPtr()[row];
            for (Index ival = ival_start; ival < ival_start + nnz; ival++) {
                if (sparse.innerIndexPtr()[ival] >= new_col_idx) {
                    sparse.innerIndexPtr()[ival]++;
                }
            }
        }

        // Copy the outer and nnz indices to the new positions
        auto* src_ptr_outer = sparse.outerIndexPtr() + new_row_idx;
        auto* src_ptr_nnz = sparse.innerNonZeroPtr() + new_row_idx;
        const size_t elems_to_copy = (sparse.outerSize() - new_row_idx - 1);
        memmove(src_ptr_outer + 1, src_ptr_outer,
                elems_to_copy * sizeof(*src_ptr_outer));
        memmove(src_ptr_nnz + 1, src_ptr_nnz,
                elems_to_copy * sizeof(*src_ptr_nnz));

        sparse.innerNonZeroPtr()[new_row_idx] = 0;
    }
}

void move_delete_row_fun(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                         Index move_row_idx) {
    PROFILE_FUNCTION();
    // Removing a node will left a "hole", so the matrix should be in uncompress
    // mode
    if (sparse.isCompressed()) {
        sparse.uncompress();
    }

    // Remove row
    // The outer and nnz vectors are displaced, overwriting the information in
    // the selected row.
    memmove(
        sparse.outerIndexPtr() + move_row_idx,
        sparse.outerIndexPtr() + move_row_idx + 1,
        (sparse.rows() - move_row_idx - 1) * sizeof(*sparse.outerIndexPtr()));

    memmove(
        sparse.innerNonZeroPtr() + move_row_idx,
        sparse.innerNonZeroPtr() + move_row_idx + 1,
        (sparse.rows() - move_row_idx - 1) * sizeof(*sparse.innerNonZeroPtr()));

    // There can`t be any holes before the first row. If removing the first row,
    // the coefficient of the second row should be the first
    if (move_row_idx == 0) {
        if (sparse.innerNonZeroPtr()[0] != 0) {
            // The second row has coefficient. They should be copied to the
            // begining of the value vector (values and inner vectors)
            memmove(sparse.valuePtr(),
                    sparse.valuePtr() + sparse.outerIndexPtr()[0],
                    (sparse.innerNonZeroPtr()[0]) * sizeof(*sparse.valuePtr()));

            memmove(sparse.innerIndexPtr(),
                    sparse.innerIndexPtr() + sparse.outerIndexPtr()[0],
                    (sparse.innerNonZeroPtr()[0]) *
                        sizeof(*sparse.innerIndexPtr()));
        }
        // Outer index should start always at 0
        sparse.outerIndexPtr()[0] = 0;
    }
}

void move_delete_row_col_fun(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index move_col_idx) {
    PROFILE_FUNCTION();
    // Removing a node will left a "hole", so the matrix should be in uncompress
    // mode
    if (sparse.isCompressed()) {
        sparse.uncompress();
    }

    // Remove column

    // All inner indices should be reduced in 1
    for (Index iouter = 0; iouter < sparse.outerSize(); iouter++) {
        const Index ival_start =
            sparse.outerIndexPtr()[iouter];  // Index of the first value of the
                                             // row
        auto nnz = sparse.innerNonZeroPtr()[iouter];

        // Loop through the inner vector
        for (Index ival = ival_start; ival < ival_start + nnz; ival++) {
            if (sparse.innerIndexPtr()[ival] > move_col_idx) {
                sparse.innerIndexPtr()[ival]--;
            } else if (sparse.innerIndexPtr()[ival] == move_col_idx) {
                if (ival == ival_start) {
                    // Check if first non zero value in the matrix
                    if (ival > 0) {
                        sparse.outerIndexPtr()[iouter]++;
                    }

                } else if ((ival < ival_start + nnz - 1)) {
                    // The value is in the middle
                    auto elems_to_move = (ival - ival_start - 1 + nnz);
                    memmove(sparse.valuePtr() + ival,
                            sparse.valuePtr() + ival + 1,
                            (elems_to_move) * sizeof(*sparse.valuePtr()));
                    memmove(sparse.innerIndexPtr() + ival,
                            sparse.innerIndexPtr() + ival + 1,
                            (elems_to_move) * sizeof(*sparse.innerIndexPtr()));
                    sparse.innerIndexPtr()[ival]--;
                }
                sparse.innerNonZeroPtr()[iouter]--;
            }
        }
    }
}

template <class T>
[[nodiscard]] inline bool equal_buffers(const T* a, const T* b,
                                        std::size_t n) noexcept {
    // std::equal is clear and typically vectorizes / becomes memcmp where
    // possible.
    return std::equal(a, a + n, b);
}

[[nodiscard]] bool compare_compressed(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& a,
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& b) noexcept {
    const auto nnz = static_cast<std::size_t>(a.nonZeros());
    const auto outer_len = static_cast<std::size_t>(a.outerSize() + 1);

    return equal_buffers(a.innerIndexPtr(), b.innerIndexPtr(), nnz) &&
           equal_buffers(a.outerIndexPtr(), b.outerIndexPtr(), outer_len);
}

[[nodiscard]] bool compare_uncompressed(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& a,
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& b) noexcept {
    const auto outer = static_cast<std::size_t>(a.outerSize());
    const auto outer_len = outer + 1;

    // Same number of actually-used entries per outer index (row in RowMajor).
    if (!equal_buffers(a.innerNonZeroPtr(), b.innerNonZeroPtr(), outer)) {
        return false;
    }

    // Same reserved block boundaries for each outer index.
    if (!equal_buffers(a.outerIndexPtr(), b.outerIndexPtr(), outer_len)) {
        return false;
    }

    // Compare only the *used* inner indices for each outer block; ignore
    // reserved tail.
    const auto* ia = a.innerIndexPtr();
    const auto* ib = b.innerIndexPtr();
    const auto* start = a.outerIndexPtr();
    const auto* used = a.innerNonZeroPtr();

    for (std::size_t i = 0; i < outer; ++i) {
        const auto beg = static_cast<std::size_t>(start[i]);
        const auto cnt = static_cast<std::size_t>(used[i]);
        if (!equal_buffers(ia + beg, ib + beg, cnt)) {
            return false;
        }
    }
    return true;
}

}  // namespace

void add_zero_row(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                  Index new_row_idx) {
    // Change the size of the matrix without deleting the values
    sparse.conservativeResize(sparse.rows() + 1, sparse.cols());
    add_zero_row_fun(sparse, new_row_idx);
}

void add_zero_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                  Index new_col_idx) {
    // Change the size of the matrix without deleting the values
    sparse.conservativeResize(sparse.rows(), sparse.cols() + 1);
    add_zero_col_fun(sparse, new_col_idx);
}

void add_zero_row_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                      Index new_row_idx, Index new_col_idx) {
    // Change the size of the matrix without deleting the values
    sparse.conservativeResize(sparse.rows() + 1, sparse.cols() + 1);
    add_zero_row_fun(sparse, new_row_idx);
    add_zero_col_fun(sparse, new_col_idx);
}

void add_zero_diag_square(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse) {
    // Add a zero diagonal entry. The values that were already in the matrix
    // remain the same. The rest values of the diagonal are converted to
    // explicit zeros.
    if (sparse.cols() != sparse.rows()) {
        throw std::runtime_error(
            "This operation only support squared matrices.");
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor> zero_diag;
    zero_diag.conservativeResize(sparse.rows(), sparse.rows());
    zero_diag.setIdentity();
    zero_diag *= 0.0;
    sparse += zero_diag;
    sparse.makeCompressed();  // Just in case
}

// move_rows with RAII scratch buffers (C++20/23)
void move_rows(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
               Index from_idx, Index to_idx) {
    PROFILE_FUNCTION();

    using StorageIndexPtr = decltype(sparse.innerIndexPtr());
    using ValuesPtr = decltype(sparse.valuePtr());
    using StorageIndex = std::remove_pointer_t<StorageIndexPtr>;
    using Values = std::remove_pointer_t<ValuesPtr>;

    // Ensure from_idx > to_idx
    if (from_idx >= to_idx) {
        if (from_idx == to_idx) {
            return;
        }
        std::swap(from_idx, to_idx);
    }

    if ((from_idx < 0) || (to_idx >= sparse.rows())) {
        if (VERBOSE) {
            std::cout << "Error while moving rows. Invalid indexes.\n";
        }
        return;
    }

    // Work in non-compressed mode
    if (sparse.isCompressed()) {
        sparse.uncompress();
    }

    // Cache raw pointers once (minor perf win vs repeated calls)
    auto* inner = sparse.innerIndexPtr();
    auto* vals = sparse.valuePtr();
    auto* outer = sparse.outerIndexPtr();
    auto* nnz = sparse.innerNonZeroPtr();

    // Row metadata
    const int num_coeff_from = nnz[from_idx];
    const int num_holes_from = outer[from_idx + 1] - outer[from_idx];

    const int num_coeff_to = nnz[to_idx];
    const int num_holes_to = outer[to_idx + 1] - outer[to_idx];

    const Index from_start_vidx = outer[from_idx];
    const Index to_start_vidx = outer[to_idx];

    // Same number of coefficients: swap in place
    if (num_coeff_from == num_coeff_to) {
        std::swap_ranges(vals + from_start_vidx,
                         vals + from_start_vidx + num_coeff_from,
                         vals + to_start_vidx);
        std::swap_ranges(inner + from_start_vidx,
                         inner + from_start_vidx + num_coeff_from,
                         inner + to_start_vidx);
        return;
    }

    // If they fit within existing holes, swap directly
    const int min_number_holes = std::min(num_holes_from, num_holes_to);
    const int max_number_coeff = std::max(num_coeff_from, num_coeff_to);

    if (min_number_holes >= max_number_coeff) {
        std::swap_ranges(vals + from_start_vidx,
                         vals + from_start_vidx + max_number_coeff,
                         vals + to_start_vidx);
        std::swap_ranges(inner + from_start_vidx,
                         inner + from_start_vidx + max_number_coeff,
                         inner + to_start_vidx);
        std::swap(nnz[from_idx], nnz[to_idx]);
        return;
    }

    // Need scratch space: use RAII arrays (no zero-init, no leaks)
    const auto sz_if = static_cast<std::size_t>(num_holes_from);
    const auto sz_it = static_cast<std::size_t>(num_holes_to);

    // NOLINTBEGIN(hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    auto inner_from_copy = std::vector<StorageIndex>(sz_if);
    auto values_from_copy = std::vector<Values>(sz_if);
    auto inner_to_copy = std::vector<StorageIndex>(sz_it);
    auto values_to_copy = std::vector<Values>(sz_it);
    // NOLINTEND(hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

    // Copy out the two row slots
    std::memcpy(inner_from_copy.data(), inner + from_start_vidx,
                sz_if * sizeof(StorageIndex));
    std::memcpy(values_from_copy.data(), vals + from_start_vidx,
                sz_if * sizeof(Values));
    std::memcpy(inner_to_copy.data(), inner + to_start_vidx,
                sz_it * sizeof(StorageIndex));
    std::memcpy(values_to_copy.data(), vals + to_start_vidx,
                sz_it * sizeof(Values));

    // Move the between-rows block to make room
    const int num_holes_between_rows = outer[to_idx] - outer[from_idx + 1];

    std::memmove(
        inner + from_start_vidx + num_holes_to,
        inner + from_start_vidx + num_holes_from,
        static_cast<std::size_t>(num_holes_between_rows) * sizeof(*inner));
    std::memmove(
        vals + from_start_vidx + num_holes_to,
        vals + from_start_vidx + num_holes_from,
        static_cast<std::size_t>(num_holes_between_rows) * sizeof(*vals));

    // Lay down swapped rows plus preserved middle
    std::memcpy(inner + from_start_vidx, inner_to_copy.data(),
                sz_it * sizeof(StorageIndex));
    std::memcpy(inner + from_start_vidx + num_holes_between_rows + num_holes_to,
                inner_from_copy.data(), sz_if * sizeof(StorageIndex));

    std::memcpy(vals + from_start_vidx, values_to_copy.data(),
                sz_it * sizeof(Values));
    std::memcpy(vals + from_start_vidx + num_holes_between_rows + num_holes_to,
                values_from_copy.data(), sz_if * sizeof(Values));

    // Swap per-row nnz and fix the outers
    std::swap(nnz[from_idx], nnz[to_idx]);

    const int diff_holes = num_holes_to - num_holes_from;
    for (Index i = from_idx + 1; i <= to_idx; ++i) {
        outer[i] += diff_holes;
    }
}

// TODO: Fix complexity and remove supression
// NOLINTBEGIN(readability-function-cognitive-complexity)
void move_cols(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
               Index from_idx, Index to_idx) {
    PROFILE_FUNCTION();

    constexpr int limit_linear_search = 10;

    using StorageIndexPtr = decltype(sparse.innerIndexPtr());
    using ValuesPtr = decltype(sparse.valuePtr());
    using StorageIndex = std::remove_pointer_t<StorageIndexPtr>;
    using Values = std::remove_pointer_t<ValuesPtr>;

    // Ensure from_idx > to_idx
    if (from_idx >= to_idx) {
        if (from_idx == to_idx) {
            return;
        }
        std::swap(from_idx, to_idx);
    }

    if ((from_idx < 0) || (to_idx >= sparse.cols())) {
        if (VERBOSE) {
            std::cout << "Error while moving cols. Invalid indexes." << '\n';
        }
        return;
    }

    // The nnz is needed, easier to just work in uncompressed mode
    if (sparse.isCompressed()) {
        sparse.uncompress();
    }

    // Iterate each row
    for (Index iou = 0; iou < sparse.outerSize(); iou++) {
        const int& nnz = sparse.innerNonZeroPtr()[iou];
        if (nnz == 0) {
            continue;
        }  // Next-row if there are no coefficients

        bool fcol_found = false;
        bool tcol_found = false;
        Index finnidx = 0;  // From inner index
        Index tinnidx = 0;  // To inner index
        if (nnz < limit_linear_search) {
            // Linear search
            finnidx = sparse.outerIndexPtr()[iou];
            tinnidx = sparse.outerIndexPtr()[iou];
            for (Index inn = sparse.outerIndexPtr()[iou];
                 inn < sparse.outerIndexPtr()[iou] + nnz; inn++) {
                const int& col = sparse.innerIndexPtr()[inn];
                if (col <= from_idx) {
                    tinnidx++;
                    finnidx++;
                    if (col == from_idx) {
                        fcol_found = true;
                        finnidx--;
                    } else {
                        // finnidx = inn;
                    }
                } else if (col <= to_idx) {
                    tinnidx++;
                    if (col == to_idx) {
                        tcol_found = true;
                        tinnidx--;
                    } else {
                        // tinnidx = inn;
                    }
                }
            }
            // finnidx++;
            // tinnidx++;
        } else {
            // Binary search. Assuming from_idx <= to_idx (the swap at the
            // beginning ensures it)
            auto* inndx_ptr = std::lower_bound(
                sparse.innerIndexPtr() + sparse.outerIndexPtr()[iou],
                sparse.innerIndexPtr() + sparse.outerIndexPtr()[iou] + nnz,
                from_idx);
            finnidx = inndx_ptr - sparse.innerIndexPtr();
            if (inndx_ptr <
                sparse.innerIndexPtr() + sparse.outerIndexPtr()[iou] + nnz) {
                if (*inndx_ptr == from_idx) {
                    fcol_found = true;
                }

                inndx_ptr = std::lower_bound(
                    inndx_ptr,
                    sparse.innerIndexPtr() + sparse.outerIndexPtr()[iou] + nnz,
                    to_idx);
                tinnidx = inndx_ptr - sparse.innerIndexPtr();
                if (inndx_ptr < sparse.innerIndexPtr() +
                                    sparse.outerIndexPtr()[iou] + nnz) {
                    if (*inndx_ptr == to_idx) {
                        tcol_found = true;
                    }
                }
            }
            // The lower_bound is outside the search_space. no elements in the
            // from_idx or to_idx columns in this row.
        }

        // finndx and tinndx have the position of the columns if there is an
        // element or -1 otherwise

        if (fcol_found) {
            if (tcol_found) {
                // Both columns are in the row
                // Eassiest. Just swap values
                std::swap(sparse.valuePtr()[finnidx],
                          sparse.valuePtr()[tinnidx]);
            } else {
                // Only the from column is in the row
                auto temp_value = sparse.valuePtr()[finnidx];
                memmove(
                    sparse.innerIndexPtr() + finnidx,
                    sparse.innerIndexPtr() + finnidx + 1,
                    (tinnidx - finnidx - 1) * sizeof(*sparse.innerIndexPtr()));
                memmove(sparse.valuePtr() + finnidx,
                        sparse.valuePtr() + finnidx + 1,
                        (tinnidx - finnidx - 1) * sizeof(*sparse.valuePtr()));

                sparse.innerIndexPtr()[tinnidx - 1] =
                    static_cast<StorageIndex>(to_idx);
                sparse.valuePtr()[tinnidx - 1] = temp_value;
            }
        } else {
            if (tcol_found) {
                // Only the "to" column is in the row
                auto temp_value = sparse.valuePtr()[tinnidx];
                memmove(sparse.innerIndexPtr() + finnidx + 1,
                        sparse.innerIndexPtr() + finnidx,
                        (tinnidx - finnidx) * sizeof(*sparse.innerIndexPtr()));
                memmove(sparse.valuePtr() + finnidx + 1,
                        sparse.valuePtr() + finnidx,
                        (tinnidx - finnidx) * sizeof(*sparse.valuePtr()));

                sparse.innerIndexPtr()[finnidx] =
                    static_cast<StorageIndex>(from_idx);
                sparse.valuePtr()[finnidx] = temp_value;
            }
            // else: Nothing found. Do nothing
        }
    }
}

// NOLINTEND(readability-function-cognitive-complexity)

void move_row_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                  Index from_idx, Index to_idx) {
    move_rows(sparse, from_idx, to_idx);
    move_cols(sparse, from_idx, to_idx);
}

void remove_row(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                Index del_row_idx) {
    PROFILE_FUNCTION();
    if ((del_row_idx < sparse.rows()) && (del_row_idx >= 0)) {
        move_delete_row_fun(sparse, del_row_idx);
        sparse.conservativeResize(sparse.rows() - 1, sparse.cols());
    } else {
        if (VERBOSE) {
            std::cout
                << "Error: At removing row from sparse, invalid row index."
                << '\n';
        }
    }
}

void remove_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                Index del_col_idx) {
    PROFILE_FUNCTION();
    if ((del_col_idx < sparse.cols()) && (del_col_idx >= 0)) {
        move_delete_row_col_fun(sparse, del_col_idx);
        sparse.conservativeResize(sparse.rows(), sparse.cols() - 1);
    } else {
        if (VERBOSE) {
            std::cout
                << "Error: At removing col from sparse, invalid col index."
                << '\n';
        }
    }
}

void remove_row_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                    Index del_idx) {
    PROFILE_FUNCTION();
    if ((del_idx >= 0) && (del_idx < sparse.cols()) &&
        (del_idx < sparse.rows())) {
        move_delete_row_fun(sparse, del_idx);
        move_delete_row_col_fun(sparse, del_idx);
        sparse.conservativeResize(sparse.rows() - 1, sparse.cols() - 1);
    } else {
        if (VERBOSE) {
            std::cout << "Error: At removing row and col from sparse, invalid "
                         "row or col index."
                      << '\n';
        }
    }
}

bool is_trivial_zero(const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                     Index idx1, Index idx2) {
    // Check if element is a trivial zero in RowMajor SparseMatrix efficiently
    // (binary_search)
    if ((idx1 >= sparse.outerSize()) || (idx2 >= sparse.innerSize())) {
        return false;
    }
    if ((idx1 < 0) || (idx2 < 0)) {
        return false;
    }

    if (sparse.isCompressed()) {
        // binary_search return true when the element is found (not zero then)
        return !std::binary_search(
            &sparse.innerIndexPtr()[sparse.outerIndexPtr()[idx1]],
            &sparse.innerIndexPtr()[sparse.outerIndexPtr()[idx1 + 1]], idx2);
    } else {
        // binary_search return true when the element is found (not zero then)
        return !std::binary_search(
            &sparse.innerIndexPtr()[sparse.outerIndexPtr()[idx1]],
            &sparse.innerIndexPtr()[sparse.outerIndexPtr()[idx1] +
                                    sparse.innerNonZeroPtr()[idx1]],
            idx2);
    }
}

bool are_compressed_sparse_identical(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse1,
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse2) {
    // Comparing only in compressed mode
    if (!sparse1.isCompressed()) {
        sparse1.makeCompressed();
    }
    if (!sparse2.isCompressed()) {
        sparse2.makeCompressed();
    }

    bool equal_outer = false;
    bool equal_inner = false;
    bool equal_values = false;
    if (sparse1.outerSize() == sparse2.outerSize()) {
        // Check outter
        equal_outer = std::equal(sparse1.outerIndexPtr(),
                                 sparse1.outerIndexPtr() + sparse1.outerSize(),
                                 sparse2.outerIndexPtr());
        if (sparse1.outerIndexPtr()[sparse1.outerSize()] ==
            sparse2.outerIndexPtr()[sparse2.outerSize()]) {
            equal_inner =
                std::equal(sparse1.valuePtr(),
                           sparse1.valuePtr() +
                               sparse1.outerIndexPtr()[sparse1.outerSize()],
                           sparse2.valuePtr());
            equal_values =
                std::equal(sparse1.innerIndexPtr(),
                           sparse1.innerIndexPtr() +
                               sparse1.outerIndexPtr()[sparse1.outerSize()],
                           sparse2.innerIndexPtr());
        }
    }

    return equal_inner && equal_outer && equal_values;
}

[[nodiscard]] bool has_same_structure(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse1,
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse2) {
    // Fast-shape checks (size, compression mode, and # of stored entries)
    const bool same_size = (sparse1.rows() == sparse2.rows()) &&
                           (sparse1.cols() == sparse2.cols());
    const bool same_compression =
        (sparse1.isCompressed() == sparse2.isCompressed());
    const bool same_nnz = (sparse1.nonZeros() == sparse2.nonZeros());

    if (!(same_size && same_compression && same_nnz)) {
        return false;
    }

    // Structural comparison depends on compression state.
    return sparse1.isCompressed() ? compare_compressed(sparse1, sparse2)
                                  : compare_uncompressed(sparse1, sparse2);
}

void random_fill_sparse(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                        double sparsity_ratio, double min, double max,
                        int seed) {
    const auto rows = static_cast<Index>(sparse.rows());
    const auto cols = static_cast<Index>(sparse.cols());

    if ((sparsity_ratio < 0.0) || (sparsity_ratio > 1.0)) {
        if (VERBOSE) {
            std::cout << "Error, sparsity ratio should be between 0 and 1\n";
        }
        return;
    }

    if ((rows == Index{0}) || (cols == Index{0})) {
        return;  // nothing to do
    }

    const double density = 1.0 - sparsity_ratio;

    // Per-row (outer) reservation, clamped to [0, cols]
    const auto approx_values_per_outer = std::clamp<Index>(
        static_cast<Index>(std::llround(density * static_cast<double>(cols))),
        Index{0}, cols);

    // Total number of insertions, clamped to [0, rows*cols]
    const auto elements_to_insert = std::clamp<Index>(
        static_cast<Index>(std::llround(density * static_cast<double>(rows) *
                                        static_cast<double>(cols))),
        Index{0}, rows * cols);

    // Avoid Eigen::VectorXi (int) to prevent narrowing; use Index-sized vector.
    const std::vector<Index> reserve_per_outer(static_cast<std::size_t>(rows),
                                               approx_values_per_outer);
    sparse.reserve(reserve_per_outer);

    // Use generators with Index bounds to avoid narrowing to int.
    random_generators::IntGenerator<Index> rrig(Index{0}, rows - Index{1},
                                                seed);
    random_generators::IntGenerator<Index> rcig(Index{0}, cols - Index{1},
                                                seed + 1);
    random_generators::RealGenerator<double> rdg(min, max, seed + 2);

    for (auto k = Index{0}; k < elements_to_insert; ++k) {
        const auto ir = rrig.generate_random();
        const auto ic = rcig.generate_random();
        sparse.coeffRef(ir, ic) = rdg.generate_random();
    }
}

void set_to_zero(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse) {
    // Only when compling with IEEE754: 8 zero bytes == double +0.0
    memset(sparse.valuePtr(), 0,
           sparse.outerIndexPtr()[sparse.outerSize()] *
               sizeof(*sparse.valuePtr()));
}

void copy_values_same_nnz(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sp_dest,
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sp_from) {
    // Copy the non-zero values from the sparse matrix "sp_from" to "sp_dest".
    // This is a just a copy of the value vector, so matrices needs to have the
    // same nnz and same structure.

    // If debug, assert sparse structure is the same. This check is very costly,
    // only done in debug.
    PYCANHA_ASSERT(has_same_structure(sp_from, sp_dest),
                   "Matrices don't have the same structure");
#if !PYCANHA_USE_MKL
    memcpy(sp_dest.valuePtr(), sp_from.valuePtr(),
           sp_dest.nonZeros() * sizeof(*sp_dest.valuePtr()));
#elif PYCANHA_USE_MKL
    // A little bit better than memcpy
    const auto nnz = sp_dest.nonZeros();
    PYCANHA_ASSERT(nnz <= std::numeric_limits<MKL_INT>::max(),
                   "MKL integer range exceeded");
    const MKL_INT nnz_mkl = static_cast<MKL_INT>(nnz);
    // NOLINTNEXTLINE(misc-include-cleaner)
    cblas_dcopy(nnz_mkl, sp_from.valuePtr(), 1, sp_dest.valuePtr(), 1);
#endif
}

void copy_sum_values_same_nnz(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sp_dest,
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sp_from) {
    // Sum all the values from "sp_from" to "sp_dest". It sums directly the
    // value vector, so
    //  the sparse matrices should have the same structure for this operation to
    //  make sense.

    // If debug, assert sparse structure is the same. This check is very costly,
    // only done in debug.
    PYCANHA_ASSERT(has_same_structure(sp_from, sp_dest),
                   "Matrices don't have the same structure");

#if !PYCANHA_USE_MKL
    // Slow, but Eigen sum with wrapped vector doesn't work....
    for (int i = 0; i < sp_dest.nonZeros(); i++) {
        sp_dest.valuePtr()[i] += sp_from.valuePtr()[i];
    }
#elif PYCANHA_USE_MKL
    const auto nnz = sp_dest.nonZeros();
    PYCANHA_ASSERT(nnz <= std::numeric_limits<MKL_INT>::max(),
                   "MKL integer range exceeded");
    const MKL_INT nnz_mkl = static_cast<MKL_INT>(nnz);
    // NOLINTNEXTLINE(misc-include-cleaner)
    cblas_daxpy(nnz_mkl, 1.0, sp_from.valuePtr(), 1, sp_dest.valuePtr(), 1);
#endif
}

void copy_values_with_idx(double* dest, const double* from,
                          const std::vector<int>& dest_idx) {
    // Copy the values 'from' to 'dest'. The ival value is inserted in dest in
    // dest + dest_idx[i_nnz] No Checks performed.
    for (int ival = 0; ival < dest_idx.size(); ival++) {
        dest[dest_idx[ival]] = from[ival];
    }
}

void copy_2_values_with_idx(double* dest, const double* from,
                            const std::vector<int>& dest_idx_1,
                            const std::vector<int>& dest_idx_2) {
    // Copy the values 'from' to 'dest'. The ival value is inserted twice in:
    //  - dest in dest + dest_idx_1[i_nnz]
    //  - dest in dest + dest_idx_2[i_nnz]
    // No Checks performed.
    for (int ival = 0; ival < dest_idx_1.size(); ival++) {
        dest[dest_idx_1[ival]] = from[ival];
        dest[dest_idx_2[ival]] = from[ival];
    }
}

void copy_sum_values_with_idx(double* dest, const double* from,
                              const std::vector<int>& dest_idx) {
    // Same as the copy version, but sum the values.
    for (int ival = 0; ival < dest_idx.size(); ival++) {
        dest[dest_idx[ival]] += from[ival];
    }
}

void copy_sum_2_values_with_idx(double* dest, const double* from,
                                const std::vector<int>& dest_idx_1,
                                const std::vector<int>& dest_idx_2) {
    // Same as the copy version, but sum the values.
    for (int ival = 0; ival < dest_idx_1.size(); ival++) {
        dest[dest_idx_1[ival]] += from[ival];
        dest[dest_idx_2[ival]] += from[ival];
    }
}

std::tuple<Index, Index, double> get_row_col_value_from_value_idx(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index vidx) {
    // Given vidx, where -1 < vix < sparse.nonZeros()
    PYCANHA_ASSERT(vidx >= 0, "vidx should be positive");
    PYCANHA_ASSERT(vidx < sparse.nonZeros(),
                   "vidx out of limits of sparse non zeros");

    const double val = sparse.valuePtr()[vidx];
    const Index col = sparse.innerIndexPtr()[vidx];

    // We need to find the first outer idx where the value is <= vidx
    const auto* upper =
        std::upper_bound(sparse.outerIndexPtr(),
                         sparse.outerIndexPtr() + sparse.outerSize() + 1, vidx);
    const Index row = std::distance(sparse.outerIndexPtr(), upper) - 1;

    return std::make_tuple(row, col, val);
}

void print_sparse(const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse) {
    std::cout << '\n';
    std::cout << "   Sparse matrix   \n";
    std::cout << "-------------------\n";

    for (Eigen::Index row = 0; row < sparse.rows(); row++) {
        for (Eigen::Index col = 0; col < sparse.cols(); col++) {
            std::cout << std::fixed << std::setprecision(0)
                      << sparse.coeff(row, col) << " ";
        }
        std::cout << '\n';
    }
    std::cout << "-------------------\n";
    std::cout << '\n';
}

void print_sparse_values(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse) {
    std::cout << "Values: ";
    if (!sparse.isCompressed()) {
        for (int i = 0; i < sparse.outerSize(); i++) {
            int nnz = 0;
            for (int j = sparse.outerIndexPtr()[i];
                 j < sparse.outerIndexPtr()[i + 1]; j++) {
                if (nnz < sparse.innerNonZeroPtr()[i]) {
                    std::cout << sparse.valuePtr()[j] << ", ";
                } else {
                    std::cout << "_, ";
                }
                nnz++;
            }
        }
        std::cout << "\n";
    } else {
        for (int i = 0; i < sparse.outerIndexPtr()[sparse.outerSize()]; i++) {
            std::cout << sparse.valuePtr()[i] << ", ";
        }
        std::cout << "\n";
    }
}

void print_sparse_inner(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse) {
    std::cout << " Inner: ";
    if (!sparse.isCompressed()) {
        for (int i = 0; i < sparse.outerSize(); i++) {
            int nnz = 0;
            for (int j = sparse.outerIndexPtr()[i];
                 j < sparse.outerIndexPtr()[i + 1]; j++) {
                if (nnz < sparse.innerNonZeroPtr()[i]) {
                    std::cout << sparse.innerIndexPtr()[j] << ", ";
                } else {
                    std::cout << "_, ";
                }
                nnz++;
            }
        }
        std::cout << "\n";
    } else {
        for (int i = 0; i < sparse.outerIndexPtr()[sparse.outerSize()]; i++) {
            std::cout << sparse.innerIndexPtr()[i] << ", ";
        }
        std::cout << "\n";
    }
}

void print_sparse_outer(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse) {
    std::cout << " Outer: ";
    for (int i = 0; i < sparse.outerSize() + 1; i++) {
        std::cout << sparse.outerIndexPtr()[i] << ", ";
    }
    std::cout << "\n";
}

void print_sparse_nnz(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse) {
    std::cout << "   NNZ: ";
    if (!sparse.isCompressed()) {
        for (int i = 0; i < sparse.outerSize(); i++) {
            std::cout << sparse.innerNonZeroPtr()[i] << ", ";
        }
        std::cout << "\n";
    } else {
        std::cout << " **empty** \n";
    }
}

void print_sparse_format(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse) {
    if (sparse.isCompressed()) {
        std::cout << "Sparse Row Major in COMPRESSED format\n";
    } else {
        std::cout << "Sparse Row Major in UNCOMPRESSED format\n";
    }
}

void print_sparse_structure(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse) {
    std::cout << "*****************************\n";
    print_sparse_format(sparse);
    print_sparse(sparse);
    print_sparse_values(sparse);
    print_sparse_inner(sparse);
    print_sparse_outer(sparse);
    print_sparse_nnz(sparse);
    std::cout << "*****************************\n";
    std::cout.flush();
}

}  // namespace pycanha::sparse_utils

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
