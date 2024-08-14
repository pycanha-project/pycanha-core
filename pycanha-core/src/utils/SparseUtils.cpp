
#include "pycanha-core/utils/SparseUtils.hpp"

#include <iostream>

#include "pycanha-core/utils/RandomGenerators.hpp"

// USE MKL FUNCTION IF AVAILABLE
#if defined(CYCANHA_USE_MKL)
#include "mkl.h"
#endif

namespace SparseUtils {

void _add_zero_row(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                   Index new_row_idx) {
    if (sparse.isCompressed()) {
        // Copy the outer indices to the new positions
        auto src_ptr_outer = sparse.outerIndexPtr() + new_row_idx;
        memmove(
            src_ptr_outer + 1, src_ptr_outer,
            (sparse.outerSize() - new_row_idx - 1) * sizeof(*src_ptr_outer));
    } else {
        // Copy the outer and nnz indices to the new positions
        auto src_ptr_outer = sparse.outerIndexPtr() + new_row_idx;
        auto src_ptr_nnz = sparse.innerNonZeroPtr() + new_row_idx;
        size_t elems_to_copy = (sparse.outerSize() - new_row_idx - 1);
        memmove(src_ptr_outer + 1, src_ptr_outer,
                elems_to_copy * sizeof(*src_ptr_outer));
        memmove(src_ptr_nnz + 1, src_ptr_nnz,
                elems_to_copy * sizeof(*src_ptr_nnz));

        sparse.innerNonZeroPtr()[new_row_idx] = 0;
    }
}

void _add_zero_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
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
            Index ival_start = sparse.outerIndexPtr()[row];
            Index nnz = sparse.innerNonZeroPtr()[row];
            for (Index ival = ival_start; ival < ival_start + nnz; ival++) {
                if (sparse.innerIndexPtr()[ival] >= new_col_idx) {
                    sparse.innerIndexPtr()[ival]++;
                }
            }
        }
    }
}

void _add_zero_row_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
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
        auto src_ptr_outer = sparse.outerIndexPtr() + new_row_idx;
        memmove(
            src_ptr_outer + 1, src_ptr_outer,
            (sparse.outerSize() - new_row_idx - 1) * sizeof(*src_ptr_outer));
    } else {
        // The column index (innerIndexPtr) of any coefficient equal or bigger
        // than new_index should be increased in one.
        for (Index row = 0; row < sparse.outerSize(); row++) {
            Index ival_start = sparse.outerIndexPtr()[row];
            Index nnz = sparse.innerNonZeroPtr()[row];
            for (Index ival = ival_start; ival < ival_start + nnz; ival++) {
                if (sparse.innerIndexPtr()[ival] >= new_col_idx) {
                    sparse.innerIndexPtr()[ival]++;
                }
            }
        }

        // Copy the outer and nnz indices to the new positions
        auto src_ptr_outer = sparse.outerIndexPtr() + new_row_idx;
        auto src_ptr_nnz = sparse.innerNonZeroPtr() + new_row_idx;
        size_t elems_to_copy = (sparse.outerSize() - new_row_idx - 1);
        memmove(src_ptr_outer + 1, src_ptr_outer,
                elems_to_copy * sizeof(*src_ptr_outer));
        memmove(src_ptr_nnz + 1, src_ptr_nnz,
                elems_to_copy * sizeof(*src_ptr_nnz));

        sparse.innerNonZeroPtr()[new_row_idx] = 0;
    }
}

void add_zero_row(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                  Index new_row_idx) {
    // Change the size of the matrix without deleting the values
    sparse.conservativeResize(sparse.rows() + 1, sparse.cols());
    _add_zero_row(sparse, new_row_idx);
}

void add_zero_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                  Index new_col_idx) {
    // Change the size of the matrix without deleting the values
    sparse.conservativeResize(sparse.rows(), sparse.cols() + 1);
    _add_zero_col(sparse, new_col_idx);
}

void add_zero_row_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                      Index new_row_idx, Index new_col_idx) {
    // Change the size of the matrix without deleting the values
    sparse.conservativeResize(sparse.rows() + 1, sparse.cols() + 1);
    _add_zero_row(sparse, new_row_idx);
    _add_zero_col(sparse, new_col_idx);
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

void move_rows(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
               Index from_idx, Index to_idx) {
    PROFILE_FUNCTION();

    typedef decltype(sparse.innerIndexPtr()) StorageIndex_ptr;
    typedef decltype(sparse.valuePtr()) Values_ptr;

    typedef std::remove_pointer<StorageIndex_ptr>::type StorageIndex;
    typedef std::remove_pointer<Values_ptr>::type Values;

    // Ensure from_idx > to_idx
    if (from_idx >= to_idx) {
        if (from_idx == to_idx) {
            return;
        }
        std::swap(from_idx, to_idx);
    }

    if ((from_idx < 0) || (to_idx >= sparse.rows())) {
        if (VERBOSE) {
            std::cout << "Error while moving rows. Invalid indexes."
                      << std::endl;
        }
        return;
    }

    // Easier to work in non-compressed mode
    if (sparse.isCompressed()) {
        sparse.uncompress();
    }

    // Assuming from_idx and to_idx lies within the size of the matrix
    int num_coeff_from = sparse.innerNonZeroPtr()[from_idx];
    int num_holes_from =
        sparse.outerIndexPtr()[from_idx + 1] - sparse.outerIndexPtr()[from_idx];

    int num_coeff_to = sparse.innerNonZeroPtr()[to_idx];
    int num_holes_to =
        sparse.outerIndexPtr()[to_idx + 1] - sparse.outerIndexPtr()[to_idx];

    Index from_start_vidx = sparse.outerIndexPtr()[from_idx];
    Index to_start_vidx = sparse.outerIndexPtr()[to_idx];

    // Same number of coefficientes: A simple swap of the values and the inners
    if (num_coeff_from == num_coeff_to) {
        std::swap_ranges(sparse.valuePtr() + from_start_vidx,
                         sparse.valuePtr() + from_start_vidx + num_coeff_from,
                         sparse.valuePtr() + to_start_vidx);
        std::swap_ranges(
            sparse.innerIndexPtr() + from_start_vidx,
            sparse.innerIndexPtr() + from_start_vidx + num_coeff_from,
            sparse.innerIndexPtr() + to_start_vidx);
        return;
    }

    // When different number of coefficients, chek if fit in the holes
    int min_number_holes = std::min(num_holes_from, num_holes_to);
    int max_number_coeff = std::max(num_coeff_from, num_coeff_to);

    if (min_number_holes >= max_number_coeff) {
        // The values can be swapped directly. They fit.
        std::swap_ranges(sparse.valuePtr() + from_start_vidx,
                         sparse.valuePtr() + from_start_vidx + max_number_coeff,
                         sparse.valuePtr() + to_start_vidx);
        std::swap_ranges(
            sparse.innerIndexPtr() + from_start_vidx,
            sparse.innerIndexPtr() + from_start_vidx + max_number_coeff,
            sparse.innerIndexPtr() + to_start_vidx);
        std::swap(sparse.innerNonZeroPtr()[from_idx],
                  sparse.innerNonZeroPtr()[to_idx]);
        return;
    }

    // There is not room (not enough "holes") to fit the arrays

    // Copy the from a to row, inner and values arrays to a buffer
    StorageIndex_ptr inner_from_copy = static_cast<StorageIndex_ptr>(
        malloc(num_holes_from * sizeof(StorageIndex)));
    Values_ptr values_from_copy =
        static_cast<Values_ptr>(malloc(num_holes_from * sizeof(Values)));
    StorageIndex_ptr inner_to_copy = static_cast<StorageIndex_ptr>(
        malloc(num_holes_to * sizeof(StorageIndex)));
    Values_ptr values_to_copy =
        static_cast<Values_ptr>(malloc(num_holes_to * sizeof(Values)));

    memcpy(inner_from_copy, sparse.innerIndexPtr() + from_start_vidx,
           num_holes_from * sizeof(StorageIndex));
    memcpy(values_from_copy, sparse.valuePtr() + from_start_vidx,
           num_holes_from * sizeof(Values));
    memcpy(inner_to_copy, sparse.innerIndexPtr() + to_start_vidx,
           num_holes_to * sizeof(StorageIndex));
    memcpy(values_to_copy, sparse.valuePtr() + to_start_vidx,
           num_holes_to * sizeof(Values));

    // Reorder the values
    int num_holes_between_rows =
        sparse.outerIndexPtr()[to_idx] - sparse.outerIndexPtr()[from_idx + 1];
    memmove(sparse.innerIndexPtr() + from_start_vidx + num_holes_to,
            sparse.innerIndexPtr() + from_start_vidx + num_holes_from,
            num_holes_between_rows * sizeof(*sparse.innerIndexPtr()));

    memmove(sparse.valuePtr() + from_start_vidx + num_holes_to,
            sparse.valuePtr() + from_start_vidx + num_holes_from,
            num_holes_between_rows * sizeof(*sparse.valuePtr()));

    memcpy(sparse.innerIndexPtr() + from_start_vidx, inner_to_copy,
           num_holes_to * sizeof(StorageIndex));
    memcpy(sparse.innerIndexPtr() + from_start_vidx + num_holes_between_rows +
               num_holes_to,
           inner_from_copy, num_holes_from * sizeof(StorageIndex));

    memcpy(sparse.valuePtr() + from_start_vidx, values_to_copy,
           num_holes_to * sizeof(Values));
    memcpy(sparse.valuePtr() + from_start_vidx + num_holes_between_rows +
               num_holes_to,
           values_from_copy, num_holes_from * sizeof(Values));

    // Swap nnz and outer
    std::swap(sparse.innerNonZeroPtr()[from_idx],
              sparse.innerNonZeroPtr()[to_idx]);

    int diff_holes = num_holes_to - num_holes_from;
    for (Index i = from_idx + 1; i <= to_idx; i++) {
        sparse.outerIndexPtr()[i] += diff_holes;
    }

    free(inner_from_copy);
    free(values_from_copy);
    free(inner_to_copy);
    free(values_to_copy);
}

void move_cols(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
               Index from_idx, Index to_idx) {
    PROFILE_FUNCTION();

    constexpr int LIMIT_LINEAR_SEARCH = 10;

    typedef decltype(sparse.innerIndexPtr()) StorageIndex_ptr;
    typedef decltype(sparse.valuePtr()) Values_ptr;

    typedef std::remove_pointer<StorageIndex_ptr>::type StorageIndex;
    typedef std::remove_pointer<Values_ptr>::type Values;

    // Ensure from_idx > to_idx
    if (from_idx >= to_idx) {
        if (from_idx == to_idx) {
            return;
        }
        std::swap(from_idx, to_idx);
    }

    if ((from_idx < 0) || (to_idx >= sparse.cols())) {
        if (VERBOSE) {
            std::cout << "Error while moving cols. Invalid indexes."
                      << std::endl;
        }
        return;
    }

    // The nnz is needed, easier to just work in uncompressed mode
    if (sparse.isCompressed()) {
        sparse.uncompress();
    }

    // Iterate each row
    for (Index iou = 0; iou < sparse.outerSize(); iou++) {
        int& nnz = sparse.innerNonZeroPtr()[iou];
        if (nnz == 0) {
            continue;
        }  // Next-row if there are no coefficients

        bool fcol_found = false;
        bool tcol_found = false;
        Index finnidx;  // From inner index
        Index tinnidx;  // To inner index
        if (nnz < LIMIT_LINEAR_SEARCH) {
            // Linear search
            finnidx = sparse.outerIndexPtr()[iou];
            tinnidx = sparse.outerIndexPtr()[iou];
            for (Index inn = sparse.outerIndexPtr()[iou];
                 inn < sparse.outerIndexPtr()[iou] + nnz; inn++) {
                int& col = sparse.innerIndexPtr()[inn];
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
            auto inndx_ptr = std::lower_bound(
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

                sparse.innerIndexPtr()[tinnidx - 1] = to_idx;
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

                sparse.innerIndexPtr()[finnidx] = from_idx;
                sparse.valuePtr()[finnidx] = temp_value;
            }
            // else: Nothing found. Do nothing
        }
    }
}

void move_row_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                  Index from_idx, Index to_idx) {
    move_rows(sparse, from_idx, to_idx);
    move_cols(sparse, from_idx, to_idx);
}

void _move_delete_row(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
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

void _move_delete_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                      Index move_col_idx) {
    PROFILE_FUNCTION();
    // Removing a node will left a "hole", so the matrix should be in uncompress
    // mode
    if (sparse.isCompressed()) {
        sparse.uncompress();
    }

    // Remove column

    // All inner indices should be reduced in 1
    for (Index iouter = 0; iouter < sparse.outerSize(); iouter++) {
        Index ival_start =
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

void remove_row(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                Index del_row_idx) {
    PROFILE_FUNCTION();
    if ((del_row_idx < sparse.rows()) && (del_row_idx >= 0)) {
        _move_delete_row(sparse, del_row_idx);
        sparse.conservativeResize(sparse.rows() - 1, sparse.cols());
    } else {
        if (VERBOSE) {
            std::cout
                << "Error: At removing row from sparse, invalid row index."
                << std::endl;
        }
    }
}

void remove_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                Index del_col_idx) {
    PROFILE_FUNCTION();
    if ((del_col_idx < sparse.cols()) && (del_col_idx >= 0)) {
        _move_delete_col(sparse, del_col_idx);
        sparse.conservativeResize(sparse.rows(), sparse.cols() - 1);
    } else {
        if (VERBOSE) {
            std::cout
                << "Error: At removing col from sparse, invalid col index."
                << std::endl;
        }
    }
}

void remove_row_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                    Index del_idx) {
    PROFILE_FUNCTION();
    if ((del_idx >= 0) && (del_idx < sparse.cols()) &&
        (del_idx < sparse.rows())) {
        _move_delete_row(sparse, del_idx);
        _move_delete_col(sparse, del_idx);
        sparse.conservativeResize(sparse.rows() - 1, sparse.cols() - 1);
    } else {
        if (VERBOSE) {
            std::cout << "Error: At removing row and col from sparse, invalid "
                         "row or col index."
                      << std::endl;
        }
    }
}

bool is_trivial_zero(const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                     int idx1, int idx2) {
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
        if (sparse1.outerIndexPtr()[sparse1.outerSize() + 1] ==
            sparse2.outerIndexPtr()[sparse2.outerSize() + 1]) {
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

    if (equal_inner && equal_outer && equal_values) {
        return true;
    } else {
        return false;
    }
}

bool has_same_structure(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse1,
                        Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse2) {
    // Check if two sparse matrices have the same internal structure. They can
    // be in compressed or uncompressed mode.
    bool same_size;
    bool same_compression;
    bool same_nnz;
    bool same_rows;
    bool same_cols;

    same_size = (sparse1.cols() == sparse2.cols()) &&
                (sparse1.rows() == sparse2.rows());
    same_compression = (sparse1.isCompressed() && sparse2.isCompressed()) ||
                       ((!sparse1.isCompressed()) && (!sparse2.isCompressed()));
    same_nnz = sparse1.nonZeros() == sparse2.nonZeros();

    if (!(same_size && same_compression && same_nnz)) {
        return false;
    }

    if (sparse1.isCompressed()) {
        same_rows =
            memcmp(sparse1.innerIndexPtr(), sparse2.innerIndexPtr(),
                   sparse1.nonZeros() * sizeof(*sparse1.innerIndexPtr())) == 0;
        same_cols = memcmp(sparse1.outerIndexPtr(), sparse2.outerIndexPtr(),
                           (sparse1.outerSize() + 1) *
                               sizeof(*sparse1.outerIndexPtr())) == 0;
    } else {
        if (memcmp(sparse1.innerNonZeroPtr(), sparse2.innerNonZeroPtr(),
                   sparse1.outerSize() * sizeof(*sparse1.innerNonZeroPtr())) ==
            0) {
            if (!(sparse1.outerIndexPtr()[0] == sparse2.outerIndexPtr()[0])) {
                return false;
            }
            for (int i = 0; i < sparse1.outerSize(); i++) {
                int nnz = 0;
                if (!(sparse1.outerIndexPtr()[i + 1] ==
                      sparse2.outerIndexPtr()[i + 1])) {
                    return false;
                }
                for (int j = sparse1.outerIndexPtr()[i];
                     j < sparse1.outerIndexPtr()[i + 1]; j++) {
                    if (nnz < sparse1.innerNonZeroPtr()[i]) {
                        if (!(sparse1.innerIndexPtr()[j] ==
                              sparse2.innerIndexPtr()[j])) {
                            return false;
                        }
                    } else {
                        // Nothing, the reserved values can contain garbage.
                    }
                    nnz++;
                }
            }
            same_rows = true;
            same_cols = true;
        } else {
            same_rows = false;
            same_cols = false;
        }
    }

    return (same_rows && same_cols);
}

void random_fill_sparse(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                        double sparsity_ratio, double min, double max,
                        int seed) {
    int rows = sparse.rows();
    int cols = sparse.cols();

    if ((sparsity_ratio < 0.0) || (sparsity_ratio > 1.0)) {
        if (VERBOSE) {
            std::cout << "Error, sparsity ratio should be between 0 and 1"
                      << std::endl;
        }
        return;
    }

    int approx_values_per_row = (1.0 - sparsity_ratio) * cols;
    int elements_to_insert = (1.0 - sparsity_ratio) * cols * rows;
    sparse.reserve(
        Eigen::VectorXi::Constant(sparse.rows(), approx_values_per_row));

    RandomGenerators::IntGenerator<int> rrig(0, rows - 1, seed);
    RandomGenerators::IntGenerator<int> rcig(0, cols - 1, seed + 1);
    RandomGenerators::RealGenerator<double> rdg(0.0, 9.5, seed + 2);

    Index ir;
    Index ic;
    for (int i = 0; i < elements_to_insert; i++) {
        ir = rrig.generate_random();
        ic = rcig.generate_random();
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
    CYCANHA_ASSERT(has_same_structure(sp_from, sp_dest),
                   "Matrices don't have the same structure");
#if defined(CYCANHA_USE_ONLY_EIGEN)
    memcpy(sp_dest.valuePtr(), sp_from.valuePtr(),
           sp_dest.nonZeros() * sizeof(*sp_dest.valuePtr()));
#elif defined(CYCANHA_USE_MKL)
    // A little bit better than memcpy
    cblas_dcopy(sp_dest.nonZeros(), sp_from.valuePtr(), 1, sp_dest.valuePtr(),
                1);
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
    CYCANHA_ASSERT(has_same_structure(sp_from, sp_dest),
                   "Matrices don't have the same structure");

#if defined(CYCANHA_USE_ONLY_EIGEN)
    // Slow, but Eigen sum with wrapped vector doesn't work....
    for (int i = 0; i < sp_dest.nonZeros(); i++) {
        sp_dest.valuePtr()[i] += sp_from.valuePtr()[i];
    }
#elif defined(CYCANHA_USE_MKL)
    cblas_daxpy(sp_dest.nonZeros(), 1.0, sp_from.valuePtr(), 1,
                sp_dest.valuePtr(), 1);
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

std::tuple<int, int, double> get_row_col_value_from_value_idx(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, int vidx) {
    // Given vidx, where -1 < vix < sparse.nonZeros()
    CYCANHA_ASSERT(vidx >= 0, "vidx should be positive");
    CYCANHA_ASSERT(vidx < sparse.nonZeros(),
                   "vidx out of limits of sparse non zeros");

    double val = sparse.valuePtr()[vidx];
    int col = sparse.innerIndexPtr()[vidx];

    // We need to find the first outer idx where the value is <= vidx
    auto upper =
        std::upper_bound(sparse.outerIndexPtr(),
                         sparse.outerIndexPtr() + sparse.outerSize() + 1, vidx);
    int row = std::distance(sparse.outerIndexPtr(), upper) - 1;

    return std::make_tuple(row, col, val);
}

void print_sparse(const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse) {
    std::cout << std::endl;
    std::cout << "   Sparse matrix   \n";
    std::cout << "-------------------\n";

    for (Eigen::Index row = 0; row < sparse.rows(); row++) {
        for (Eigen::Index col = 0; col < sparse.cols(); col++) {
            std::cout << std::fixed << std::setprecision(0)
                      << sparse.coeff(row, col) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-------------------\n";
    std::cout << std::endl;
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
    if (!sparse.isCompressed()) {
        std::cout << "\n";
    } else {
        std::cout << "\n";
    }
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

}  // namespace SparseUtils
