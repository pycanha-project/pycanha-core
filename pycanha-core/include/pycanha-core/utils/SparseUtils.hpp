#pragma once

#include <Eigen/Sparse>
#include <tuple>
#include <vector>

#include "../config.hpp"
#include "../parameters.hpp"

namespace sparse_utils {

// Print information to std output
using pycanha::VERBOSE;

using Index = pycanha::Index;

void add_zero_row(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                  Index new_row_idx);
void add_zero_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                  Index new_col_idx);
void add_zero_row_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                      Index new_row_idx, Index new_col_idx);
void add_zero_diag_square(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse);

void move_rows(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
               Index from_idx, Index to_idx);
void move_cols(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
               Index from_idx, Index to_idx);
void move_row_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                  Index from_idx, Index to_idx);

/**
 * Move a row of a sparse matrix to the end setting all the coefficients to
 * zero. The sparse matrix is converted to non-compressed mode leaving "holes"
 * where the coefficient were.
 *
 * After applying this method, downsizing the matrix in one would have the same
 * effect as deleting the row
 *
 */
void remove_row(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                Index del_row_idx);
void remove_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                Index del_col_idx);
void remove_row_col(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                    Index del_idx);

bool is_trivial_zero(const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                     Index idx1, Index idx2);
bool are_compressed_sparse_identical(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse1,
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse2);
bool has_same_structure(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse1,
                        Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse2);

void random_fill_sparse(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse,
                        double sparsity_ratio, double min = 0.0,
                        double max = 9.5, int seed = 1000);
void set_to_zero(Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse);

void copy_values_same_nnz(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sp_dest,
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sp_from);
void copy_sum_values_same_nnz(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sp_dest,
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sp_from);
void copy_values_with_idx(double* dest, const double* from,
                          const std::vector<int>& dest_idx);
void copy_2_values_with_idx(double* dest, const double* from,
                            const std::vector<int>& dest_idx_1,
                            const std::vector<int>& dest_idx_2);
void copy_sum_values_with_idx(double* dest, const double* from,
                              const std::vector<int>& dest_idx);
void copy_sum_2_values_with_idx(double* dest, const double* from,
                                const std::vector<int>& dest_idx_1,
                                const std::vector<int>& dest_idx_2);

std::tuple<Index, Index, double> get_row_col_value_from_value_idx(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index vidx);

void print_sparse(const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse);
void print_sparse_values(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse);
void print_sparse_inner(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse);
void print_sparse_outer(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse);
void print_sparse_nnz(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse);
void print_sparse_format(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse);
void print_sparse_structure(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse);

}  // namespace sparse_utils
