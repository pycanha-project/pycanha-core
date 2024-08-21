#include <Eigen/Sparse>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <string>

#include "pycanha-core/utils/SparseUtils.hpp"
#include "pycanha-core/utils/random_generators.hpp"

// NOLINTBEGIN(readability-function-cognitive-complexity)

using namespace SparseUtils;  // NOLINT

// Size of sparse for testing
int ROW_SIZE = 20;
int COL_SIZE = 20;

void trivial_zero_and_identity_test() {
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse1(ROW_SIZE, COL_SIZE);
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse2(ROW_SIZE, COL_SIZE);

    random_fill_sparse(sparse1, 0.4, -9.5, 9.5, 100);
    random_fill_sparse(sparse2, 0.4, -9.5, 9.5, 100);
    // sparse1 and sparse2 use the same seed -> They are identical

    sparse1.makeCompressed();
    sparse2.makeCompressed();

    // Matrices should be equal
    REQUIRE(are_compressed_sparse_identical(sparse1, sparse2));

    sparse1.coeffRef(sparse1.rows() - 2, sparse1.cols() - 3) += 1.0;

    // Now matrices shouldn't be equal
    REQUIRE(!are_compressed_sparse_identical(sparse1, sparse2));

    // Any non-zero value in this matrix should be a trivial zero
    for (Index ir = 0; ir < sparse2.rows(); ir++) {
        for (Index ic = 0; ic < sparse2.cols(); ic++) {
            if (sparse2.coeff(ir, ic) == 0.0) {
                REQUIRE(is_trivial_zero(sparse2, ir, ic));
            } else {
                REQUIRE(!is_trivial_zero(sparse2, ir, ic));
            }
        }
    }
}

void zero_row_col_test() {
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse1(ROW_SIZE, COL_SIZE);
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse2(ROW_SIZE, COL_SIZE);

    random_fill_sparse(sparse1, 0.4, -9.5, 9.5, 100);
    random_fill_sparse(sparse2, 0.4, -9.5, 9.5, 100);

    std::vector<Index> zero_row_indexes;
    std::vector<Index> zero_col_indexes;

    constexpr int num_zero_row_cols = 50;

    bool use_row_col_function = false;
    for (int i = 0; i < num_zero_row_cols; i++) {
        random_generators::IntGenerator<Index> row_rand_gen(
            0, sparse1.rows() - 1, i + 567);
        random_generators::IntGenerator<Index> col_rand_gen(
            0, sparse1.cols() - 1, i + 567 + num_zero_row_cols);

        Index row = row_rand_gen.generate_random();
        Index col = col_rand_gen.generate_random();

        // Use the two different methods alternatively
        use_row_col_function = !use_row_col_function;
        if (use_row_col_function) {
            add_zero_row_col(sparse1, row, col);
        } else {
            add_zero_row(sparse1, row);
            add_zero_col(sparse1, col);
        }

        auto row_lower_bound = std::lower_bound(zero_row_indexes.begin(),
                                                zero_row_indexes.end(), row);
        for (auto it = row_lower_bound; it < zero_row_indexes.end(); it++) {
            (*it)++;
        }
        zero_row_indexes.insert(row_lower_bound, row);

        auto col_lower_bound = std::lower_bound(zero_col_indexes.begin(),
                                                zero_col_indexes.end(), col);
        for (auto it = col_lower_bound; it < zero_col_indexes.end(); it++) {
            (*it)++;
        }
        zero_col_indexes.insert(col_lower_bound, col);
    }

    // Add invalid indexes at the end to avoid accessing invalid indexes
    zero_row_indexes.push_back(-1);
    zero_col_indexes.push_back(-1);

    Index row2 = 0;
    Index col2 = 0;
    int zero_row_idx = 0;
    int zero_col_idx = 0;

    for (Index row1 = 0; row1 < sparse1.rows(); row1++) {
        zero_col_idx = 0;
        col2 = 0;

        if (row1 == zero_row_indexes[zero_row_idx]) {
            // Zero row
            for (Index col1 = 0; col1 < sparse1.cols(); col1++) {
                // Check added rows are trivial zeros
                REQUIRE(is_trivial_zero(sparse1, row1, col1));
            }
            zero_row_idx++;
        } else {
            for (Index col1 = 0; col1 < sparse1.cols(); col1++) {
                if (col1 == zero_col_indexes[zero_col_idx]) {
                    // Check added cols are trivial zeros
                    REQUIRE(is_trivial_zero(sparse1, row1, col1));
                    zero_col_idx++;
                } else {
                    // Check the coefficients are in place and the same as the
                    // sparse2
                    REQUIRE(sparse2.coeff(row2, col2) ==
                            sparse1.coeff(row1, col1));
                    col2++;
                }
            }
            row2++;
        }
    }
}

void move_test() {
    // Test move columns and rows
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse(ROW_SIZE, COL_SIZE);
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_copy(ROW_SIZE,
                                                             COL_SIZE);

    random_fill_sparse(sparse, 0.4, -9.5, 9.5, 100);
    random_fill_sparse(sparse_copy, 0.4, -9.5, 9.5, 100);

    constexpr int num_permutation = 100;

    std::vector<int> rows_idxs(sparse.rows()), cols_idxs(sparse.cols());
    // Initialize vectors in sequence
    for (int i = 0; i < std::ssize(rows_idxs); i++) {
        rows_idxs[i] = i;
    }
    for (int i = 0; i < std::ssize(cols_idxs); i++) {
        cols_idxs[i] = i;
    }

    random_generators::IntGenerator<Index> row_rand_gen(0, sparse.rows() - 1,
                                                        100);
    random_generators::IntGenerator<Index> col_rand_gen(0, sparse.cols() - 1,
                                                        120);

    // Permute randomly using move_rows and move_cols
    for (int iper = 0; iper < num_permutation; iper++) {
        Index from_row = row_rand_gen.generate_random();
        Index to_row = row_rand_gen.generate_random();
        Index from_col = col_rand_gen.generate_random();
        Index to_col = col_rand_gen.generate_random();

        move_rows(sparse, from_row, to_row);
        move_cols(sparse, from_col, to_col);

        std::swap(rows_idxs[from_row], rows_idxs[to_row]);
        std::swap(cols_idxs[from_col], cols_idxs[to_col]);
    }

    // Permute randomly using move_row_cols
    for (int iper = 0; iper < num_permutation; iper++) {
        Index from_row_col;
        Index to_row_col;
        // For rectangular matrices, use the lowest size
        if (sparse.rows() > sparse.cols()) {
            from_row_col = col_rand_gen.generate_random();
            to_row_col = col_rand_gen.generate_random();
        } else {
            from_row_col = row_rand_gen.generate_random();
            to_row_col = row_rand_gen.generate_random();
        }

        move_row_col(sparse, from_row_col, to_row_col);

        std::swap(rows_idxs[from_row_col], rows_idxs[to_row_col]);
        std::swap(cols_idxs[from_row_col], cols_idxs[to_row_col]);
    }

    // Move rows again to the original position
    for (int i = 0; i < std::ssize(rows_idxs); i++) {
        if (rows_idxs[i] != i) {
            for (int j = i + 1; j < std::ssize(rows_idxs); j++) {
                if (rows_idxs[j] == i) {
                    std::swap(rows_idxs[i], rows_idxs[j]);
                    move_rows(sparse, i, j);
                    break;
                }
            }
        }
    }

    // Move cols again to the original position
    for (int i = 0; i < std::ssize(cols_idxs); i++) {
        if (cols_idxs[i] != i) {
            for (int j = i + 1; j < std::ssize(cols_idxs); j++) {
                if (cols_idxs[j] == i) {
                    std::swap(cols_idxs[i], cols_idxs[j]);
                    move_cols(sparse, i, j);
                    break;
                }
            }
        }
    }

    // Assert sparse matrix is identical
    REQUIRE(are_compressed_sparse_identical(sparse_copy, sparse));

    // Try to move to invalid positions. No exception thrown and matrix should
    // be equal

    // From to same position
    move_rows(sparse, 0, 0);
    move_cols(sparse, 0, 0);
    move_rows(sparse, sparse.rows() - 1, sparse.rows() - 1);
    move_cols(sparse, sparse.cols() - 1, sparse.cols() - 1);
    move_rows(sparse, (sparse.rows() - 1) / 2, (sparse.rows() - 1) / 2);
    move_cols(sparse, (sparse.cols() - 1) / 2, (sparse.cols() - 1) / 2);

    // To/from invalid indexes
    move_rows(sparse, (sparse.rows() - 1) / 2, sparse.rows());
    move_rows(sparse, (sparse.rows() - 1) / 2, -1);
    move_rows(sparse, sparse.rows(), sparse.rows());
    move_rows(sparse, -1, -1);
    move_rows(sparse, sparse.rows(), -1);
    move_cols(sparse, (sparse.cols() - 1) / 2, sparse.cols());
    move_cols(sparse, (sparse.cols() - 1) / 2, -1);
    move_cols(sparse, sparse.cols(), sparse.cols());
    move_cols(sparse, -1, -1);
    move_cols(sparse, sparse.cols(), -1);
    REQUIRE(are_compressed_sparse_identical(sparse_copy, sparse));

    // To/from invalid indexes (opposite sense)
    move_rows(sparse, sparse.rows(), (sparse.rows() - 1) / 2);
    move_rows(sparse, -1, (sparse.rows() - 1) / 2);
    move_rows(sparse, -1, sparse.rows());
    move_cols(sparse, sparse.cols(), (sparse.cols() - 1) / 2);
    move_cols(sparse, -1, (sparse.cols() - 1) / 2);
    move_cols(sparse, -1, sparse.cols());
    REQUIRE(are_compressed_sparse_identical(sparse_copy, sparse));
}

void remove_test() {
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse(ROW_SIZE, COL_SIZE);
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_copy(ROW_SIZE,
                                                             COL_SIZE);

    std::vector<int> original_row_idxs;
    std::vector<int> original_col_idxs;

    original_row_idxs.reserve(sparse.rows());
    original_col_idxs.reserve(sparse.cols());

    for (int i = 0; i < sparse.rows(); i++) {
        original_row_idxs.push_back(i);
    }
    for (int i = 0; i < sparse.cols(); i++) {
        original_col_idxs.push_back(i);
    }

    // Remove rows/cols randomly and compress the matrix every 4 removals
    random_generators::IntGenerator<Index> random_bool(0, 1, 666);
    constexpr int compress_matrix_every = 4;

    bool row_or_col_to_remove;
    int remove_count = 0;

    while ((sparse.rows() > 1) && (sparse.cols() > 1)) {
        if (sparse.rows() <= 1) {
            row_or_col_to_remove = 1;  // Row 0, col 1
        } else if (sparse.cols() <= 1) {
            row_or_col_to_remove = 0;  // Row 0, col 1
        } else {
            row_or_col_to_remove =
                random_bool.generate_random();  // Row 0, col 1
        }

        if (row_or_col_to_remove) {
            // Col
            random_generators::IntGenerator<Index> col_generator(
                0, sparse.cols() - 1, 923 + remove_count);
            Index idx = col_generator.generate_random();
            remove_col(sparse, idx);
            original_col_idxs.erase(original_col_idxs.begin() + idx);
        } else {
            // Row
            random_generators::IntGenerator<Index> row_generator(
                0, sparse.rows() - 1, 923 + remove_count);
            Index idx = row_generator.generate_random();
            remove_row(sparse, idx);
            original_row_idxs.erase(original_row_idxs.begin() + idx);
        }

        // Test matrix is ok
        for (Index irow = 0;
             irow < static_cast<Index>(std::ssize(original_row_idxs)); irow++) {
            for (Index icol = 0;
                 icol < static_cast<Index>(std::ssize(original_col_idxs));
                 icol++) {
                REQUIRE(sparse.coeff(irow, icol) ==
                        sparse_copy.coeff(original_row_idxs[irow],
                                          original_col_idxs[icol]));
            }
        }

        remove_count++;
        if (remove_count % compress_matrix_every == 0) {
            sparse.makeCompressed();
        }
    }

    // TODO: Change philosophy of SparseUtils and raise exceptions when trying
    // to remove invalid rows/cols
    //  Test no error raised when removing wrong row/col or last col/row
    Eigen::SparseMatrix<double, Eigen::RowMajor> sp1(1, 1);
    sp1.coeffRef(0, 0) = 1.0;
    remove_col(sp1, -1);
    remove_row(sp1, -1);
    remove_col(sp1, 1);
    remove_row(sp1, 1);
    remove_col(sp1, 0);
    remove_row(sp1, 0);
    remove_col(sp1, 0);
    remove_row(sp1, 0);

    Eigen::SparseMatrix<double, Eigen::RowMajor> sp2(1, 1);
    sp2.coeffRef(0, 0) = 1.0;
    remove_row(sp2, 0);
    remove_col(sp2, 0);
    remove_row(sp2, 0);
    remove_col(sp2, 0);
}

TEST_CASE("SparseUtils Tests") {
    SECTION("SQUARE MATRICES TEST") {
        ROW_SIZE = 20;
        COL_SIZE = 20;
        trivial_zero_and_identity_test();
        zero_row_col_test();
        move_test();
        remove_test();
    }

    SECTION("RECTANGULAR MATRICES TEST 1") {
        ROW_SIZE = 10;
        COL_SIZE = 30;
        trivial_zero_and_identity_test();
        zero_row_col_test();
        move_test();
        remove_test();
    }

    SECTION("RECTANGULAR MATRICES TEST 2") {
        ROW_SIZE = 30;
        COL_SIZE = 10;
        trivial_zero_and_identity_test();
        zero_row_col_test();
        move_test();
        remove_test();
    }
}

// NOLINTEND(readability-function-cognitive-complexity)
