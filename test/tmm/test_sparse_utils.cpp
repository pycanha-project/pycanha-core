#include <Eigen/Sparse>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <vector>

#include "pycanha-core/utils/RandomGenerators.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"

// NOLINTBEGIN(readability-function-cognitive-complexity)

using namespace sparse_utils;  // NOLINT

namespace {
using VectorIndex = pycanha::VectorIndex;  // NOLINT(misc-include-cleaner)

// Size of sparse for testing
int ROW_SIZE = 20;  // NOLINT
int COL_SIZE = 20;  // NOLINT

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
            0, sparse1.rows() - 1, static_cast<unsigned int>(i + 567));
        random_generators::IntGenerator<Index> col_rand_gen(
            0, sparse1.cols() - 1,
            static_cast<unsigned int>(i + 567 + num_zero_row_cols));

        const Index row = row_rand_gen.generate_random();
        const Index col = col_rand_gen.generate_random();

        // Use the two different methods alternatively
        use_row_col_function = !use_row_col_function;
        if (use_row_col_function) {
            add_zero_row_col(sparse1, row, col);
        } else {
            add_zero_row(sparse1, row);
            add_zero_col(sparse1, col);
        }

        // NOLINTNEXTLINE(boost-use-ranges,modernize-use-ranges)
        auto row_lower_bound = std::lower_bound(zero_row_indexes.begin(),
                                                zero_row_indexes.end(), row);
        // NOLINTNEXTLINE(boost-use-ranges,modernize-use-ranges)
        for (auto it = row_lower_bound; it < zero_row_indexes.end(); ++it) {
            (*it)++;
        }
        // NOLINTNEXTLINE(boost-use-ranges,modernize-use-ranges)
        zero_row_indexes.insert(row_lower_bound, row);

        // NOLINTNEXTLINE(boost-use-ranges,modernize-use-ranges)
        auto col_lower_bound = std::lower_bound(zero_col_indexes.begin(),
                                                zero_col_indexes.end(), col);
        // NOLINTNEXTLINE(boost-use-ranges,modernize-use-ranges)
        for (auto it = col_lower_bound; it < zero_col_indexes.end(); ++it) {
            (*it)++;
        }
        // NOLINTNEXTLINE(boost-use-ranges,modernize-use-ranges)
        zero_col_indexes.insert(col_lower_bound, col);
    }

    // Add invalid indexes at the end to avoid accessing invalid indexes
    zero_row_indexes.push_back(-1);
    zero_col_indexes.push_back(-1);

    Index row2 = 0;
    Index col2 = 0;
    VectorIndex zero_row_idx = 0;
    VectorIndex zero_col_idx = 0;

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
                    REQUIRE((sparse2.coeff(row2, col2) ==
                             sparse1.coeff(row1, col1)));
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

    std::vector<int> rows_idxs(static_cast<VectorIndex>(sparse.rows()));
    std::vector<int> cols_idxs(static_cast<VectorIndex>(sparse.cols()));
    // Initialize vectors in sequence
    for (VectorIndex i = 0; i < static_cast<VectorIndex>(rows_idxs.size());
         i++) {
        rows_idxs[i] = static_cast<int>(i);
    }
    for (VectorIndex i = 0; i < static_cast<VectorIndex>(cols_idxs.size());
         i++) {
        cols_idxs[i] = static_cast<int>(i);
    }

    random_generators::IntGenerator<Index> row_rand_gen(0, sparse.rows() - 1,
                                                        100);
    random_generators::IntGenerator<Index> col_rand_gen(0, sparse.cols() - 1,
                                                        120);

    // Permute randomly using move_rows and move_cols
    for (int iper = 0; iper < num_permutation; iper++) {
        const Index from_row = row_rand_gen.generate_random();
        const Index to_row = row_rand_gen.generate_random();
        const Index from_col = col_rand_gen.generate_random();
        const Index to_col = col_rand_gen.generate_random();

        move_rows(sparse, from_row, to_row);
        move_cols(sparse, from_col, to_col);

        std::swap(rows_idxs[static_cast<VectorIndex>(from_row)],
                  rows_idxs[static_cast<VectorIndex>(to_row)]);
        std::swap(cols_idxs[static_cast<VectorIndex>(from_col)],
                  cols_idxs[static_cast<VectorIndex>(to_col)]);
    }

    // Permute randomly using move_row_cols
    for (int iper = 0; iper < num_permutation; iper++) {
        Index from_row_col = 0;
        Index to_row_col = 0;
        // For rectangular matrices, use the lowest size
        if (sparse.rows() > sparse.cols()) {
            from_row_col = col_rand_gen.generate_random();
            to_row_col = col_rand_gen.generate_random();
        } else {
            from_row_col = row_rand_gen.generate_random();
            to_row_col = row_rand_gen.generate_random();
        }

        move_row_col(sparse, from_row_col, to_row_col);

        std::swap(rows_idxs[static_cast<VectorIndex>(from_row_col)],
                  rows_idxs[static_cast<VectorIndex>(to_row_col)]);
        std::swap(cols_idxs[static_cast<VectorIndex>(from_row_col)],
                  cols_idxs[static_cast<VectorIndex>(to_row_col)]);
    }

    // Move rows again to the original position
    for (VectorIndex i = 0; i < static_cast<VectorIndex>(rows_idxs.size());
         i++) {
        if (rows_idxs[i] != static_cast<int>(i)) {
            for (VectorIndex j = i + 1;
                 j < static_cast<VectorIndex>(rows_idxs.size()); j++) {
                if (rows_idxs[j] == static_cast<int>(i)) {
                    std::swap(rows_idxs[i], rows_idxs[j]);
                    move_rows(sparse, static_cast<Index>(i),
                              static_cast<Index>(j));
                    break;
                }
            }
        }
    }

    // Move cols again to the original position
    for (VectorIndex i = 0; i < static_cast<VectorIndex>(cols_idxs.size());
         i++) {
        if (cols_idxs[i] != static_cast<int>(i)) {
            for (VectorIndex j = i + 1;
                 j < static_cast<VectorIndex>(cols_idxs.size()); j++) {
                if (cols_idxs[j] == static_cast<int>(i)) {
                    std::swap(cols_idxs[i], cols_idxs[j]);
                    move_cols(sparse, static_cast<Index>(i),
                              static_cast<Index>(j));
                    break;
                }
            }
        }
    }

    // Assert sparse matrix is identical
    REQUIRE(are_compressed_sparse_identical(
        sparse_copy, sparse));  // NOLINT(misc-const-correctness)

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
    REQUIRE(are_compressed_sparse_identical(
        sparse_copy, sparse));  // NOLINT(misc-const-correctness)

    // To/from invalid indexes (opposite sense)
    move_rows(sparse, sparse.rows(), (sparse.rows() - 1) / 2);
    move_rows(sparse, -1, (sparse.rows() - 1) / 2);
    move_rows(sparse, -1, sparse.rows());
    move_cols(sparse, sparse.cols(), (sparse.cols() - 1) / 2);
    move_cols(sparse, -1, (sparse.cols() - 1) / 2);
    move_cols(sparse, -1, sparse.cols());
    REQUIRE(are_compressed_sparse_identical(
        sparse_copy, sparse));  // NOLINT(misc-const-correctness)
}

void remove_test() {
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse(ROW_SIZE, COL_SIZE);
    const Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_copy(ROW_SIZE,
                                                                   COL_SIZE);

    std::vector<int> original_row_idxs;
    std::vector<int> original_col_idxs;

    original_row_idxs.reserve(static_cast<VectorIndex>(sparse.rows()));
    original_col_idxs.reserve(static_cast<VectorIndex>(sparse.cols()));

    for (Index i = 0; i < sparse.rows(); i++) {
        original_row_idxs.push_back(static_cast<int>(i));
    }
    for (Index i = 0; i < sparse.cols(); i++) {
        original_col_idxs.push_back(static_cast<int>(i));
    }

    // Remove rows/cols randomly and compress the matrix every 4 removals
    random_generators::IntGenerator<Index> random_bool(0, 1, 666);
    constexpr int compress_matrix_every = 4;

    bool row_or_col_to_remove = false;
    int remove_count = 0;

    while ((sparse.rows() > 1) && (sparse.cols() > 1)) {
        if (sparse.rows() <= 1) {
            row_or_col_to_remove = true;  // Row 0, col 1
        } else if (sparse.cols() <= 1) {
            row_or_col_to_remove = false;  // Row 0, col 1
        } else {
            row_or_col_to_remove =
                (random_bool.generate_random() != 0);  // Row 0, col 1
        }

        if (row_or_col_to_remove) {
            // Col
            random_generators::IntGenerator<Index> col_generator(
                0, sparse.cols() - 1,
                static_cast<unsigned int>(923 + remove_count));
            const Index idx = col_generator.generate_random();
            remove_col(sparse, idx);
            original_col_idxs.erase(
                original_col_idxs.begin() +
                static_cast<std::vector<int>::difference_type>(idx));
        } else {
            // Row
            random_generators::IntGenerator<Index> row_generator(
                0, sparse.rows() - 1,
                static_cast<unsigned int>(923 + remove_count));
            const Index idx = row_generator.generate_random();
            remove_row(sparse, idx);
            original_row_idxs.erase(
                original_row_idxs.begin() +
                static_cast<std::vector<int>::difference_type>(idx));
        }

        // Test matrix is ok
        for (Index irow = 0;
             irow < static_cast<Index>(std::ssize(original_row_idxs)); irow++) {
            for (Index icol = 0;
                 icol < static_cast<Index>(std::ssize(original_col_idxs));
                 icol++) {
                REQUIRE(
                    (sparse.coeff(irow, icol) ==
                     sparse_copy.coeff(
                         static_cast<Index>(
                             original_row_idxs[static_cast<VectorIndex>(irow)]),
                         static_cast<Index>(
                             original_col_idxs[static_cast<VectorIndex>(
                                 icol)]))));
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

void has_same_structure_test() {
    using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    // Small helpers
    const auto pick_nonzero = [](const Sparse& m, Index& row_index,
                                 Index& col_index) -> bool {
        for (Index row = 0; row < m.rows(); ++row) {
            auto it = Sparse::InnerIterator(m, row);
            if (it) {
                row_index = static_cast<Index>(it.row());
                col_index = static_cast<Index>(it.col());
                return true;
            }
        }
        return false;
    };
    const auto pick_zero = [](const Sparse& m, Index& r, Index& c) -> bool {
        for (Index i = 0; i < m.rows(); i++) {
            for (Index j = 0; j < m.cols(); j++) {
                if (m.coeff(i, j) == 0.0) {
                    r = i;
                    c = j;
                    return true;
                }
            }
        }
        return false;
    };

    // --- Uncompressed: identical -> true
    {
        Sparse a(ROW_SIZE, COL_SIZE);
        Sparse b(ROW_SIZE, COL_SIZE);
        random_fill_sparse(a, 0.35, -5.0, 5.0, 4242);
        random_fill_sparse(b, 0.35, -5.0, 5.0, 4242);
        a.uncompress();
        b.uncompress();
        REQUIRE(has_same_structure(a, b));

        // Value change only (keep nonzero) -> true
        Index r = 0;
        Index c = 0;
        if (pick_nonzero(a, r, c)) {
            a.coeffRef(r, c) += 1.0;
            REQUIRE(has_same_structure(a, b));
        }

        // Different reserved boundaries per row (capacity) -> false
        {
            const auto rows = static_cast<int>(a.rows());
            Eigen::VectorXi cap_a = Eigen::VectorXi::Constant(rows, 2);
            Eigen::VectorXi cap_b = Eigen::VectorXi::Constant(rows, 6);
            a.reserve(cap_a);
            b.reserve(cap_b);
            REQUIRE_FALSE(has_same_structure(a, b));
        }
    }

    // --- Uncompressed: insert a new nonzero changes used counts -> false
    {
        Sparse u1(ROW_SIZE, COL_SIZE);
        Sparse u2(ROW_SIZE, COL_SIZE);
        random_fill_sparse(u1, 0.25, -3.0, 3.0, 777);
        random_fill_sparse(u2, 0.25, -3.0, 3.0, 777);
        u1.uncompress();
        u2.uncompress();
        Index zr = 0;
        Index zc = 0;
        if (pick_zero(u1, zr, zc)) {
            u1.coeffRef(zr, zc) = 3.14;
            REQUIRE_FALSE(has_same_structure(u1, u2));
        }
    }

    // --- Compression mismatch -> false
    {
        Sparse c1(ROW_SIZE, COL_SIZE);
        Sparse c2(ROW_SIZE, COL_SIZE);
        random_fill_sparse(c1, 0.40, -9.0, 9.0, 9898);
        random_fill_sparse(c2, 0.40, -9.0, 9.0, 9898);
        c1.makeCompressed();
        c2.uncompress();
        REQUIRE_FALSE(has_same_structure(c1, c2));
    }

    // --- Compressed: identical -> true; remove an entry -> false
    {
        Sparse k1(ROW_SIZE, COL_SIZE);
        Sparse k2(ROW_SIZE, COL_SIZE);
        random_fill_sparse(k1, 0.40, -9.0, 9.0, 2024);
        random_fill_sparse(k2, 0.40, -9.0, 9.0, 2024);
        k1.makeCompressed();
        k2.makeCompressed();
        REQUIRE(has_same_structure(k1, k2));

        // Value change only in existing slot -> true
        Index r = 0;
        Index c = 0;
        if (pick_nonzero(k1, r, c)) {
            k1.coeffRef(r, c) += 0.5;  // stays nonzero
            REQUIRE(has_same_structure(k1, k2));
        }

        // Remove a nonzero (nnz differs) -> false
        Index rr = 0;
        Index cc = 0;
        if (pick_nonzero(k1, rr, cc)) {
            k1.coeffRef(rr, cc) = 0.0;
            k1.prune(0.0);
            REQUIRE_FALSE(has_same_structure(k1, k2));
        }
    }

    // --- Size mismatch -> false
    {
        Sparse bigger(ROW_SIZE + 1, COL_SIZE);
        Sparse ref(ROW_SIZE, COL_SIZE);
        random_fill_sparse(bigger, 0.30, -2.0, 2.0, 1);
        random_fill_sparse(ref, 0.30, -2.0, 2.0, 1);
        bigger.makeCompressed();
        ref.makeCompressed();
        REQUIRE_FALSE(has_same_structure(bigger, ref));
    }
}

}  // namespace

TEST_CASE("sparse_utils Tests") {
    SECTION("SQUARE MATRICES TEST") {
        ROW_SIZE = 20;
        COL_SIZE = 20;
        trivial_zero_and_identity_test();
        zero_row_col_test();
        move_test();
        remove_test();
        has_same_structure_test();
    }

    SECTION("RECTANGULAR MATRICES TEST 1") {
        ROW_SIZE = 10;
        COL_SIZE = 30;
        trivial_zero_and_identity_test();
        zero_row_col_test();
        move_test();
        remove_test();
        has_same_structure_test();
    }

    SECTION("RECTANGULAR MATRICES TEST 2") {
        ROW_SIZE = 30;
        COL_SIZE = 10;
        trivial_zero_and_identity_test();
        zero_row_col_test();
        move_test();
        remove_test();
        has_same_structure_test();
    }
}

// NOLINTEND(readability-function-cognitive-complexity)
