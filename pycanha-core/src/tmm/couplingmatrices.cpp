#include "pycanha-core/tmm/couplingmatrices.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

#include "pycanha-core/parameters.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"

bool are_coupling_values_almost_equal(double val1, double val2) {
    return fabs(val1 - val2) <=
           ((fabs(val1) < fabs(val2) ? fabs(val2) : fabs(val1)) *
            ALMOST_EQUAL_COUPLING_EPSILON);
}

CouplingMatrices::CouplingMatrices() {}

inline int CouplingMatrices::get_num_diff_nodes() const {
    return sparse_dd.rows();
}

inline int CouplingMatrices::get_num_bound_nodes() const {
    return sparse_db.cols();
}

inline int CouplingMatrices::get_num_nodes() const {
    return sparse_db.rows() + sparse_db.cols();
}

const Eigen::SparseMatrix<double, Eigen::RowMajor> *
CouplingMatrices::return_sparse_dd() {
    sparse_dd.makeCompressed();
    return &sparse_dd;
}

const Eigen::SparseMatrix<double, Eigen::RowMajor>
CouplingMatrices::get_sparse_dd() const {
    return sparse_dd;
}

void CouplingMatrices::add_ovw_coupling_from_node_idxs(int idx1, int idx2,
                                                       double val) {
    _validate_coupling_call_add_generic(
        idx1, idx2, val, &CouplingMatrices::_add_ovw_coupling_sparse);
}
void CouplingMatrices::add_ovw_coupling_from_node_idxs_verbose(int idx1,
                                                               int idx2,
                                                               double val) {
    _validate_coupling_call_add_generic(
        idx1, idx2, val, &CouplingMatrices::_add_ovw_coupling_sparse_verbose);
}
void CouplingMatrices::add_sum_coupling_from_node_idxs(int idx1, int idx2,
                                                       double val) {
    _validate_coupling_call_add_generic(
        idx1, idx2, val, &CouplingMatrices::_add_sum_coupling_sparse);
}
void CouplingMatrices::add_sum_coupling_from_node_idxs_verbose(int idx1,
                                                               int idx2,
                                                               double val) {
    _validate_coupling_call_add_generic(
        idx1, idx2, val, &CouplingMatrices::_add_sum_coupling_sparse_verbose);
}
void CouplingMatrices::add_new_coupling_from_node_idxs(int idx1, int idx2,
                                                       double val) {
    _validate_coupling_call_add_generic(
        idx1, idx2, val, &CouplingMatrices::_add_new_coupling_sparse);
}

double CouplingMatrices::get_conductor_value_from_idx(Index idx1, Index idx2) {
    auto [sp_ptr, sp_idx1, sp_idx2] = _get_sp_ptr_and_sp_idx(idx1, idx2);
    if (sp_ptr == nullptr) {
        if (VERBOSE) std::cout << "ERROR! Invalid indexes." << std::endl;
        return nan("");
    }
    return sp_ptr->coeff(sp_idx1, sp_idx2);
}

void CouplingMatrices::set_conductor_value_from_idx(Index idx1, Index idx2,
                                                    double val) {
    // This method will change the conductor value only if it was previously
    // created

    auto [sp_ptr, sp_idx1, sp_idx2] = _get_sp_ptr_and_sp_idx(idx1, idx2);

    if (sp_ptr == nullptr) {
        if (VERBOSE)
            std::cout << "ERROR! Conductor has not been set." << std::endl;
        return;
    }

    if (!_validate_conductor_value(val)) {
        // TODO: Error/log handling
        if (VERBOSE)
            std::cout << "VALUE ERROR. Conductor should be positive."
                      << std::endl;
        return;
    }

    // Check existance of the conductor
    if (!SparseUtils::is_trivial_zero(*sp_ptr, sp_idx1, sp_idx2)) {
        sp_ptr->coeffRef(sp_idx1, sp_idx2) = val;
    } else {
        if (VERBOSE)
            std::cout << "Conductor does not exist. Value has not been set. "
                         "Add it before trying to change the value. "
                      << std::endl;
    }
}

double *CouplingMatrices::get_conductor_value_ref_from_idx(Index idx1,
                                                           Index idx2) {
    // TODO: make this method const, it does not change the model

    // Obtain where the value should be
    auto [sp_ptr, sp_idx1, sp_idx2] = _get_sp_ptr_and_sp_idx(idx1, idx2);

    // Invalid indexes return nullptr
    if (sp_ptr == nullptr) {
        return nullptr;
    }

    // Get the memory address of the value of a conductor.
    if (!SparseUtils::is_trivial_zero(*sp_ptr, sp_idx1, sp_idx2)) {
        return &(sp_ptr->coeffRef(sp_idx1, sp_idx2));
    } else {
        return nullptr;
    }
}

IntAddress CouplingMatrices::get_conductor_value_address_from_idx(Index idx1,
                                                                  Index idx2) {
    // TODO: make this method const, it does not change the model
    return reinterpret_cast<IntAddress>(
        get_conductor_value_ref_from_idx(idx1, idx2));
}

Eigen::SparseMatrix<double, Eigen::RowMajor>
CouplingMatrices::sparse_dd_copy() {
    sparse_dd.makeCompressed();
    return sparse_dd;
}

Eigen::SparseMatrix<double, Eigen::RowMajor>
CouplingMatrices::sparse_db_copy() {
    sparse_db.makeCompressed();
    return sparse_db;
}

Eigen::SparseMatrix<double, Eigen::RowMajor>
CouplingMatrices::sparse_bb_copy() {
    sparse_bb.makeCompressed();
    return sparse_bb;
}

int CouplingMatrices::get_num_diff_diff_couplings() const {
    return sparse_dd.nonZeros();
}

int CouplingMatrices::get_num_diff_bound_couplings() const {
    return sparse_db.nonZeros();
}

int CouplingMatrices::get_num_bound_bound_couplings() const {
    return sparse_bb.nonZeros();
}

int CouplingMatrices::get_num_total_couplings() const {
    return get_num_diff_diff_couplings() + get_num_diff_bound_couplings() +
           get_num_bound_bound_couplings();
}

std::tuple<int, int, double>
CouplingMatrices::get_idxs_and_coupling_value_from_coupling_idx(
    int cidx) const {
    if (cidx < 0) {
        std::cout << "Invalid coupling index: " << cidx << std::endl;
        std::cout << "Coupling Index should be positive." << std::endl;
        return std::make_tuple(-1, -1, nan(""));
    }

    if (cidx < get_num_diff_diff_couplings()) {
        // The cidx is less than the number of dd couplings, so return value
        // from sparse_dd
        return SparseUtils::get_row_col_value_from_value_idx(sparse_dd, cidx);
    }

    cidx -= get_num_diff_diff_couplings();

    if (cidx < get_num_diff_bound_couplings()) {
        // The cidx is between dd couplings and dd + db couplings, so return
        // value from sparse_
        auto [row, col, val] =
            SparseUtils::get_row_col_value_from_value_idx(sparse_db, cidx);

        // Col need to be increased to the number of diffusive nodes
        return std::make_tuple(row, col + sparse_dd.cols(), val);
    }

    cidx -= get_num_diff_bound_couplings();
    if (cidx < get_num_bound_bound_couplings()) {
        // The cidx is between dd couplings and dd + db couplings, so return
        // value from sparse_
        auto [row, col, val] =
            SparseUtils::get_row_col_value_from_value_idx(sparse_bb, cidx);

        // Col need to be increased to the number of diffusive nodes
        return std::make_tuple(row + sparse_dd.rows(), col + sparse_dd.cols(),
                               val);
    }

    // cidx is out of bounds
    std::cout << "Invalid coupling index: " << cidx + get_num_total_couplings()
              << std::endl;
    std::cout << "Coupling index >= Total num couplings." << std::endl;
    return std::make_tuple(-1, -1, nan(""));
}

bool CouplingMatrices::coupling_exists_from_idxs(int idx1, int idx2) {
    // The function get_conductor_value_ref_from_idx return nullptr if
    //  wrong indices or if the coupling doesn't exists. Maybe is not the most
    //  efficcient way of checking this, but it works for now

    auto val_ptr = get_conductor_value_ref_from_idx(idx1, idx2);
    return val_ptr != nullptr;
}

void CouplingMatrices::reserve(int nnz) {
    // TODO
}

void CouplingMatrices::_move_node(Index to_idx, Index from_idx) {
    // TODO
}

void CouplingMatrices::print_sparse() {
    std::cout << std::endl;
    std::cout << "     Kdd matrix    \n";
    std::cout << "-------------------\n";
    SparseUtils::print_sparse(sparse_dd);

    std::cout << std::endl;
    std::cout << "     Kdb matrix    \n";
    std::cout << "-------------------\n";
    SparseUtils::print_sparse(sparse_db);

    std::cout << std::endl;
    std::cout << "     Kbb matrix    \n";
    std::cout << "-------------------\n";
    SparseUtils::print_sparse(sparse_bb);
}

void CouplingMatrices::_add_node_diff(Index InsertPosition) {
    // NEW STORAGE USING K_dd and K_db
    SparseUtils::add_zero_row(sparse_db, InsertPosition);
    SparseUtils::add_zero_row_col(sparse_dd, InsertPosition, InsertPosition);
}

void CouplingMatrices::_add_node_bound(Index InsertPosition) {
    // NEW STORAGE USING K_dd and K_db
    SparseUtils::add_zero_col(sparse_db, InsertPosition);
    SparseUtils::add_zero_row_col(sparse_bb, InsertPosition, InsertPosition);
}

void CouplingMatrices::_remove_node_diff(Index idx) {
    // NEW STORAGE USING K_dd and K_db
    SparseUtils::remove_row(sparse_db, idx);
    SparseUtils::remove_row_col(sparse_dd, idx);
}

void CouplingMatrices::_remove_node_bound(Index idx) {
    // NEW STORAGE USING K_dd and K_db
    SparseUtils::remove_col(sparse_db, idx);
    SparseUtils::remove_row_col(sparse_bb, idx);
}

void CouplingMatrices::_add_ovw_coupling_sparse(
    Eigen::SparseMatrix<double, Eigen::RowMajor> &sparse, int sp_idx1,
    int sp_idx2, double val) {
    sparse.coeffRef(sp_idx1, sp_idx2) = val;
}

void CouplingMatrices::_add_ovw_coupling_sparse_verbose(
    Eigen::SparseMatrix<double, Eigen::RowMajor> &sparse, int sp_idx1,
    int sp_idx2, double val) {
    if (!SparseUtils::is_trivial_zero(sparse, sp_idx1, sp_idx2)) {
        // (VERBOSE) -> left this as a tag to find messages that should be
        // logged in the future
        double &coupling_val = sparse.coeffRef(sp_idx1, sp_idx2);
        if (!are_coupling_values_almost_equal(coupling_val, val)) {
            std::cout << "Duplicated coupling at indexes (" << sp_idx1 << ", "
                      << sp_idx2;
            std::cout << "). Overwritting old value: "
                      << sparse.coeff(sp_idx1, sp_idx2) << " with: " << val
                      << std::endl;
        }
        coupling_val = val;
    } else {
        _add_ovw_coupling_sparse(sparse, sp_idx1, sp_idx2, val);
    }
}

void CouplingMatrices::_add_sum_coupling_sparse(
    Eigen::SparseMatrix<double, Eigen::RowMajor> &sparse, int sp_idx1,
    int sp_idx2, double val) {
    sparse.coeffRef(sp_idx1, sp_idx2) += val;
}

void CouplingMatrices::_add_sum_coupling_sparse_verbose(
    Eigen::SparseMatrix<double, Eigen::RowMajor> &sparse, int sp_idx1,
    int sp_idx2, double val) {
    if (!SparseUtils::is_trivial_zero(sparse, sp_idx1, sp_idx2)) {
        // (VERBOSE) -> left this as a tag to find messages that should be
        // logged in the future
        std::cout << "Duplicated coupling at indexes (" << sp_idx1 << ", "
                  << sp_idx2;
        std::cout << "). Adding up old value: "
                  << sparse.coeff(sp_idx1, sp_idx2) << " with: " << val
                  << std::endl;
    }

    _add_sum_coupling_sparse(sparse, sp_idx1, sp_idx2, val);
}

void CouplingMatrices::_add_new_coupling_sparse(
    Eigen::SparseMatrix<double, Eigen::RowMajor> &sparse, int sp_idx1,
    int sp_idx2, double val) {
    if (!SparseUtils::is_trivial_zero(sparse, sp_idx1, sp_idx2)) {
        // TODO: RAISE EXCEPTION
        std::cout << "Duplicated coupling at indexes (" << sp_idx1 << ", "
                  << sp_idx2;
        std::cout << "). Old value: " << sparse.coeff(sp_idx1, sp_idx2)
                  << " left unchanged." << std::endl;
        return;
    }

    _add_ovw_coupling_sparse(sparse, sp_idx1, sp_idx2, val);
}

inline std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor> *, Index, Index>
CouplingMatrices::_get_sp_ptr_and_sp_idx(Index idx1, Index idx2) {
    if (!_validate_idxs(idx1, idx2)) {
        return {nullptr, -1, -1};
    }

    Eigen::SparseMatrix<double, Eigen::RowMajor> *sparse_ptr;
    int num_diff_nodes = get_num_diff_nodes();

    // Check quadrant (Kdd, Kdb or Kbb)
    if (idx2 < num_diff_nodes) {
        sparse_ptr = &(sparse_dd);

    } else if (idx1 < num_diff_nodes) {
        sparse_ptr = &(sparse_db);
        idx2 = idx2 - num_diff_nodes;
    } else {
        sparse_ptr = &(sparse_bb);
        idx2 = idx2 - num_diff_nodes;
        idx1 = idx1 - num_diff_nodes;
    }

    return {sparse_ptr, idx1, idx2};
}

inline bool CouplingMatrices::_validate_idxs(Index &idx1, Index &idx2) {
    if (idx1 < 0 || idx2 < 0 || idx2 >= get_num_nodes() ||
        idx1 >= get_num_nodes()) {
        return false;
    }

    if (idx1 > idx2) {
        std::swap(idx1, idx2);
    }

    return true;
}

inline bool CouplingMatrices::_validate_idx(Index idx) {
    if (idx < 0 || idx > get_num_nodes()) {
        return false;
    }
    return true;
}

inline bool CouplingMatrices::_validate_conductor_value(double value) {
    if (value < 0.0) {
        return false;
    }
    return true;
}

void CouplingMatrices::_validate_coupling_call_add_generic(
    int idx1, int idx2, double val, _add_coupling_generic add_coupling_fun) {
    // NOTE: All add_coupling function use this method, so invalid indexes or
    // invalid coupling values error are always printed.
    auto [sp_ptr, sp_idx1, sp_idx2] = _get_sp_ptr_and_sp_idx(idx1, idx2);
    if (sp_ptr == nullptr) {
        if (VERBOSE) std::cout << "ERROR! Invalid indexes.\n";
        return;
    }
    if (!_validate_conductor_value(val)) {
        // TODO: Error/log handling
        if (VERBOSE) std::cout << "VALUE ERROR. Coupling should be positive.\n";
        return;
    }

    // Call the appropiate add_coupling function
    (this->*add_coupling_fun)(*sp_ptr, sp_idx1, sp_idx2, val);
}
