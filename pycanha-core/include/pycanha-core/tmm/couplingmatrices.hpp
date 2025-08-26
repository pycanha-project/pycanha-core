
/// C++ Implementation of ThermalConductos (TCs)
/**
 * Contain all the information regarding the thermal conductors in such a way
 * that it can be easily an efficiently readed by the solvers. The information
 * is stored in sparse matrices as defined by Eigen
 * (Eigen::SparseMatrix<double>).
 *
 */

#pragma once

#include <Eigen/Sparse>
#include <tuple>

#include "./nodes.hpp"

namespace pycanha {

class CouplingMatrices {
    friend class Nodes;
    friend class Couplings;
    friend class ThermalNetwork;

  public:
    // Interface
    [[nodiscard]] inline int get_num_diff_nodes() const;
    [[nodiscard]] inline int get_num_bound_nodes() const;
    [[nodiscard]] inline int get_num_nodes() const;

    // Numerical values. TODO: Make private
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_dd;
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_db;
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_bb;

    // Constructors
    CouplingMatrices();

    // Add couplings
    void add_ovw_coupling_from_node_idxs(
        int idx1, int idx2,
        double val);  ///< Add coupling, overwriting if already exists.
    void add_ovw_coupling_from_node_idxs_verbose(
        int idx1, int idx2,
        double val);  ///< Add coupling, overwriting if already exists. Prints
                      ///< message if overwrite.
    void add_sum_coupling_from_node_idxs(
        int idx1, int idx2,
        double val);  ///< Add coupling, sum the values if already exists.
    void add_sum_coupling_from_node_idxs_verbose(
        int idx1, int idx2,
        double val);  ///< Add coupling, sum the values if already exists.
                      ///< Prints message if sum.
    void add_new_coupling_from_node_idxs(
        int idx1, int idx2,
        double val);  ///< Add coupling, only if wasn't there already.

    // Get value
    double get_conductor_value_from_idx(Index idx1, Index idx2);

    // Set value
    void set_conductor_value_from_idx(Index i, Index j, double val);

    // Get value ref
    double *get_conductor_value_ref_from_idx(Index idx1, Index idx2);

    // Get address of value
    IntAddress get_conductor_value_address_from_idx(Index i, Index j);

    // Return the sparse matrix representation
    const Eigen::SparseMatrix<double, Eigen::RowMajor> *return_sparse_dd();
    [[nodiscard]] const Eigen::SparseMatrix<double, Eigen::RowMajor>
    get_sparse_dd() const;

    // Return a copy of the sparse matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_dd_copy();
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_db_copy();
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_bb_copy();

    // Num couplings
    [[nodiscard]] int get_num_diff_diff_couplings() const;
    [[nodiscard]] int get_num_diff_bound_couplings() const;
    [[nodiscard]] int get_num_bound_bound_couplings() const;
    [[nodiscard]] int get_num_total_couplings() const;

    // For coupling iteration
    [[nodiscard]] std::tuple<int, int, double>
    get_idxs_and_coupling_value_from_coupling_idx(int cidx) const;
    bool coupling_exists_from_idxs(int idx1, int idx2);

    void print_sparse();

    void reserve(int nnz);

  private:
    void _move_node(Index to_idx, Index from_idx);

    inline bool _is_thermal_nodes_valid();

    inline std::tuple<Index, Index, bool> _get_internal_node_numbers(Index i,
                                                                     Index j);

    // NEW FOR REFACTORING
    void _add_node_diff(Index insert_idx);
    void _add_node_bound(Index insert_idx);

    void _remove_node_diff(Index idx);
    void _remove_node_bound(Index idx);

    using AddCouplingGeneric = void (CouplingMatrices::*)(
        Eigen::SparseMatrix<double, Eigen::RowMajor> &, int, int, double);

    void _add_ovw_coupling_sparse(
        Eigen::SparseMatrix<double, Eigen::RowMajor> &sparse, int sp_idx1,
        int sp_idx2, double val);  ///< Add entry to the sparse matrix,
                                   ///< overwriting if already exists.

    void _add_ovw_coupling_sparse_verbose(
        Eigen::SparseMatrix<double, Eigen::RowMajor> &sparse, int sp_idx1,
        int sp_idx2,
        double val);  ///< Add entry to the sparse matrix, overwriting if
                      ///< already exists. Prints message if overwrite.

    void _add_sum_coupling_sparse(
        Eigen::SparseMatrix<double, Eigen::RowMajor> &sparse, int sp_idx1,
        int sp_idx2, double val);  ///< Add entry to the sparse matrix, sum the
                                   ///< values if already exists.

    void _add_sum_coupling_sparse_verbose(
        Eigen::SparseMatrix<double, Eigen::RowMajor> &sparse, int sp_idx1,
        int sp_idx2,
        double val);  ///< Add entry to the sparse matrix, sum the values if
                      ///< already exists. Prints message if already exists.

    void _add_new_coupling_sparse(
        Eigen::SparseMatrix<double, Eigen::RowMajor> &sparse, int sp_idx1,
        int sp_idx2, double val);  ///< Add entry to the sparse matrix, only if
                                   ///< wasn't there already.

    // From internal numbers idx1 and idx2, get the correct matrix ptr (Kdd, Kdb
    // or Kbb) and the indexes in that matrix
    inline std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor> *, Index,
                      Index>
    _get_sp_ptr_and_sp_idx(Index idx1, Index idx2);

    // Ensures idx2 > idx1 (swap values if not) and that the indexes
    //  are inside the matrices. Return fale otherwise.
    // TODO Raise exception instead of returning bool
    inline bool _validate_idxs(Index &idx1, Index &idx2);
    inline bool _validate_idx(Index idx);
    inline bool _validate_conductor_value(double value);

    void _validate_coupling_call_add_generic(
        int idx1, int idx2, double val, AddCouplingGeneric add_coupling_fun);

    // TODO
    void _diff_to_bound(Index insert_position, Index int_bound_num);
};

}  // namespace pycanha
