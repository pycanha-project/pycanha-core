#include "pycanha-core/tmm/couplings.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"

namespace pycanha {

namespace {

[[nodiscard]] std::optional<int> to_int_node_number(Index node_num) {
    if (node_num > static_cast<Index>(std::numeric_limits<int>::max()) ||
        node_num < static_cast<Index>(std::numeric_limits<int>::min())) {
        if (VERBOSE) {
            std::cout << "Couplings: Node number " << node_num
                      << " exceeds supported int range." << '\n';
        }
        return std::nullopt;
    }
    return static_cast<int>(node_num);
}

template <typename SparseMatrix>
void ensure_sparse_dimensions(SparseMatrix& matrix, Index rows, Index cols) {
    if (matrix.rows() >= rows && matrix.cols() >= cols) {
        return;
    }

    SparseMatrix expanded(rows, cols);

    for (Index outer = 0; outer < matrix.outerSize(); ++outer) {
        for (typename SparseMatrix::InnerIterator it(matrix, outer); it; ++it) {
            if (it.row() < rows && it.col() < cols) {
                expanded.insert(it.row(), it.col()) = it.value();
            }
        }
    }

    expanded.makeCompressed();
    matrix = std::move(expanded);
}

}  // namespace

Couplings::Couplings(std::shared_ptr<Nodes> nodes) noexcept
    : _nodes(std::move(nodes)) {
    synchronize_structure();
}

const CouplingMatrices& Couplings::get_coupling_matrices() const noexcept {
    return _matrices;
}

CouplingMatrices& Couplings::get_coupling_matrices() noexcept {
    return _matrices;
}

double Couplings::get_coupling_value(Index node_num_1, Index node_num_2) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return _matrices.get_conductor_value_from_idx(indices->first,
                                                  indices->second);
}

void Couplings::set_coupling_value(Index node_num_1, Index node_num_2,
                                   double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.set_conductor_value_from_idx(indices->first, indices->second,
                                           value);
}

void Couplings::add_ovw_coupling(Index node_num_1, Index node_num_2,
                                 double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.add_ovw_coupling_from_node_idxs(indices->first, indices->second,
                                              value);
}

void Couplings::add_ovw_coupling(const Coupling& coupling) {
    add_ovw_coupling(coupling.get_node_1(), coupling.get_node_2(),
                     coupling.get_value());
}

void Couplings::add_ovw_coupling_verbose(Index node_num_1, Index node_num_2,
                                         double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.add_ovw_coupling_from_node_idxs_verbose(indices->first,
                                                      indices->second, value);
}

void Couplings::add_ovw_coupling_verbose(const Coupling& coupling) {
    add_ovw_coupling_verbose(coupling.get_node_1(), coupling.get_node_2(),
                             coupling.get_value());
}

void Couplings::add_sum_coupling(Index node_num_1, Index node_num_2,
                                 double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.add_sum_coupling_from_node_idxs(indices->first, indices->second,
                                              value);
}

void Couplings::add_sum_coupling(const Coupling& coupling) {
    add_sum_coupling(coupling.get_node_1(), coupling.get_node_2(),
                     coupling.get_value());
}

void Couplings::add_sum_coupling_verbose(Index node_num_1, Index node_num_2,
                                         double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.add_sum_coupling_from_node_idxs_verbose(indices->first,
                                                      indices->second, value);
}

void Couplings::add_sum_coupling_verbose(const Coupling& coupling) {
    add_sum_coupling_verbose(coupling.get_node_1(), coupling.get_node_2(),
                             coupling.get_value());
}

void Couplings::add_new_coupling(Index node_num_1, Index node_num_2,
                                 double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.add_new_coupling_from_node_idxs(indices->first, indices->second,
                                              value);
}

void Couplings::add_new_coupling(const Coupling& coupling) {
    add_new_coupling(coupling.get_node_1(), coupling.get_node_2(),
                     coupling.get_value());
}

void Couplings::add_coupling(Index node_num_1, Index node_num_2, double value) {
    add_new_coupling(node_num_1, node_num_2, value);
}

void Couplings::add_coupling(const Coupling& coupling) {
    add_new_coupling(coupling);
}

double* Couplings::get_coupling_value_ref(Index node_num_1, Index node_num_2) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return nullptr;
    }

    return _matrices.get_conductor_value_ref_from_idx(indices->first,
                                                      indices->second);
}

IntAddress Couplings::get_coupling_value_address(Index node_num_1,
                                                 Index node_num_2) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return 0U;
    }

    return _matrices.get_conductor_value_address_from_idx(indices->first,
                                                          indices->second);
}

bool Couplings::coupling_exists(Index node_num_1, Index node_num_2) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return false;
    }

    return _matrices.coupling_exists_from_idxs(indices->first, indices->second);
}

Coupling Couplings::get_coupling_from_coupling_idx(Index cidx) {
    const auto [idx1, idx2, value] =
        _matrices.get_idxs_and_coupling_value_from_coupling_idx(cidx);
    if (idx1 < 0 || idx2 < 0) {
        return {Index{-1}, Index{-1}, std::numeric_limits<double>::quiet_NaN()};
    }

    const Index node_num_1 =
        _nodes != nullptr
            ? static_cast<Index>(_nodes->get_node_num_from_idx(idx1))
            : idx1;
    const Index node_num_2 =
        _nodes != nullptr
            ? static_cast<Index>(_nodes->get_node_num_from_idx(idx2))
            : idx2;
    return {node_num_1, node_num_2, value};
}

void Couplings::synchronize_structure() {
    if (_nodes == nullptr) {
        return;
    }

    const auto diff_count =
        static_cast<Index>(_nodes->_diff_node_num_vector.size());
    const auto bound_count =
        static_cast<Index>(_nodes->_bound_node_num_vector.size());

    ensure_sparse_dimensions(_matrices.sparse_dd, diff_count, diff_count);
    ensure_sparse_dimensions(_matrices.sparse_db, diff_count, bound_count);
    ensure_sparse_dimensions(_matrices.sparse_bb, bound_count, bound_count);
}

std::optional<std::pair<Index, Index>> Couplings::get_indices_from_node_numbers(
    Index node_num_1, Index node_num_2) {
    if (_nodes == nullptr) {
        if (VERBOSE) {
            std::cout << "Couplings::get_indices_from_node_numbers called with "
                         "null Nodes pointer."
                      << '\n';
        }
        return std::nullopt;
    }

    synchronize_structure();

    const auto node_num_1_int = to_int_node_number(node_num_1);
    const auto node_num_2_int = to_int_node_number(node_num_2);
    if (!node_num_1_int.has_value() || !node_num_2_int.has_value()) {
        return std::nullopt;
    }

    auto idx1 = _nodes->get_idx_from_node_num(*node_num_1_int);
    auto idx2 = _nodes->get_idx_from_node_num(*node_num_2_int);

    if (idx1 < 0 || idx2 < 0) {
        if (VERBOSE) {
            std::cout << "Couplings: Invalid node numbers " << node_num_1
                      << ", " << node_num_2 << "\n";
        }
        return std::nullopt;
    }

    if (idx1 == idx2) {
        if (VERBOSE) {
            std::cout << "Couplings: Node numbers correspond to the same node."
                      << '\n';
        }
        return std::nullopt;
    }

    if (idx1 > idx2) {
        std::swap(idx1, idx2);
    }

    return std::pair<Index, Index>{idx1, idx2};
}

std::tuple<Index, Index, Eigen::SparseMatrix<double, Eigen::RowMajor>*>
Couplings::get_indices_and_sparse_from_node_numbers(Index node_num_1,
                                                    Index node_num_2) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return {Index{-1}, Index{-1}, nullptr};
    }

    auto sp_idx1 = indices->first;
    auto sp_idx2 = indices->second;
    auto* sparse_ptr =
        static_cast<Eigen::SparseMatrix<double, Eigen::RowMajor>*>(nullptr);

    const auto num_diff_nodes = _matrices.sparse_dd.rows();

    if (sp_idx2 < num_diff_nodes) {
        sparse_ptr = &_matrices.sparse_dd;
    } else if (sp_idx1 < num_diff_nodes) {
        sparse_ptr = &_matrices.sparse_db;
        sp_idx2 -= num_diff_nodes;
    } else {
        sparse_ptr = &_matrices.sparse_bb;
        sp_idx1 -= num_diff_nodes;
        sp_idx2 -= num_diff_nodes;
    }

    return {sp_idx1, sp_idx2, sparse_ptr};
}

bool Couplings::is_coupling_trivial_zero_from_node_numbers(Index node_num_1,
                                                           Index node_num_2) {
    auto [idx1, idx2, sparse_ptr] =
        get_indices_and_sparse_from_node_numbers(node_num_1, node_num_2);
    if (sparse_ptr == nullptr) {
        if (VERBOSE) {
            std::cout << "Couplings: Invalid sparse pointer for nodes "
                      << node_num_1 << ", " << node_num_2 << '\n';
        }
        return false;
    }

    return sparse_utils::is_trivial_zero(*sparse_ptr, idx1, idx2);
}

}  // namespace pycanha
