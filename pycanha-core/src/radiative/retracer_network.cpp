#include "retracer_network.hpp"

#include <Eigen/Sparse>
#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"

namespace pycanha::radiative::detail {

namespace {

using Triplet = Eigen::Triplet<double, SparseIndex>;

void add_pair(std::vector<Triplet>& triplets, Index first, Index second,
              double value) {
    triplets.emplace_back(static_cast<SparseIndex>(std::min(first, second)),
                          static_cast<SparseIndex>(std::max(first, second)),
                          value);
}

// One entry of a virtual bucket column of the faces of `node`: the space
// column becomes a coupling to the space node (when there is one), and
// every bucket is totalled.
void add_bucket(Index bucket, Index node, Index space_index, double value,
                NodeCouplings& result, std::vector<Triplet>& triplets) {
    if (bucket == space_column_offset) {
        result.space_total += value;
        if (space_index >= 0 && space_index != node) {
            add_pair(triplets, node, space_index, value);
        }
    } else if (bucket == inactive_column_offset) {
        result.inactive_total += value;
    } else {
        result.lost_total += value;
    }
}

// The stored columns and values of a compressed row-major matrix.
struct RowStorage {
    std::span<const SparseIndex> outer;
    std::span<const SparseIndex> inner;
    std::span<double> values;
};

RowStorage storage_of(SparseMatrix& matrix) {
    const auto stored = static_cast<std::size_t>(matrix.nonZeros());
    return {
        .outer = std::span<const SparseIndex>(
            matrix.outerIndexPtr(),
            static_cast<std::size_t>(matrix.outerSize()) + 1U),
        .inner = std::span<const SparseIndex>(matrix.innerIndexPtr(), stored),
        .values = std::span<double>(matrix.valuePtr(), stored)};
}

// Writes the trace entries of one row, `cols` (block columns are `cols` minus
// `col_offset`) and `values`, into the slots of row `row` of `block`. Slots
// without an entry get 0; entries without a slot are appended to `overflow`
// at their global position.
void merge_row(std::span<const SparseIndex> cols,
               std::span<const double> values, SparseIndex col_offset,
               const RowStorage& block, Index row, Index global_row,
               std::vector<Triplet>& overflow, double& overflow_total) {
    const auto first =
        static_cast<std::size_t>(block.outer[static_cast<std::size_t>(row)]);
    const auto last = static_cast<std::size_t>(
        block.outer[static_cast<std::size_t>(row) + 1U]);
    const std::span<const SparseIndex> slot_cols =
        block.inner.subspan(first, last - first);
    const std::span<double> slots = block.values.subspan(first, last - first);

    std::size_t entry = 0;
    std::size_t slot = 0;
    while (entry < cols.size() || slot < slot_cols.size()) {
        const bool entry_left = entry < cols.size();
        const bool slot_left = slot < slot_cols.size();
        const SparseIndex entry_col =
            entry_left ? cols[entry] - col_offset : SparseIndex{0};
        if (slot_left && (!entry_left || slot_cols[slot] < entry_col)) {
            slots[slot] = 0.0;
            ++slot;
        } else if (slot_left && slot_cols[slot] == entry_col) {
            slots[slot] = values[entry];
            ++slot;
            ++entry;
        } else {
            overflow.emplace_back(static_cast<SparseIndex>(global_row),
                                  cols[entry], values[entry]);
            overflow_total += values[entry];
            ++entry;
        }
    }
}

}  // namespace

NodeCouplings reduce_to_nodes(const SparseMatrix& factors,
                              std::span<const Index> face_node,
                              Index space_index, Index num_nodes) {
    const Index num_faces = factors.rows();
    if (std::cmp_not_equal(face_node.size(), num_faces) ||
        factors.cols() != num_faces + num_virtual_columns) {
        throw std::invalid_argument(
            "reduce_to_nodes: the face node map does not match the factors");
    }
    NodeCouplings result;
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(factors.nonZeros()));
    for (Index face = 0; face < num_faces; ++face) {
        const Index node = face_node[static_cast<std::size_t>(face)];
        if (node < 0) {
            continue;
        }
        for (SparseMatrix::InnerIterator it(factors, face); it; ++it) {
            const Index col = it.col();
            const double value = it.value();
            if (col >= num_faces) {
                add_bucket(col - num_faces, node, space_index, value, result,
                           triplets);
                continue;
            }
            const Index other = face_node[static_cast<std::size_t>(col)];
            if (other < 0) {
                continue;
            }
            if (other == node) {
                result.intra_node_total += value;
            } else {
                add_pair(triplets, node, other, value);
            }
        }
    }
    result.couplings.resize(num_nodes, num_nodes);
    result.couplings.setFromTriplets(triplets.begin(), triplets.end());
    result.couplings.makeCompressed();
    return result;
}

SparseMatrix pattern_union(std::span<const SparseMatrix* const> traces,
                           Index num_nodes) {
    std::vector<Triplet> triplets;
    for (const SparseMatrix* trace : traces) {
        for (Index row = 0; row < trace->outerSize(); ++row) {
            for (SparseMatrix::InnerIterator it(*trace, row); it; ++it) {
                triplets.emplace_back(static_cast<SparseIndex>(row),
                                      static_cast<SparseIndex>(it.col()), 0.0);
            }
        }
    }
    SparseMatrix pattern(num_nodes, num_nodes);
    pattern.setFromTriplets(triplets.begin(), triplets.end());
    pattern.makeCompressed();
    return pattern;
}

SplitResult split_into_slots(const SparseMatrix& trace,
                             CouplingMatrices& matrices) {
    SparseMatrix& dd = matrices.sparse_dd;
    SparseMatrix& db = matrices.sparse_db;
    SparseMatrix& bb = matrices.sparse_bb;
    const Index nd = dd.rows();
    const Index num_nodes = nd + db.cols();
    if (trace.rows() != num_nodes || !trace.isCompressed() ||
        !dd.isCompressed() || !db.isCompressed() || !bb.isCompressed()) {
        throw std::invalid_argument(
            "split_into_slots: the trace does not match the compressed "
            "coupling blocks");
    }

    SplitResult result;
    std::vector<Triplet> overflow;
    const RowStorage dd_rows = storage_of(dd);
    const RowStorage db_rows = storage_of(db);
    const RowStorage bb_rows = storage_of(bb);
    const auto stored = static_cast<std::size_t>(trace.nonZeros());
    const std::span<const SparseIndex> outer(
        trace.outerIndexPtr(), static_cast<std::size_t>(num_nodes) + 1U);
    const std::span<const SparseIndex> inner(trace.innerIndexPtr(), stored);
    const std::span<const double> values(trace.valuePtr(), stored);
    const auto boundary_offset = static_cast<SparseIndex>(nd);
    for (Index row = 0; row < num_nodes; ++row) {
        const auto begin =
            static_cast<std::size_t>(outer[static_cast<std::size_t>(row)]);
        const auto end =
            static_cast<std::size_t>(outer[static_cast<std::size_t>(row) + 1U]);
        const std::span<const SparseIndex> cols =
            inner.subspan(begin, end - begin);
        const std::span<const double> row_values =
            values.subspan(begin, end - begin);
        for (const double value : row_values) {
            result.trace_total += value;
        }
        if (row < nd) {
            // Columns below nd are diffusive (the dd block), the rest
            // boundary (the db block); both runs are sorted.
            const auto split = static_cast<std::size_t>(
                std::ranges::lower_bound(cols, boundary_offset) - cols.begin());
            merge_row(cols.first(split), row_values.first(split), 0, dd_rows,
                      row, row, overflow, result.overflow_total);
            merge_row(cols.subspan(split), row_values.subspan(split),
                      boundary_offset, db_rows, row, row, overflow,
                      result.overflow_total);
        } else {
            merge_row(cols, row_values, boundary_offset, bb_rows, row - nd, row,
                      overflow, result.overflow_total);
        }
    }
    if (!overflow.empty()) {
        result.overflow.resize(num_nodes, num_nodes);
        result.overflow.setFromTriplets(overflow.begin(), overflow.end());
        result.overflow.makeCompressed();
    }
    return result;
}

void overflow_load(const SparseMatrix& overflow, std::span<const double> t,
                   std::vector<double>& load) {
    load.assign(t.size(), 0.0);
    for (Index row = 0; row < overflow.outerSize(); ++row) {
        const double t_row = t[static_cast<std::size_t>(row)];
        const double t4_row = t_row * t_row * t_row * t_row;
        for (SparseMatrix::InnerIterator it(overflow, row); it; ++it) {
            const auto col = static_cast<std::size_t>(it.col());
            const double t_col = t[col];
            const double flow = STF_BOLTZ * it.value() *
                                ((t_col * t_col * t_col * t_col) - t4_row);
            load[static_cast<std::size_t>(row)] += flow;
            load[col] -= flow;
        }
    }
}

}  // namespace pycanha::radiative::detail
