// Face -> node aggregation (pure CPU, no Vulkan).

#include "pycanha-core/radiative/aggregate.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <span>
#include <stdexcept>
#include <vector>

#include "csr_assembly.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/radiative/results.hpp"

namespace pycanha::radiative {

using detail::check_sparse_capacity;
using detail::make_csr;
using gmm::NO_NODE;

namespace {

// Position of each face's node in the sorted unique node list; -1 for
// NO_NODE faces.
[[nodiscard]] std::vector<std::int64_t> slot_to_node_index(
    std::span<const NodeNum> node_numbers, const std::vector<NodeNum>& nodes) {
    std::vector<std::int64_t> mapping(node_numbers.size(), -1);
    for (std::size_t face = 0; face < node_numbers.size(); ++face) {
        if (node_numbers[face] == NO_NODE) {
            continue;
        }
        const auto it = std::ranges::lower_bound(nodes, node_numbers[face]);
        mapping[face] = std::distance(nodes.begin(), it);
    }
    return mapping;
}

void check_sizes(std::size_t faces, std::span<const NodeNum> node_numbers) {
    if (node_numbers.size() != faces) {
        throw std::invalid_argument(
            "pycanha::radiative: node_numbers size must match the face "
            "count");
    }
}

}  // namespace

std::vector<NodeNum> aggregate_nodes(std::span<const NodeNum> node_numbers) {
    std::vector<NodeNum> nodes(node_numbers.begin(), node_numbers.end());
    std::erase(nodes, NO_NODE);
    std::ranges::sort(nodes);
    const auto duplicates = std::ranges::unique(nodes);
    nodes.erase(duplicates.begin(), duplicates.end());
    return nodes;
}

namespace {

// Packs per-row ordered maps into a compressed matrix; node counts are small
// next to face counts, so the maps stay cheap and hand back sorted columns.
[[nodiscard]] SparseMatrix pack_node_rows(
    const std::vector<std::map<SparseIndex, double>>& rows,
    std::size_t node_cols) {
    std::vector<SparseIndex> row_starts;
    row_starts.reserve(rows.size() + 1);
    std::vector<SparseIndex> columns;
    std::vector<double> values;
    row_starts.push_back(0);
    for (const auto& row : rows) {
        for (const auto& [col, value] : row) {
            columns.push_back(col);
            values.push_back(value);
        }
        check_sparse_capacity(values.size());
        row_starts.push_back(static_cast<SparseIndex>(values.size()));
    }
    return make_csr(static_cast<Eigen::Index>(rows.size()),
                    static_cast<Eigen::Index>(node_cols), row_starts, columns,
                    values);
}

}  // namespace

AggregateResult aggregate_matrix(const SparseMatrix& face_matrix,
                                 std::span<const NodeNum> node_numbers) {
    const auto faces = static_cast<std::size_t>(face_matrix.rows());
    check_sizes(faces, node_numbers);
    if (face_matrix.cols() < face_matrix.rows()) {
        throw std::invalid_argument(
            "pycanha::radiative: the face matrix cannot have fewer columns "
            "than rows");
    }

    const std::vector<NodeNum> nodes = aggregate_nodes(node_numbers);
    const std::vector<std::int64_t> node_of =
        slot_to_node_index(node_numbers, nodes);

    AggregateResult result;
    std::vector<std::map<SparseIndex, double>> rows(nodes.size());
    for (std::size_t face_row = 0; face_row < faces; ++face_row) {
        const std::int64_t node_row = node_of[face_row];
        if (node_row < 0) {
            continue;
        }
        for (SparseMatrix::InnerIterator entry(
                 face_matrix, static_cast<Eigen::Index>(face_row));
             entry; ++entry) {
            const auto face_col = static_cast<std::size_t>(entry.col());
            // Bucket columns carry no node label of their own here; the
            // row/column overload is what maps them.
            if (face_col >= faces) {
                continue;
            }
            const std::int64_t node_col = node_of[face_col];
            if (node_col < 0) {
                continue;
            }
            if (node_col == node_row) {
                result.intra_node_total += entry.value();
                continue;
            }
            const auto lower =
                static_cast<SparseIndex>(std::min(node_row, node_col));
            const auto upper =
                static_cast<SparseIndex>(std::max(node_row, node_col));
            rows[static_cast<std::size_t>(lower)][upper] += entry.value();
        }
    }
    result.matrix = pack_node_rows(rows, nodes.size());
    return result;
}

AggregateResult aggregate_matrix(const SparseMatrix& face_matrix,
                                 std::span<const NodeNum> row_node_numbers,
                                 std::span<const NodeNum> col_node_numbers) {
    const auto faces = static_cast<std::size_t>(face_matrix.rows());
    check_sizes(faces, row_node_numbers);
    if (col_node_numbers.size() !=
        static_cast<std::size_t>(face_matrix.cols())) {
        throw std::invalid_argument(
            "pycanha::radiative: col_node_numbers size must match the "
            "matrix column count (real faces plus the virtual bucket "
            "columns)");
    }

    const std::vector<NodeNum> row_nodes = aggregate_nodes(row_node_numbers);
    const std::vector<NodeNum> col_nodes = aggregate_nodes(col_node_numbers);
    const std::vector<std::int64_t> row_node_of =
        slot_to_node_index(row_node_numbers, row_nodes);
    const std::vector<std::int64_t> col_node_of =
        slot_to_node_index(col_node_numbers, col_nodes);

    std::vector<std::map<SparseIndex, double>> rows(row_nodes.size());
    for (std::size_t face_row = 0; face_row < faces; ++face_row) {
        const std::int64_t node_row = row_node_of[face_row];
        if (node_row < 0) {
            continue;
        }
        for (SparseMatrix::InnerIterator entry(
                 face_matrix, static_cast<Eigen::Index>(face_row));
             entry; ++entry) {
            const auto face_col = static_cast<std::size_t>(entry.col());
            const std::int64_t node_col = col_node_of[face_col];
            if (node_col < 0) {
                continue;
            }
            rows[static_cast<std::size_t>(node_row)]
                [static_cast<SparseIndex>(node_col)] += entry.value();
        }
    }
    return AggregateResult{.matrix = pack_node_rows(rows, col_nodes.size()),
                           .intra_node_total = 0.0};
}

Eigen::VectorXd aggregate_flux(const Eigen::VectorXd& face_flux_w_m2,
                               std::span<const NodeNum> node_numbers,
                               std::span<const double> face_areas) {
    const auto faces = static_cast<std::size_t>(face_flux_w_m2.rows());
    check_sizes(faces, node_numbers);
    if (face_areas.size() != faces) {
        throw std::invalid_argument(
            "pycanha::radiative: face_areas size must match the face "
            "count");
    }

    const std::vector<NodeNum> nodes = aggregate_nodes(node_numbers);
    const std::vector<std::int64_t> node_of =
        slot_to_node_index(node_numbers, nodes);

    Eigen::VectorXd out =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nodes.size()));
    for (std::size_t face = 0; face < faces; ++face) {
        const std::int64_t node = node_of[face];
        if (node < 0) {
            continue;
        }
        out(static_cast<Eigen::Index>(node)) +=
            face_flux_w_m2(static_cast<Eigen::Index>(face)) * face_areas[face];
    }
    return out;
}

}  // namespace pycanha::radiative
