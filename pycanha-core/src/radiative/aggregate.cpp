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

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/radiative/sparse.hpp"

namespace pycanha::radiative {

using gmm::NO_NODE;

namespace {

// Position of each slot's node in the sorted unique node list; -1 for
// NO_NODE slots.
[[nodiscard]] std::vector<std::int64_t> slot_to_node_index(
    std::span<const NodeNum> node_numbers, const std::vector<NodeNum>& nodes) {
    std::vector<std::int64_t> mapping(node_numbers.size(), -1);
    for (std::size_t slot = 0; slot < node_numbers.size(); ++slot) {
        if (node_numbers[slot] == NO_NODE) {
            continue;
        }
        const auto it = std::ranges::lower_bound(nodes, node_numbers[slot]);
        mapping[slot] = std::distance(nodes.begin(), it);
    }
    return mapping;
}

void check_sizes(std::size_t slots, std::span<const NodeNum> node_numbers,
                 std::span<const double> face_areas) {
    if (node_numbers.size() != slots || face_areas.size() != slots) {
        throw std::invalid_argument(
            "pycanha::radiative: node_numbers/face_areas size must match "
            "the face-slot count");
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

SparseF64 aggregate_matrix(const SparseF64& face_matrix,
                           std::span<const NodeNum> node_numbers,
                           std::span<const double> face_areas) {
    const auto slots = static_cast<std::size_t>(face_matrix.rows);
    check_sizes(slots, node_numbers, face_areas);
    const auto cols = static_cast<std::size_t>(face_matrix.cols);
    if (cols < slots) {
        throw std::invalid_argument(
            "pycanha::radiative: the face matrix cannot have fewer columns "
            "than rows");
    }
    // Columns beyond the row labels (the virtual space/inactive/lost
    // buckets of the matrix results) are dropped here; the row/column
    // overload maps them explicitly.
    std::vector<NodeNum> col_nodes(node_numbers.begin(), node_numbers.end());
    col_nodes.resize(cols, NO_NODE);
    return aggregate_matrix(face_matrix, node_numbers, col_nodes, face_areas);
}

SparseF64 aggregate_matrix(const SparseF64& face_matrix,
                           std::span<const NodeNum> row_node_numbers,
                           std::span<const NodeNum> col_node_numbers,
                           std::span<const double> face_areas) {
    const auto slots = static_cast<std::size_t>(face_matrix.rows);
    check_sizes(slots, row_node_numbers, face_areas);
    if (col_node_numbers.size() != static_cast<std::size_t>(face_matrix.cols)) {
        throw std::invalid_argument(
            "pycanha::radiative: col_node_numbers size must match the "
            "matrix column count (real face slots plus the virtual bucket "
            "columns)");
    }

    const std::vector<NodeNum> row_nodes = aggregate_nodes(row_node_numbers);
    const std::vector<NodeNum> col_nodes = aggregate_nodes(col_node_numbers);
    const std::vector<std::int64_t> row_node_of =
        slot_to_node_index(row_node_numbers, row_nodes);
    const std::vector<std::int64_t> col_node_of =
        slot_to_node_index(col_node_numbers, col_nodes);

    // Accumulate into ordered per-row maps: node counts are small next to
    // face counts, so this stays cheap and yields sorted CSR columns.
    std::vector<std::map<std::int64_t, double>> rows(row_nodes.size());
    for (std::size_t face_row = 0; face_row < slots; ++face_row) {
        const std::int64_t node_row = row_node_of[face_row];
        if (node_row < 0) {
            continue;
        }
        const double weight = face_areas[face_row];
        for (std::int64_t k =
                 face_matrix.indptr(static_cast<Eigen::Index>(face_row));
             k < face_matrix.indptr(static_cast<Eigen::Index>(face_row) + 1);
             ++k) {
            const auto face_col = static_cast<std::size_t>(
                face_matrix.indices(static_cast<Eigen::Index>(k)));
            const std::int64_t node_col = col_node_of[face_col];
            if (node_col < 0) {
                continue;
            }
            rows[static_cast<std::size_t>(node_row)][node_col] +=
                weight * face_matrix.values(static_cast<Eigen::Index>(k));
        }
    }

    SparseF64 out;
    out.rows = static_cast<std::int64_t>(row_nodes.size());
    out.cols = static_cast<std::int64_t>(col_nodes.size());
    out.indptr.resize(static_cast<Eigen::Index>(row_nodes.size()) + 1);
    std::vector<std::int32_t> indices;
    std::vector<double> values;
    out.indptr(0) = 0;
    for (std::size_t row = 0; row < rows.size(); ++row) {
        for (const auto& [col, value] : rows[row]) {
            indices.push_back(static_cast<std::int32_t>(col));
            values.push_back(value);
        }
        out.indptr(static_cast<Eigen::Index>(row) + 1) =
            static_cast<std::int64_t>(values.size());
    }
    out.indices = Eigen::Map<const Eigen::VectorX<std::int32_t>>(
        indices.data(), static_cast<Eigen::Index>(indices.size()));
    out.values = Eigen::Map<const Eigen::VectorXd>(
        values.data(), static_cast<Eigen::Index>(values.size()));
    return out;
}

Eigen::VectorXd aggregate_flux(const Eigen::VectorXd& face_flux_w_m2,
                               std::span<const NodeNum> node_numbers,
                               std::span<const double> face_areas) {
    const auto slots = static_cast<std::size_t>(face_flux_w_m2.rows());
    check_sizes(slots, node_numbers, face_areas);

    const std::vector<NodeNum> nodes = aggregate_nodes(node_numbers);
    const std::vector<std::int64_t> node_of =
        slot_to_node_index(node_numbers, nodes);

    Eigen::VectorXd out =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nodes.size()));
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const std::int64_t node = node_of[slot];
        if (node < 0) {
            continue;
        }
        out(static_cast<Eigen::Index>(node)) +=
            face_flux_w_m2(static_cast<Eigen::Index>(slot)) * face_areas[slot];
    }
    return out;
}

}  // namespace pycanha::radiative
