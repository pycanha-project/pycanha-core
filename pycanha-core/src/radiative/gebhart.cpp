// CPU Gebhart services (pure CPU, no Vulkan): B = (I - F R)^-1 F E from a
// geometric VF matrix — the diffuse-gray fast path that re-derives exchange
// factors without re-tracing.

#include "pycanha-core/radiative/gebhart.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/radiative/aggregate.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/sparse.hpp"

namespace pycanha::radiative {

using gmm::NO_NODE;

namespace {

// The dense solve is O(n^3) time and O(n^2) memory; past this size the
// node-level adjoint path is the intended tool.
constexpr std::int64_t max_dense_slots = 20'000;

void validate_gebhart_inputs(const SparseF64& vf,
                             const Eigen::VectorXd& emissivity,
                             double space_fraction_policy) {
    // Traced VF results carry the virtual space/inactive/lost columns; a
    // hand-built plain square matrix is equally fine. Columns beyond the
    // face slots never re-emit, so both shapes solve the same system.
    if (vf.cols != vf.rows && vf.cols != vf.rows + num_virtual_columns) {
        throw std::invalid_argument(
            "pycanha::radiative: the VF matrix must be square or carry "
            "exactly the virtual bucket columns");
    }
    if (emissivity.rows() != vf.rows) {
        throw std::invalid_argument(
            "pycanha::radiative: emissivity size must match the VF matrix");
    }
    for (Eigen::Index slot = 0; slot < emissivity.rows(); ++slot) {
        const double eps = emissivity(slot);
        if (std::isnan(eps) || eps < 0.0 || eps > 1.0) {
            throw std::invalid_argument(
                "pycanha::radiative: emissivity of slot " +
                std::to_string(slot) + " is outside [0, 1]");
        }
    }
    if (std::isnan(space_fraction_policy) || space_fraction_policy < 0.0 ||
        space_fraction_policy > 1.0) {
        throw std::invalid_argument(
            "pycanha::radiative: space_fraction_policy must be in [0, 1]");
    }
}

// Per-row multipliers implementing the space policy: with policy p, each
// row is divided by row_sum + p * (1 - row_sum) — p = 1 keeps the matrix
// as-is (the deficit is a real view to space), p = 0 renormalizes rows to
// one (the deficit is Monte-Carlo noise on a closed enclosure). Only real
// face columns count toward row_sum: the virtual bucket columns ARE the
// deficit.
[[nodiscard]] std::vector<double> row_scales(const SparseF64& vf,
                                             double space_fraction_policy) {
    std::vector<double> scales(static_cast<std::size_t>(vf.rows), 1.0);
    for (Eigen::Index row = 0; row < vf.rows; ++row) {
        double row_sum = 0.0;
        for (std::int64_t k = vf.indptr(row); k < vf.indptr(row + 1); ++k) {
            const auto entry = static_cast<Eigen::Index>(k);
            if (vf.indices(entry) < vf.rows) {
                row_sum += vf.values(entry);
            }
        }
        const double denominator =
            row_sum + (space_fraction_policy * (1.0 - row_sum));
        if (denominator > 0.0) {
            scales[static_cast<std::size_t>(row)] = 1.0 / denominator;
        }
    }
    return scales;
}

// CSR from a dense row-major scan, keeping every nonzero entry.
[[nodiscard]] SparseF64 pack_dense(const Eigen::MatrixXd& dense) {
    SparseF64 out;
    out.rows = dense.rows();
    out.cols = dense.cols();
    out.indptr.resize(dense.rows() + 1);
    std::vector<std::int32_t> indices;
    std::vector<double> values;
    out.indptr(0) = 0;
    for (Eigen::Index row = 0; row < dense.rows(); ++row) {
        for (Eigen::Index col = 0; col < dense.cols(); ++col) {
            const double value = dense(row, col);
            if (value != 0.0) {
                indices.push_back(static_cast<std::int32_t>(col));
                values.push_back(value);
            }
        }
        out.indptr(row + 1) = static_cast<std::int64_t>(values.size());
    }
    out.indices = Eigen::Map<const Eigen::VectorX<std::int32_t>>(
        indices.data(), static_cast<Eigen::Index>(indices.size()));
    out.values = Eigen::Map<const Eigen::VectorXd>(
        values.data(), static_cast<Eigen::Index>(values.size()));
    return out;
}

}  // namespace

SparseF64 gebhart_factors(const SparseF64& vf,
                          const Eigen::VectorXd& emissivity,
                          double space_fraction_policy) {
    validate_gebhart_inputs(vf, emissivity, space_fraction_policy);
    if (vf.rows > max_dense_slots) {
        throw std::invalid_argument(
            "pycanha::radiative: gebhart_factors solves a dense " +
            std::to_string(vf.rows) + "x" + std::to_string(vf.rows) +
            " system, which is limited to " + std::to_string(max_dense_slots) +
            " face slots; use gebhart_node_factors for large models");
    }
    const auto n = static_cast<Eigen::Index>(vf.rows);
    const std::vector<double> scales = row_scales(vf, space_fraction_policy);

    Eigen::MatrixXd f = Eigen::MatrixXd::Zero(n, n);
    for (Eigen::Index row = 0; row < n; ++row) {
        const double scale = scales[static_cast<std::size_t>(row)];
        for (std::int64_t k = vf.indptr(row); k < vf.indptr(row + 1); ++k) {
            const auto entry = static_cast<Eigen::Index>(k);
            if (vf.indices(entry) < n) {  // bucket columns never re-emit
                f(row, vf.indices(entry)) = vf.values(entry) * scale;
            }
        }
    }

    // B = (I - F R)^-1 F E; space needs no explicit column — energy that
    // reaches it never returns, so it is exactly the row deficit of B.
    const Eigen::VectorXd reflectivity = Eigen::VectorXd::Ones(n) - emissivity;
    const Eigen::MatrixXd system =
        Eigen::MatrixXd::Identity(n, n) - f * reflectivity.asDiagonal();
    const Eigen::MatrixXd rhs = f * emissivity.asDiagonal();
    return pack_dense(system.partialPivLu().solve(rhs));
}

SparseF64 gebhart_node_factors(const SparseF64& vf,
                               const Eigen::VectorXd& emissivity,
                               std::span<const NodeNum> node_numbers,
                               std::span<const double> face_areas,
                               double space_fraction_policy) {
    validate_gebhart_inputs(vf, emissivity, space_fraction_policy);
    const auto n = static_cast<Eigen::Index>(vf.rows);
    if (node_numbers.size() != static_cast<std::size_t>(n) ||
        face_areas.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument(
            "pycanha::radiative: node_numbers/face_areas size must match "
            "the VF matrix");
    }

    const std::vector<NodeNum> nodes = aggregate_nodes(node_numbers);
    const auto num_nodes = static_cast<Eigen::Index>(nodes.size());
    if (num_nodes == 0) {
        SparseF64 empty;
        empty.indptr = Eigen::VectorX<std::int64_t>::Zero(1);
        return empty;
    }
    std::vector<Eigen::Index> node_of(static_cast<std::size_t>(n), -1);
    for (std::size_t slot = 0; slot < node_numbers.size(); ++slot) {
        if (node_numbers[slot] == NO_NODE) {
            continue;
        }
        const auto it = std::ranges::lower_bound(nodes, node_numbers[slot]);
        node_of[slot] = std::distance(nodes.begin(), it);
    }

    // Sparse system (I - F R) and the tall-skinny RHS F E V (columns =
    // nodes): one factorization + num_nodes solves instead of a dense
    // inverse — num_nodes stays small no matter how fine the mesh is.
    const std::vector<double> scales = row_scales(vf, space_fraction_policy);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(vf.nnz()) +
                     static_cast<std::size_t>(n));
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n, num_nodes);
    for (Eigen::Index row = 0; row < n; ++row) {
        triplets.emplace_back(row, row, 1.0);
        const double scale = scales[static_cast<std::size_t>(row)];
        for (std::int64_t k = vf.indptr(row); k < vf.indptr(row + 1); ++k) {
            const auto entry = static_cast<Eigen::Index>(k);
            const Eigen::Index col = vf.indices(entry);
            if (col >= n) {
                continue;  // bucket columns never re-emit
            }
            const double f_entry = vf.values(entry) * scale;
            triplets.emplace_back(row, col, -f_entry * (1.0 - emissivity(col)));
            const Eigen::Index node_col =
                node_of[static_cast<std::size_t>(col)];
            if (node_col >= 0) {
                rhs(row, node_col) += f_entry * emissivity(col);
            }
        }
    }
    Eigen::SparseMatrix<double> system(n, n);
    system.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(system);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "pycanha::radiative: the Gebhart system (I - F R) is singular — "
            "the VF matrix is not physically consistent");
    }
    const Eigen::MatrixXd y = solver.solve(rhs);  // = B V, faces x nodes

    // GR = W^T (B V) with W(i, m) = A_i * eps_i for faces of node m.
    Eigen::MatrixXd gr = Eigen::MatrixXd::Zero(num_nodes, num_nodes);
    for (Eigen::Index slot = 0; slot < n; ++slot) {
        const Eigen::Index node_row = node_of[static_cast<std::size_t>(slot)];
        if (node_row < 0) {
            continue;
        }
        const double weight =
            face_areas[static_cast<std::size_t>(slot)] * emissivity(slot);
        gr.row(node_row) += weight * y.row(slot);
    }
    return pack_dense(gr);
}

}  // namespace pycanha::radiative
