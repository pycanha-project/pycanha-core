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
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "csr_assembly.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/radiative/aggregate.hpp"
#include "pycanha-core/radiative/results.hpp"

namespace pycanha::radiative {

using detail::check_sparse_capacity;
using detail::make_csr;
using gmm::NO_NODE;

namespace {

// The dense solve is O(n^3) time and O(n^2) memory; past this size the
// node-level adjoint path is the intended tool.
constexpr Eigen::Index max_dense_faces = 20'000;

void validate_gebhart_inputs(const SparseMatrix& vf,
                             const Eigen::VectorXd& emissivity,
                             std::span<const double> face_areas,
                             double space_fraction_policy) {
    // Traced VF results carry the virtual space/inactive/lost columns; a
    // hand-built plain square matrix is equally fine. Columns beyond the
    // faces never re-emit, so both shapes solve the same system.
    if (vf.cols() != vf.rows() &&
        vf.cols() != vf.rows() + num_virtual_columns) {
        throw std::invalid_argument(
            "pycanha::radiative: the VF matrix must be square or carry "
            "exactly the virtual bucket columns");
    }
    if (emissivity.rows() != vf.rows() ||
        face_areas.size() != static_cast<std::size_t>(vf.rows())) {
        throw std::invalid_argument(
            "pycanha::radiative: emissivity/face_areas size must match the "
            "VF matrix");
    }
    for (Eigen::Index face = 0; face < emissivity.rows(); ++face) {
        const double eps = emissivity(face);
        if (std::isnan(eps) || eps < 0.0 || eps > 1.0) {
            throw std::invalid_argument(
                "pycanha::radiative: emissivity of face " +
                std::to_string(face) + " is outside [0, 1]");
        }
    }
    if (std::isnan(space_fraction_policy) || space_fraction_policy < 0.0 ||
        space_fraction_policy > 1.0) {
        throw std::invalid_argument(
            "pycanha::radiative: space_fraction_policy must be in [0, 1]");
    }
}

// Both view factors of every stored coupling, as a square n x n matrix. The
// VF result keeps only the upper triangle of the symmetric extensive
// G_ij = A_i F_ij, so the two directions have to be expanded before anything
// can solve with them; the virtual bucket columns never re-emit and are
// dropped here.
[[nodiscard]] SparseMatrix expand_view_factors(
    const SparseMatrix& vf, std::span<const double> face_areas) {
    const Eigen::Index n = vf.rows();
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(vf.nonZeros()) * 2);
    for (Eigen::Index row = 0; row < n; ++row) {
        for (SparseMatrix::InnerIterator entry(vf, row); entry; ++entry) {
            const Eigen::Index col = entry.col();
            if (col >= n) {
                continue;
            }
            if (col < row) {
                throw std::invalid_argument(
                    "pycanha::radiative: the VF matrix must hold only its "
                    "upper triangle; entry (" +
                    std::to_string(row) + ", " + std::to_string(col) +
                    ") duplicates a coupling and would be counted twice");
            }
            const double coupling = entry.value();
            triplets.emplace_back(
                row, col, coupling / face_areas[static_cast<std::size_t>(row)]);
            if (col != row) {
                triplets.emplace_back(
                    col, row,
                    coupling / face_areas[static_cast<std::size_t>(col)]);
            }
        }
    }
    SparseMatrix expanded(n, n);
    expanded.setFromTriplets(triplets.begin(), triplets.end());
    return expanded;
}

// Per-row multipliers implementing the space policy: with policy p, each
// row is divided by row_sum + p * (1 - row_sum) — p = 1 keeps the matrix
// as-is (the deficit is a real view to space), p = 0 renormalizes rows to
// one (the deficit is Monte-Carlo noise on a closed enclosure). The input is
// the already-expanded square F, so every column counts.
[[nodiscard]] std::vector<double> row_scales(const SparseMatrix& expanded,
                                             double space_fraction_policy) {
    std::vector<double> scales(static_cast<std::size_t>(expanded.rows()), 1.0);
    for (Eigen::Index row = 0; row < expanded.rows(); ++row) {
        double row_sum = 0.0;
        for (SparseMatrix::InnerIterator entry(expanded, row); entry; ++entry) {
            row_sum += entry.value();
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
[[nodiscard]] SparseMatrix pack_dense(const Eigen::MatrixXd& dense) {
    std::vector<SparseIndex> row_starts;
    row_starts.reserve(static_cast<std::size_t>(dense.rows()) + 1);
    std::vector<SparseIndex> columns;
    std::vector<double> values;
    row_starts.push_back(0);
    for (Eigen::Index row = 0; row < dense.rows(); ++row) {
        for (Eigen::Index col = 0; col < dense.cols(); ++col) {
            const double value = dense(row, col);
            if (value != 0.0) {
                columns.push_back(static_cast<SparseIndex>(col));
                values.push_back(value);
            }
        }
        check_sparse_capacity(values.size());
        row_starts.push_back(static_cast<SparseIndex>(values.size()));
    }
    return make_csr(dense.rows(), dense.cols(), row_starts, columns, values);
}

}  // namespace

SparseMatrix gebhart_factors(const SparseMatrix& vf,
                             const Eigen::VectorXd& emissivity,
                             std::span<const double> face_areas,
                             double space_fraction_policy) {
    validate_gebhart_inputs(vf, emissivity, face_areas, space_fraction_policy);
    if (vf.rows() > max_dense_faces) {
        throw std::invalid_argument(
            "pycanha::radiative: gebhart_factors solves a dense " +
            std::to_string(vf.rows()) + "x" + std::to_string(vf.rows()) +
            " system, which is limited to " + std::to_string(max_dense_faces) +
            " faces; use gebhart_node_factors for large models");
    }
    const Eigen::Index n = vf.rows();
    const SparseMatrix expanded = expand_view_factors(vf, face_areas);
    const std::vector<double> scales =
        row_scales(expanded, space_fraction_policy);

    Eigen::MatrixXd f = Eigen::MatrixXd::Zero(n, n);
    for (Eigen::Index row = 0; row < n; ++row) {
        const double scale = scales[static_cast<std::size_t>(row)];
        for (SparseMatrix::InnerIterator entry(expanded, row); entry; ++entry) {
            f(row, entry.col()) = entry.value() * scale;
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

SparseMatrix gebhart_node_factors(const SparseMatrix& vf,
                                  const Eigen::VectorXd& emissivity,
                                  std::span<const NodeNum> node_numbers,
                                  std::span<const double> face_areas,
                                  double space_fraction_policy) {
    validate_gebhart_inputs(vf, emissivity, face_areas, space_fraction_policy);
    const Eigen::Index n = vf.rows();
    if (node_numbers.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument(
            "pycanha::radiative: node_numbers size must match the VF matrix");
    }

    const std::vector<NodeNum> nodes = aggregate_nodes(node_numbers);
    const auto num_nodes = static_cast<Eigen::Index>(nodes.size());
    if (num_nodes == 0) {
        return SparseMatrix{};
    }
    std::vector<Eigen::Index> node_of(static_cast<std::size_t>(n), -1);
    for (std::size_t face = 0; face < node_numbers.size(); ++face) {
        if (node_numbers[face] == NO_NODE) {
            continue;
        }
        const auto it = std::ranges::lower_bound(nodes, node_numbers[face]);
        node_of[face] = std::distance(nodes.begin(), it);
    }

    // Sparse system (I - F R) and the tall-skinny RHS F E V (columns =
    // nodes): one factorization + num_nodes solves instead of a dense
    // inverse — num_nodes stays small no matter how fine the mesh is.
    const SparseMatrix expanded = expand_view_factors(vf, face_areas);
    const std::vector<double> scales =
        row_scales(expanded, space_fraction_policy);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(expanded.nonZeros()) +
                     static_cast<std::size_t>(n));
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n, num_nodes);
    for (Eigen::Index row = 0; row < n; ++row) {
        triplets.emplace_back(row, row, 1.0);
        const double scale = scales[static_cast<std::size_t>(row)];
        for (SparseMatrix::InnerIterator entry(expanded, row); entry; ++entry) {
            const Eigen::Index col = entry.col();
            const double f_entry = entry.value() * scale;
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
    for (Eigen::Index face = 0; face < n; ++face) {
        const Eigen::Index node_row = node_of[static_cast<std::size_t>(face)];
        if (node_row < 0) {
            continue;
        }
        const double weight =
            face_areas[static_cast<std::size_t>(face)] * emissivity(face);
        gr.row(node_row) += weight * y.row(face);
    }
    return pack_dense(gr);
}

}  // namespace pycanha::radiative
