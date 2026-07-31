#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/aggregate.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/gebhart.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "pycanha-core/radiative/sparse.hpp"
#include "scene_fixtures.hpp"

namespace rad = pycanha::radiative;
using pycanha::NodeNum;
using pycanha::gmm::NO_NODE;
using radiative_fixtures::csr_value;
using radiative_fixtures::gray_row;
using radiative_fixtures::make_materials;
using radiative_fixtures::make_parallel_plates;

namespace {

// CSR from (row, col, value) triplets that are already row-major sorted.
[[nodiscard]] rad::SparseF64 make_csr(
    std::int64_t size,
    const std::vector<std::tuple<std::int64_t, std::int32_t, double>>&
        entries) {
    rad::SparseF64 out;
    out.rows = size;
    out.cols = size;
    out.indptr = Eigen::VectorX<std::int64_t>::Zero(size + 1);
    out.indices.resize(static_cast<Eigen::Index>(entries.size()));
    out.values.resize(static_cast<Eigen::Index>(entries.size()));
    for (std::size_t k = 0; k < entries.size(); ++k) {
        const auto& [row, col, value] = entries[k];
        out.indices(static_cast<Eigen::Index>(k)) = col;
        out.values(static_cast<Eigen::Index>(k)) = value;
        out.indptr(row + 1) += 1;
    }
    for (std::int64_t row = 0; row < size; ++row) {
        out.indptr(row + 1) += out.indptr(row);
    }
    return out;
}

// Two facing unit surfaces (slots 0 and 2) seeing each other with F = 0.4;
// slots 1 and 3 look into space.
[[nodiscard]] rad::SparseF64 two_surface_vf() {
    return make_csr(4, {{0, 2, 0.4}, {2, 0, 0.4}});
}

// In-place row scaling: GR aggregation needs A_i * eps_i * B_ij.
void scale_rows_by_emissivity(rad::SparseF64& matrix,
                              const Eigen::VectorXd& emissivity) {
    for (Eigen::Index row = 0; row < matrix.rows; ++row) {
        for (std::int64_t k = matrix.indptr(row); k < matrix.indptr(row + 1);
             ++k) {
            matrix.values(static_cast<Eigen::Index>(k)) *= emissivity(row);
        }
    }
}

// Entry-wise comparison of two square node matrices within 1e-12.
void require_matrices_match(const rad::SparseF64& result,
                            const rad::SparseF64& reference,
                            std::int64_t size) {
    for (std::int64_t row = 0; row < size; ++row) {
        for (std::int32_t col = 0; col < size; ++col) {
            REQUIRE(
                csr_value(result, row, col) ==
                Catch::Approx(csr_value(reference, row, col)).margin(1e-12));
        }
    }
}

}  // namespace

TEST_CASE("radiative gebhart: two-surface factors match the closed form",
          "[radiative][gebhart]") {
    const rad::SparseF64 vf = two_surface_vf();
    const Eigen::VectorXd emissivity = Eigen::VectorXd::Constant(4, 0.5);

    const rad::SparseF64 factors = rad::gebhart_factors(vf, emissivity);
    // B02 = eps * F / (1 - rho^2 F^2), B00 = eps * rho * F^2 / (same).
    REQUIRE(csr_value(factors, 0, 2) == Catch::Approx(0.2 / 0.96));
    REQUIRE(csr_value(factors, 0, 0) == Catch::Approx(0.04 / 0.96));
    REQUIRE(csr_value(factors, 2, 0) == Catch::Approx(0.2 / 0.96));

    // policy = 0 renormalizes each row to one (deficit treated as noise):
    // effectively F = 1 between the two surfaces.
    const rad::SparseF64 closed =
        rad::gebhart_factors(vf, emissivity, /*space_fraction_policy=*/0.0);
    REQUIRE(csr_value(closed, 0, 2) == Catch::Approx(0.5 / 0.75));
    REQUIRE(csr_value(closed, 0, 0) == Catch::Approx(0.25 / 0.75));
}

TEST_CASE("radiative gebhart: node factors equal the aggregated dense path",
          "[radiative][gebhart]") {
    const rad::SparseF64 vf = two_surface_vf();
    const Eigen::VectorXd emissivity = Eigen::VectorXd::Constant(4, 0.5);
    const std::array<NodeNum, 4> node_numbers{10, NO_NODE, 20, NO_NODE};
    const std::array<double, 4> areas{1.0, 1.0, 1.0, 1.0};

    const rad::SparseF64 node_gr =
        rad::gebhart_node_factors(vf, emissivity, node_numbers, areas);
    REQUIRE(node_gr.rows == 2);
    REQUIRE(node_gr.cols == 2);

    // Reference: dense face-level Gebhart, rows scaled by eps, aggregated
    // to nodes.
    rad::SparseF64 scaled = rad::gebhart_factors(vf, emissivity);
    scale_rows_by_emissivity(scaled, emissivity);
    const rad::SparseF64 reference =
        rad::aggregate_matrix(scaled, node_numbers, areas);

    require_matrices_match(node_gr, reference, 2);
    // Diffuse-gray GR is symmetric when the VF matrix is reciprocal.
    REQUIRE(csr_value(node_gr, 0, 1) ==
            Catch::Approx(csr_value(node_gr, 1, 0)).margin(1e-12));
    REQUIRE(csr_value(node_gr, 0, 1) == Catch::Approx(0.1 / 0.96));
}

TEST_CASE("radiative gebhart: the dense path guards its size",
          "[radiative][gebhart]") {
    const std::int64_t huge = 20'001;
    rad::SparseF64 vf;
    vf.rows = huge;
    vf.cols = huge;
    vf.indptr = Eigen::VectorX<std::int64_t>::Zero(huge + 1);
    const Eigen::VectorXd emissivity = Eigen::VectorXd::Constant(huge, 0.5);
    REQUIRE_THROWS_WITH(
        rad::gebhart_factors(vf, emissivity),
        Catch::Matchers::ContainsSubstring("gebhart_node_factors"));
}

TEST_CASE("radiative gebhart: invalid inputs are rejected",
          "[radiative][gebhart]") {
    const rad::SparseF64 vf = two_surface_vf();
    // Wrong emissivity size.
    REQUIRE_THROWS_AS(
        rad::gebhart_factors(vf, Eigen::VectorXd::Constant(3, 0.5)),
        std::invalid_argument);
    // Emissivity outside [0, 1].
    REQUIRE_THROWS_AS(
        rad::gebhart_factors(vf, Eigen::VectorXd::Constant(4, 1.5)),
        std::invalid_argument);
    // Policy outside [0, 1].
    REQUIRE_THROWS_AS(
        rad::gebhart_factors(vf, Eigen::VectorXd::Constant(4, 0.5), 2.0),
        std::invalid_argument);
}

TEST_CASE("radiative gebhart: matrix path agrees with the MCRT kernel",
          "[radiative][gpu][gebhart]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const double eps = 0.6;
    const auto model = make_parallel_plates(1.0);
    const std::array<std::array<float, 6>, 1> rows{
        gray_row(static_cast<float>(eps))};
    const std::array<int, 2> pair_rows{0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 20'000;
    settings.seed = 61;

    // Path A: MCRT exchange factors, traced directly with materials.
    rad::ExchangeAccumulator exchange_acc(scene, rad::Band::IR);
    scene.accumulate_exchange(exchange_acc, settings);
    const rad::ExchangeResult mcrt = exchange_acc.result();

    // Path B: geometric VF matrix + CPU Gebhart solve (D4 cross-check).
    rad::VfAccumulator vf_acc(scene);
    scene.accumulate_vf(vf_acc, settings);
    const rad::VfResult vf = vf_acc.result();
    const Eigen::VectorXd emissivity = Eigen::VectorXd::Constant(
        static_cast<Eigen::Index>(scene.num_face_slots()), eps);
    const rad::SparseF64 gebhart = rad::gebhart_factors(vf.vf, emissivity);

    REQUIRE(csr_value(mcrt.factors, 0, 2) ==
            Catch::Approx(csr_value(gebhart, 0, 2)).margin(0.02));
    REQUIRE(csr_value(mcrt.factors, 0, 0) ==
            Catch::Approx(csr_value(gebhart, 0, 0)).margin(0.02));
}
