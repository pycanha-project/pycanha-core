#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <stdexcept>
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
#include "scene_fixtures.hpp"

namespace rad = pycanha::radiative;
using pycanha::NodeNum;
using pycanha::gmm::NO_NODE;
using radiative_fixtures::csr_value;
using radiative_fixtures::gray_row;
using radiative_fixtures::make_materials;
using radiative_fixtures::make_parallel_plates;

namespace {

// Square row-major CSR from (row, col, value) triplets.
[[nodiscard]] rad::SparseMatrix make_csr(
    Eigen::Index size, const std::vector<Eigen::Triplet<double>>& entries) {
    rad::SparseMatrix out(size, size);
    out.setFromTriplets(entries.begin(), entries.end());
    return out;
}

// Two facing unit surfaces (faces 0 and 2) seeing each other with F = 0.4;
// faces 1 and 3 look into space. Unit areas make the stored extensive
// coupling numerically equal to the view factor.
constexpr std::array<double, 4> unit_areas{1.0, 1.0, 1.0, 1.0};

[[nodiscard]] rad::SparseMatrix two_surface_vf() {
    return make_csr(4, {{0, 2, 0.4}});
}

// In-place row scaling: GR aggregation needs A_i * eps_i * B_ij.
void scale_rows_by_emissivity(rad::SparseMatrix& matrix,
                              const Eigen::VectorXd& emissivity) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (rad::SparseMatrix::InnerIterator entry(matrix, row); entry;
             ++entry) {
            entry.valueRef() *= emissivity(row);
        }
    }
}

}  // namespace

TEST_CASE("radiative gebhart: two-surface factors match the closed form",
          "[radiative][gebhart]") {
    const rad::SparseMatrix vf = two_surface_vf();
    const Eigen::VectorXd emissivity = Eigen::VectorXd::Constant(4, 0.5);

    const rad::SparseMatrix factors =
        rad::gebhart_factors(vf, emissivity, unit_areas);
    // B02 = eps * F / (1 - rho^2 F^2), B00 = eps * rho * F^2 / (same).
    REQUIRE(csr_value(factors, 0, 2) == Catch::Approx(0.2 / 0.96));
    REQUIRE(csr_value(factors, 0, 0) == Catch::Approx(0.04 / 0.96));
    REQUIRE(csr_value(factors, 2, 0) == Catch::Approx(0.2 / 0.96));

    // policy = 0 renormalizes each row to one (deficit treated as noise):
    // effectively F = 1 between the two surfaces.
    const rad::SparseMatrix closed = rad::gebhart_factors(
        vf, emissivity, unit_areas, /*space_fraction_policy=*/0.0);
    REQUIRE(csr_value(closed, 0, 2) == Catch::Approx(0.5 / 0.75));
    REQUIRE(csr_value(closed, 0, 0) == Catch::Approx(0.25 / 0.75));
}

TEST_CASE("radiative gebhart: node factors equal the aggregated dense path",
          "[radiative][gebhart]") {
    const rad::SparseMatrix vf = two_surface_vf();
    const Eigen::VectorXd emissivity = Eigen::VectorXd::Constant(4, 0.5);
    const std::array<NodeNum, 4> node_numbers{10, NO_NODE, 20, NO_NODE};

    const rad::SparseMatrix node_gr =
        rad::gebhart_node_factors(vf, emissivity, node_numbers, unit_areas);
    REQUIRE(node_gr.rows() == 2);
    REQUIRE(node_gr.cols() == 2);

    // Reference: dense face-level Gebhart, rows scaled by A_i eps_i to make
    // them extensive, then condensed to nodes.
    rad::SparseMatrix scaled = rad::gebhart_factors(vf, emissivity, unit_areas);
    scale_rows_by_emissivity(scaled, emissivity);
    const rad::AggregateResult reference =
        rad::aggregate_matrix(scaled, node_numbers);

    // The node path keeps both triangles; the condensed reference folds them
    // into one, so compare the off-diagonal coupling through that.
    REQUIRE(csr_value(node_gr, 0, 1) + csr_value(node_gr, 1, 0) ==
            Catch::Approx(csr_value(reference.matrix, 0, 1)).margin(1e-12));
    // Diffuse-gray GR is symmetric when the VF matrix is reciprocal. Storing
    // one symmetric coupling per pair leaves only the round-off of the sparse
    // solve, so this holds three orders of magnitude tighter than the 1e-12
    // an independently-estimated pair of directions could support.
    REQUIRE(csr_value(node_gr, 0, 1) ==
            Catch::Approx(csr_value(node_gr, 1, 0)).margin(1e-15));
    REQUIRE(csr_value(node_gr, 0, 1) == Catch::Approx(0.1 / 0.96));
}

TEST_CASE("radiative gebhart: the dense path guards its size",
          "[radiative][gebhart]") {
    const Eigen::Index huge = 20'001;
    const rad::SparseMatrix vf(huge, huge);
    const Eigen::VectorXd emissivity = Eigen::VectorXd::Constant(huge, 0.5);
    const std::vector<double> areas(static_cast<std::size_t>(huge), 1.0);
    REQUIRE_THROWS_WITH(
        rad::gebhart_factors(vf, emissivity, areas),
        Catch::Matchers::ContainsSubstring("gebhart_node_factors"));
}

TEST_CASE("radiative gebhart: invalid inputs are rejected",
          "[radiative][gebhart]") {
    const rad::SparseMatrix vf = two_surface_vf();
    // Wrong emissivity size.
    REQUIRE_THROWS_AS(
        rad::gebhart_factors(vf, Eigen::VectorXd::Constant(3, 0.5), unit_areas),
        std::invalid_argument);
    // Emissivity outside [0, 1].
    REQUIRE_THROWS_AS(
        rad::gebhart_factors(vf, Eigen::VectorXd::Constant(4, 1.5), unit_areas),
        std::invalid_argument);
    // Policy outside [0, 1].
    REQUIRE_THROWS_AS(
        rad::gebhart_factors(vf, Eigen::VectorXd::Constant(4, 0.5), unit_areas,
                             2.0),
        std::invalid_argument);
    // A stored entry below the diagonal duplicates a coupling.
    REQUIRE_THROWS_AS(
        rad::gebhart_factors(make_csr(4, {{0, 2, 0.4}, {2, 0, 0.4}}),
                             Eigen::VectorXd::Constant(4, 0.5), unit_areas),
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
        static_cast<Eigen::Index>(scene.num_faces()), eps);
    const rad::SparseMatrix gebhart =
        rad::gebhart_factors(vf.vf, emissivity, scene.face_areas());

    // The MCRT matrix stores the extensive A_i eps_i B_ij, the Gebhart solve
    // the intensive B, so the emissive area comes back out before comparing.
    const double emissive = scene.face_areas()[0] * eps;
    REQUIRE(csr_value(mcrt.factors, 0, 2) / emissive ==
            Catch::Approx(csr_value(gebhart, 0, 2)).margin(0.02));
    REQUIRE(csr_value(mcrt.factors, 0, 0) / emissive ==
            Catch::Approx(csr_value(gebhart, 0, 0)).margin(0.02));
}
