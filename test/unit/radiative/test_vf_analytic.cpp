#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "pycanha-core/radiative/sparse.hpp"

namespace rad = pycanha::radiative;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;

namespace {

// Closed-form view factor between identical, directly opposed parallel
// rectangles of size a x b at distance c (standard catalog formula).
[[nodiscard]] double parallel_plates_vf(double a, double b, double c) {
    const double x = a / c;
    const double y = b / c;
    const double x2 = x * x;
    const double y2 = y * y;
    const double term_log =
        0.5 * std::log((1.0 + x2) * (1.0 + y2) / (1.0 + x2 + y2));
    const double term_x =
        x * std::sqrt(1.0 + y2) * std::atan(x / std::sqrt(1.0 + y2));
    const double term_y =
        y * std::sqrt(1.0 + x2) * std::atan(y / std::sqrt(1.0 + x2));
    return 2.0 / (std::numbers::pi * x * y) *
           (term_log + term_x + term_y - (x * std::atan(x)) -
            (y * std::atan(y)));
}

// Two unit plates 1 apart, side 1 of each facing the other. Slot layout:
// plate A = slots 0/1, plate B = slots 2/3.
[[nodiscard]] std::unique_ptr<GeometryModel> make_parallel_plates() {
    auto model = std::make_unique<GeometryModel>("plates");
    model->add(std::make_shared<GeometryItem>(
        "plate_a", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{}));
    // Winding chosen so the side-1 normal points down (-z), toward plate A.
    model->add(std::make_shared<GeometryItem>(
        "plate_b", Rectangle({0.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 1.0}),
        ThermalMesh{}));
    return model;
}

[[nodiscard]] double csr_value(const rad::SparseF64& matrix, std::int64_t row,
                               std::int32_t col) {
    for (std::int64_t k = matrix.indptr(row); k < matrix.indptr(row + 1); ++k) {
        if (matrix.indices(k) == col) {
            return matrix.values(k);
        }
    }
    return 0.0;
}

// Bit-exact equality of two VF results (integer counting cells make this a
// hard guarantee, not a tolerance check).
void require_bit_identical(const rad::VfResult& result,
                           const rad::VfResult& reference) {
    REQUIRE(result.vf.nnz() == reference.vf.nnz());
    REQUIRE(result.vf.values.cwiseEqual(reference.vf.values).all());
    REQUIRE(result.vf.indices.cwiseEqual(reference.vf.indices).all());
    REQUIRE(result.row_sums.size() == reference.row_sums.size());
    REQUIRE(result.row_sums.cwiseEqual(reference.row_sums).all());
}

}  // namespace

TEST_CASE("radiative vf: parallel plates match the analytic value",
          "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const auto model = make_parallel_plates();
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              model->material_table());
    rad::VfAccumulator acc(scene);

    rad::TraceSettings settings;
    settings.rays_per_face = 20'000;
    settings.seed = 7;
    const std::vector<std::uint32_t> emitters{0};
    scene.accumulate_vf(acc, settings, emitters);
    const rad::VfResult result = acc.result();

    const double expected = parallel_plates_vf(1.0, 1.0, 1.0);
    const double vf = csr_value(result.vf, 0, 2);
    // 4-sigma tolerance of the binomial estimate at 20k rays.
    const double tolerance =
        4.0 * std::sqrt(expected * (1.0 - expected) / 20'000.0);
    REQUIRE(vf == Catch::Approx(expected).margin(tolerance));
    // Every first hit from plate A lands on plate B's facing side (slot 2)
    // or escapes to space: the row sum equals the plate-to-plate factor.
    REQUIRE(result.row_sums(0) == Catch::Approx(vf));
    REQUIRE(result.stats.rays_per_face == 20'000);
    REQUIRE(result.stats.total_rays == 20'000);
    REQUIRE(result.stats.max_stderr > 0.0);
}

TEST_CASE("radiative vf: same seed reproduces bit-identical results",
          "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const auto model = make_parallel_plates();
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              model->material_table());

    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;
    settings.seed = 42;

    rad::VfAccumulator first(scene);
    scene.accumulate_vf(first, settings);
    rad::VfAccumulator second(scene);
    scene.accumulate_vf(second, settings);

    require_bit_identical(second.result(), first.result());
}

TEST_CASE("radiative vf: identity instances match the monolithic scene",
          "[radiative][gpu][vf][scene]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const auto model = make_parallel_plates();
    rad::Device device = rad::Device::create();

    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;
    settings.seed = 3;
    const std::vector<std::uint32_t> emitters{0};

    // Same geometry once as one part, once split into two rigid parts with
    // identity placements: the counting must be bit-identical because the
    // RNG is keyed on (face slot, ray index, seed), not on scene structure.
    rad::RadiativeScene monolithic(device, model->mesh_parts(),
                                   model->material_table());
    const std::vector<std::string> split{"plate_b"};
    rad::RadiativeScene instanced(device, model->mesh_parts(split),
                                  model->material_table());

    rad::VfAccumulator acc_mono(monolithic);
    monolithic.accumulate_vf(acc_mono, settings, emitters);
    rad::VfAccumulator acc_inst(instanced);
    instanced.accumulate_vf(acc_inst, settings, emitters);

    require_bit_identical(acc_inst.result(), acc_mono.result());
}

TEST_CASE("radiative vf: tiled layout is bit-identical to dense",
          "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const auto model = make_parallel_plates();
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              model->material_table());

    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;
    settings.seed = 11;

    rad::VfAccumulator dense(scene);
    scene.accumulate_vf(dense, settings);
    const rad::VfResult reference = dense.result();

    // Several tile sizes, including 1 row and num_slots - 1 (a block split
    // that exercises the row_offset bookkeeping hardest).
    for (const std::uint32_t tile_rows : {1U, 2U, 3U}) {
        rad::VfAccumulator tiled(
            scene, rad::AccumConfig{.layout = rad::AccumLayout::Tiled,
                                    .tile_rows = tile_rows,
                                    .sparse_threshold = 0.0});
        scene.accumulate_vf(tiled, settings);
        require_bit_identical(tiled.result(), reference);
    }
}

TEST_CASE("radiative vf: batches accumulate", "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const auto model = make_parallel_plates();
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              model->material_table());
    rad::VfAccumulator acc(scene);

    rad::TraceSettings settings;
    settings.rays_per_face = 2'000;
    settings.seed = 1;
    scene.accumulate_vf(acc, settings);
    settings.seed = 2;  // a new batch must use a fresh seed
    scene.accumulate_vf(acc, settings);

    const rad::VfResult result = acc.result();
    REQUIRE(result.stats.rays_per_face == 4'000);
    REQUIRE(result.row_sums(0) > 0.0);
    REQUIRE(result.row_sums(0) <= 1.0);
}
