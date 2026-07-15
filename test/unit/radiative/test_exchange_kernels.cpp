#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "scene_fixtures.hpp"

namespace rad = pycanha::radiative;
using radiative_fixtures::csr_value;
using radiative_fixtures::gray_row;
using radiative_fixtures::make_materials;
using radiative_fixtures::make_mirror_bench;
using radiative_fixtures::make_parallel_plates;

namespace {

// Accumulates the reference's two batches into a tiled accumulator and
// requires bit-identical factors plus an exact energy balance.
void require_tiled_matches(rad::RadiativeScene& scene,
                           rad::TraceSettings settings, std::uint32_t tile_rows,
                           const rad::ExchangeResult& reference) {
    rad::ExchangeAccumulator tiled(
        scene, rad::Band::IR,
        rad::AccumConfig{.layout = rad::AccumLayout::Tiled,
                         .tile_rows = tile_rows,
                         .sparse_threshold = 0.0});
    settings.seed = 1;
    scene.accumulate_exchange(tiled, settings);
    settings.seed = 2;
    scene.accumulate_exchange(tiled, settings);
    const rad::ExchangeResult result = tiled.result();
    REQUIRE(result.factors.nnz() == reference.factors.nnz());
    REQUIRE(result.factors.indices.cwiseEqual(reference.factors.indices).all());
    REQUIRE(result.factors.values.cwiseEqual(reference.factors.values).all());
    REQUIRE(tiled.conservation_error() == 0);
}

}  // namespace

TEST_CASE("radiative exchange: blackbody factors equal the view factors",
          "[radiative][gpu][exchange]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    const std::array<std::array<float, 6>, 1> rows{gray_row(1.0F)};
    const std::array<int, 2> pair_rows{0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 10'000;
    settings.seed = 5;

    rad::VfAccumulator vf_acc(scene);
    scene.accumulate_vf(vf_acc, settings);
    rad::ExchangeAccumulator ex_acc(scene, rad::Band::IR);
    scene.accumulate_exchange(ex_acc, settings);

    const rad::VfResult vf = vf_acc.result();
    const rad::ExchangeResult exchange = ex_acc.result();

    // With eps = 1 every ray deposits its full (power-of-two scaled) energy
    // at the first hit, so the exchange CSR is BIT-identical to the VF one:
    // same rays, same hits, and the u64 -> f64 conversion is exact.
    REQUIRE(exchange.band == rad::Band::IR);
    REQUIRE(exchange.factors.nnz() == vf.vf.nnz());
    REQUIRE(exchange.factors.indices.cwiseEqual(vf.vf.indices).all());
    REQUIRE(exchange.factors.values.cwiseEqual(vf.vf.values).all());
    REQUIRE(ex_acc.conservation_error() == 0);
}

TEST_CASE("radiative exchange: gray plates match the infinite-plate formula",
          "[radiative][gpu][exchange]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    // A tiny gap makes the finite plates a good stand-in for the infinite
    // pair, whose exchange is the classic 1 / (1/eps1 + 1/eps2 - 1).
    const double eps = 0.7;
    const auto model = make_parallel_plates(0.01);
    const std::array<std::array<float, 6>, 1> rows{
        gray_row(static_cast<float>(eps))};
    const std::array<int, 2> pair_rows{0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 20'000;
    settings.seed = 9;
    const std::vector<std::uint32_t> emitters{0};
    rad::ExchangeAccumulator acc(scene, rad::Band::IR);
    scene.accumulate_exchange(acc, settings, emitters);
    const rad::ExchangeResult result = acc.result();

    // GR per unit area = eps * B12 for these unit plates.
    const double expected = 1.0 / ((1.0 / eps) + (1.0 / eps) - 1.0);
    const double gr = eps * csr_value(result.factors, 0, 2);
    REQUIRE(gr == Catch::Approx(expected).margin(0.02));
    REQUIRE(acc.conservation_error() == 0);
}

TEST_CASE(
    "radiative exchange: energy conservation is exact and layouts are "
    "bit-identical",
    "[radiative][gpu][exchange][tiled]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    const std::array<std::array<float, 6>, 1> rows{gray_row(0.5F)};
    const std::array<int, 2> pair_rows{0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;

    rad::ExchangeAccumulator dense(scene, rad::Band::IR);
    settings.seed = 1;
    scene.accumulate_exchange(dense, settings);
    settings.seed = 2;  // a second batch must keep the balance exact too
    scene.accumulate_exchange(dense, settings);
    REQUIRE(dense.conservation_error() == 0);
    const rad::ExchangeResult reference = dense.result();
    REQUIRE(reference.stats.rays_per_face == 10'000);

    for (const std::uint32_t tile_rows : {1U, 2U, 3U}) {
        require_tiled_matches(scene, settings, tile_rows, reference);
    }
}

TEST_CASE("radiative exchange: the solar band scores every bounce",
          "[radiative][gpu][exchange]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    // The mirror is black in IR but a perfect specular reflector in the
    // solar band: solar energy reaches the catcher via the mirror (a
    // second-bounce path the legacy first-hit-only kernel missed), IR
    // energy is absorbed at the mirror.
    const auto model = make_mirror_bench();
    const std::array<std::array<float, 6>, 2> rows{
        gray_row(1.0F),  // source and catcher: black in both bands
        std::array<float, 6>{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F}};
    const std::array<int, 3> pair_rows{0, 1, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 20'000;
    settings.seed = 21;
    const std::vector<std::uint32_t> emitters{0};

    rad::ExchangeAccumulator solar_acc(scene, rad::Band::Solar);
    scene.accumulate_exchange(solar_acc, settings, emitters);
    const rad::ExchangeResult solar = solar_acc.result();

    rad::ExchangeAccumulator ir_acc(scene, rad::Band::IR);
    scene.accumulate_exchange(ir_acc, settings, emitters);
    const rad::ExchangeResult ir = ir_acc.result();

    // A perfect mirror absorbs exactly nothing (the fixed-point deposit of
    // a zero-absorption hit is exactly zero).
    REQUIRE(csr_value(solar.factors, 0, 2) == 0.0);
    // In IR the same geometry deposits on the mirror instead.
    REQUIRE(csr_value(ir.factors, 0, 2) > 0.3);
    // The solar catcher share exceeds the IR (direct-only) share by the
    // energy that arrived via the mirror — the multi-bounce regression.
    const double solar_catcher = csr_value(solar.factors, 0, 4);
    const double ir_catcher = csr_value(ir.factors, 0, 4);
    REQUIRE(solar_catcher > ir_catcher + 0.2);
    REQUIRE(solar_acc.conservation_error() == 0);
    REQUIRE(ir_acc.conservation_error() == 0);
}

TEST_CASE("radiative exchange: Russian roulette is threshold-invariant",
          "[radiative][gpu][exchange]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    // Highly reflective narrow cavity: paths are many bounces deep, so the
    // threshold decides how much Russian roulette actually runs.
    const auto model = make_parallel_plates(0.05);
    const std::array<std::array<float, 6>, 1> rows{gray_row(0.2F)};
    const std::array<int, 2> pair_rows{0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 10'000;
    settings.seed = 33;
    const std::vector<std::uint32_t> emitters{0};

    settings.energy_threshold = 1e-2F;
    rad::ExchangeAccumulator coarse(scene, rad::Band::IR);
    scene.accumulate_exchange(coarse, settings, emitters);
    const rad::ExchangeResult coarse_result = coarse.result();

    settings.energy_threshold = 1e-5F;
    rad::ExchangeAccumulator fine(scene, rad::Band::IR);
    scene.accumulate_exchange(fine, settings, emitters);
    const rad::ExchangeResult fine_result = fine.result();

    // Unbiasedness: the estimates agree within combined statistics.
    const double tolerance = (4.0 * (coarse_result.stats.max_stderr +
                                     fine_result.stats.max_stderr)) +
                             0.005;
    REQUIRE(
        csr_value(coarse_result.factors, 0, 2) ==
        Catch::Approx(csr_value(fine_result.factors, 0, 2)).margin(tolerance));
    REQUIRE(coarse.conservation_error() == 0);
    REQUIRE(fine.conservation_error() == 0);
}

TEST_CASE("radiative exchange: inactive faces absorb into the lost bucket",
          "[radiative][gpu][exchange]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    const std::array<std::array<float, 6>, 1> rows{gray_row(1.0F)};
    const std::array<int, 2> pair_rows{0, 0};
    rad::MaterialTable materials = make_materials(rows, pair_rows);
    materials.face_active(2) = false;  // plate B's facing side
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(), materials);

    rad::TraceSettings settings;
    settings.rays_per_face = 10'000;
    settings.seed = 13;
    const std::vector<std::uint32_t> emitters{0};
    rad::ExchangeAccumulator acc(scene, rad::Band::IR);
    scene.accumulate_exchange(acc, settings, emitters);
    const rad::ExchangeResult result = acc.result();

    // Everything that would have been absorbed at slot 2 is lost instead;
    // conservation stays exact because lost is a real bucket.
    REQUIRE(csr_value(result.factors, 0, 2) == 0.0);
    // The plate-to-plate view factor at gap 1 is about 0.2.
    REQUIRE(result.stats.lost_energy_fraction > 0.1);
    REQUIRE(result.stats.lost_energy_fraction < 0.3);
    REQUIRE(acc.conservation_error() == 0);
}

TEST_CASE("radiative exchange: update_materials swaps properties in place",
          "[radiative][gpu][exchange][scene]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    const std::array<std::array<float, 6>, 1> black{gray_row(1.0F)};
    const std::array<int, 2> pair_rows{0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(black, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 10'000;
    settings.seed = 17;
    const std::vector<std::uint32_t> emitters{0};

    rad::VfAccumulator vf_before(scene);
    scene.accumulate_vf(vf_before, settings, emitters);
    rad::ExchangeAccumulator black_acc(scene, rad::Band::IR);
    scene.accumulate_exchange(black_acc, settings, emitters);
    const double black_factor = csr_value(black_acc.result().factors, 0, 2);

    const std::array<std::array<float, 6>, 1> gray{gray_row(0.5F)};
    scene.update_materials(make_materials(gray, pair_rows));

    rad::ExchangeAccumulator gray_acc(scene, rad::Band::IR);
    scene.accumulate_exchange(gray_acc, settings, emitters);
    const double gray_factor = csr_value(gray_acc.result().factors, 0, 2);
    // Half the absorptivity halves the first-hit deposit; the multi-bounce
    // correction at this view factor is far below the margin.
    REQUIRE(gray_factor == Catch::Approx(0.5 * black_factor).margin(0.02));

    // The vf kernel never reads materials: bit-identical before and after.
    rad::VfAccumulator vf_after(scene);
    scene.accumulate_vf(vf_after, settings, emitters);
    const rad::VfResult before = vf_before.result();
    const rad::VfResult after = vf_after.result();
    REQUIRE(after.vf.values.cwiseEqual(before.vf.values).all());

    // Changing the mapping (or the row count) needs a scene rebuild.
    const std::array<std::array<float, 6>, 2> two_rows{gray_row(0.5F),
                                                       gray_row(0.9F)};
    const std::array<int, 2> remapped{0, 1};
    REQUIRE_THROWS_AS(
        scene.update_materials(make_materials(two_rows, remapped)),
        std::invalid_argument);
}
