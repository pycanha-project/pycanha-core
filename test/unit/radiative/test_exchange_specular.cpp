#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
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
using radiative_fixtures::make_plates_with_sheet;

TEST_CASE("radiative exchange: a perfect mirror redirects without absorbing",
          "[radiative][gpu][exchange]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_mirror_bench();
    const std::array<std::array<float, 6>, 2> rows{
        gray_row(1.0F),  // source and catcher: black
        std::array<float, 6>{0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F}};  // mirror
    const std::array<int, 3> pair_rows{0, 1, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 20'000;
    settings.seed = 41;
    const std::vector<std::uint32_t> emitters{0};
    rad::ExchangeAccumulator acc(scene, rad::Band::IR);
    scene.accumulate_exchange(acc, settings, emitters);
    const rad::ExchangeResult result = acc.result();

    // Zero absorptivity deposits exactly zero at the mirror.
    REQUIRE(csr_value(result.factors, 0, 2) == 0.0);
    // The 45-degree mirror folds the upward hemisphere onto the catcher:
    // its share includes both the direct grazing view and the reflection.
    REQUIRE(csr_value(result.factors, 0, 4) > 0.4);
    REQUIRE(acc.conservation_error() == 0);
}

TEST_CASE("radiative exchange: a fully transparent sheet changes nothing",
          "[radiative][gpu][exchange]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    rad::Device device = rad::Device::create();
    rad::TraceSettings settings;
    settings.rays_per_face = 10'000;
    settings.seed = 47;
    const std::vector<std::uint32_t> emitters{0};

    const auto bare_model = make_parallel_plates(1.0);
    const std::array<std::array<float, 6>, 1> black{gray_row(1.0F)};
    const std::array<int, 2> bare_pairs{0, 0};
    rad::RadiativeScene bare(device, bare_model->mesh_parts(),
                             make_materials(black, bare_pairs));
    rad::ExchangeAccumulator bare_acc(bare, rad::Band::IR);
    bare.accumulate_exchange(bare_acc, settings, emitters);
    const double bare_factor = csr_value(bare_acc.result().factors, 0, 2);

    const auto sheet_model = make_plates_with_sheet(1.0);
    const std::array<std::array<float, 6>, 2> with_sheet_rows{
        gray_row(1.0F),
        std::array<float, 6>{0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F}};  // tau = 1
    const std::array<int, 3> sheet_pairs{0, 0, 1};
    rad::RadiativeScene sheeted(device, sheet_model->mesh_parts(),
                                make_materials(with_sheet_rows, sheet_pairs));
    rad::ExchangeAccumulator sheet_acc(sheeted, rad::Band::IR);
    sheeted.accumulate_exchange(sheet_acc, settings, emitters);
    const rad::ExchangeResult sheet_result = sheet_acc.result();

    // Thin-surface transmission keeps the direction, so plate B receives
    // the same share as in the bare scene. Not bit-identical: the
    // self-intersection restart point is offset along the sheet normal,
    // which displaces oblique rays laterally by ~epsilon and can flip the
    // odd edge-grazing ray — a same-seed comparison is tight regardless.
    REQUIRE(csr_value(sheet_result.factors, 0, 2) ==
            Catch::Approx(bare_factor).margin(0.005));
    // The sheet itself absorbs exactly nothing — not even a CSR entry.
    REQUIRE(csr_value(sheet_result.factors, 0, 4) == 0.0);
    REQUIRE(sheet_acc.conservation_error() == 0);
}

TEST_CASE("radiative exchange: a half-transparent sheet splits the energy",
          "[radiative][gpu][exchange]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    rad::Device device = rad::Device::create();
    rad::TraceSettings settings;
    settings.rays_per_face = 20'000;
    settings.seed = 53;
    const std::vector<std::uint32_t> emitters{0};

    const auto bare_model = make_parallel_plates(1.0);
    const std::array<std::array<float, 6>, 1> black{gray_row(1.0F)};
    const std::array<int, 2> bare_pairs{0, 0};
    rad::RadiativeScene bare(device, bare_model->mesh_parts(),
                             make_materials(black, bare_pairs));
    rad::ExchangeAccumulator bare_acc(bare, rad::Band::IR);
    bare.accumulate_exchange(bare_acc, settings, emitters);
    const rad::ExchangeResult bare_result = bare_acc.result();
    const double bare_factor = csr_value(bare_result.factors, 0, 2);

    const auto sheet_model = make_plates_with_sheet(1.0);
    const std::array<std::array<float, 6>, 2> with_sheet_rows{
        gray_row(1.0F),
        // tau = 0.5, no absorption: the rest reflects diffusely.
        std::array<float, 6>{0.0F, 0.0F, 0.5F, 0.0F, 0.0F, 0.5F}};
    const std::array<int, 3> sheet_pairs{0, 0, 1};
    rad::RadiativeScene sheeted(device, sheet_model->mesh_parts(),
                                make_materials(with_sheet_rows, sheet_pairs));
    rad::ExchangeAccumulator sheet_acc(sheeted, rad::Band::IR);
    sheeted.accumulate_exchange(sheet_acc, settings, emitters);
    const rad::ExchangeResult sheet_result = sheet_acc.result();

    // Half the rays pass straight through (same geometry beyond the sheet),
    // so plate B receives about half the bare-scene share.
    const double tolerance =
        (4.0 * (bare_result.stats.max_stderr + sheet_result.stats.max_stderr)) +
        0.01;
    REQUIRE(csr_value(sheet_result.factors, 0, 2) ==
            Catch::Approx(0.5 * bare_factor).margin(tolerance));
    REQUIRE(sheet_acc.conservation_error() == 0);
}
