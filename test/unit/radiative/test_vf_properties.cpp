#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "scene_fixtures.hpp"

namespace rad = pycanha::radiative;
using radiative_fixtures::csr_value;
using radiative_fixtures::gray_row;
using radiative_fixtures::make_box_enclosure;
using radiative_fixtures::make_materials;
using radiative_fixtures::make_parallel_plates;

TEST_CASE("radiative vf: a closed enclosure sees no space",
          "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_box_enclosure();
    const std::array<std::array<float, 6>, 1> rows{gray_row(1.0F)};
    const std::array<int, 6> pair_rows{0, 0, 0, 0, 0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;
    settings.seed = 71;
    rad::VfAccumulator acc(scene);
    scene.accumulate_vf(acc, settings);
    const rad::VfResult result = acc.result();

    // Every inward face row is fully closed: the space column stays at (or
    // extremely near — corner rays offset by epsilon may slip through a
    // seam) zero, and rows still sum to exactly one by construction.
    const auto space_col = static_cast<Eigen::Index>(scene.num_face_slots());
    for (Eigen::Index row = 0; row < 12; row += 2) {
        REQUIRE(csr_value(result.vf, row, space_col) < 1e-3);
        REQUIRE(result.row_sums(row) == Catch::Approx(1.0).margin(1e-12));
    }
    // A parity/winding bug shows up here first.
    REQUIRE(result.stats.reciprocity_residual < 0.15);
}

TEST_CASE("radiative vf: an emitter subset reproduces the full run's rows",
          "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_box_enclosure();
    const std::array<std::array<float, 6>, 1> rows{gray_row(1.0F)};
    const std::array<int, 6> pair_rows{0, 0, 0, 0, 0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 2'000;
    settings.seed = 73;

    // Untriangulated: what is under test is the kernel's RNG keying, and
    // combining would fold in the other rows' estimates, which the subset
    // run deliberately does not have.
    const rad::AccumConfig config{
        .layout = rad::AccumLayout::Dense,
        .tile_rows = 0,
        .sparse_threshold = 0.0,
        .triangulation = {.mode = rad::TriangulationMode::None}};
    rad::VfAccumulator full(scene, config);
    scene.accumulate_vf(full, settings);
    const rad::VfResult full_result = full.result();

    // The RNG is keyed on (slot, ray, seed), so tracing only the floor's
    // rows reproduces them bit-identically; other rows must be absent.
    const std::vector<std::uint32_t> subset{0};
    rad::VfAccumulator partial(scene, config);
    scene.accumulate_vf(partial, settings, subset);
    const rad::VfResult subset_result = partial.result();

    const auto cols = static_cast<Eigen::Index>(scene.num_face_slots()) +
                      static_cast<Eigen::Index>(rad::num_virtual_columns);
    for (Eigen::Index col = 0; col < cols; ++col) {
        REQUIRE(csr_value(subset_result.vf, 0, col) ==
                csr_value(full_result.vf, 0, col));
    }
    // Only the traced row carries entries, so the whole matrix is row 0.
    REQUIRE(subset_result.vf.row(0).nonZeros() == subset_result.vf.nonZeros());
    REQUIRE(subset_result.stats.total_rays == 2'000);
}

TEST_CASE("radiative vf: normal emission fires straight along the normal",
          "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    const std::array<std::array<float, 6>, 1> rows{gray_row(1.0F)};
    const std::array<int, 2> pair_rows{0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 2'000;
    settings.seed = 79;
    settings.normal_emission = true;
    const std::vector<std::uint32_t> emitters{0};
    rad::VfAccumulator acc(scene);
    scene.accumulate_vf(acc, settings, emitters);
    const rad::VfResult result = acc.result();

    // Plate B sits directly above plate A: every normal-emitted ray hits
    // it, so the entry is exactly one and space exactly zero.
    REQUIRE(csr_value(result.vf, 0, 2) == 1.0);
    const auto space_col = static_cast<Eigen::Index>(scene.num_face_slots());
    REQUIRE(csr_value(result.vf, 0, space_col) == 0.0);
}

TEST_CASE("radiative vf: the sparse threshold prunes storage, not closure",
          "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
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
    settings.seed = 83;
    const std::vector<std::uint32_t> emitters{0};

    // At gap 1 the plate-to-plate factor is ~0.2 and space ~0.8: a 0.5
    // threshold keeps only the space entry.
    rad::VfAccumulator pruned(
        scene, rad::AccumConfig{.layout = rad::AccumLayout::Dense,
                                .tile_rows = 0,
                                .sparse_threshold = 0.5,
                                .triangulation = {}});
    scene.accumulate_vf(pruned, settings, emitters);
    const rad::VfResult result = pruned.result();

    REQUIRE(result.vf.nonZeros() == 1);
    REQUIRE(csr_value(result.vf, 0, 2) == 0.0);
    // Row statistics are computed before thresholding: closure is intact.
    REQUIRE(result.row_sums(0) == Catch::Approx(1.0).margin(1e-12));
}
