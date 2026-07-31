// API walkthrough of pycanha::radiative — the compile-time contract of the
// public surface: geometry model -> parts + materials -> device -> scene ->
// accumulators -> results. Only shapes and invariants are asserted here;
// the physics lives in test/unit/radiative.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/radiative.hpp"

namespace rad = pycanha::radiative;

namespace {

// Two facing plates, meshed by the GMM.
[[nodiscard]] std::unique_ptr<pycanha::gmm::GeometryModel> make_demo_model() {
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    auto model = std::make_unique<pycanha::gmm::GeometryModel>("api_demo");
    model->add(std::make_shared<GeometryItem>(
        "plate_a", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{}));
    model->add(std::make_shared<GeometryItem>(
        "plate_b", Rectangle({0.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 1.0}),
        ThermalMesh{}));
    return model;
}

// Sizing mechanism for the (Python-side) memory policy: 4 slots plus the 3
// virtual columns, 8-byte cells.
void require_memory_estimates(const rad::RadiativeScene& scene) {
    const rad::MemoryEstimate estimate = rad::estimate_memory(scene);
    REQUIRE(estimate.gpu_bytes_per_tile_row == (4 + 3) * 8);
    REQUIRE(estimate.gpu_bytes_dense == 4 * estimate.gpu_bytes_per_tile_row);
    REQUIRE(estimate.host_bytes_block == estimate.gpu_bytes_dense);
    REQUIRE(estimate.gpu_bytes_scene > 0);
    const rad::MemoryEstimate tiled = rad::estimate_memory(
        scene, rad::AccumConfig{.layout = rad::AccumLayout::Tiled,
                                .tile_rows = 2,
                                .sparse_threshold = 0.0});
    REQUIRE(tiled.host_bytes_block == 2 * tiled.gpu_bytes_per_tile_row);
}

// Matrix shape: face-slot rows, face-slot + virtual bucket columns; the
// explicit space column closes every emitted row exactly.
void require_result_shape(const rad::VfResult& result) {
    REQUIRE(result.vf.rows == 4);
    REQUIRE(result.vf.cols == 4 + rad::num_virtual_columns);
    REQUIRE(result.stats.total_rays == 4'000);
    for (Eigen::Index row = 0; row < result.row_sums.size(); ++row) {
        REQUIRE(result.row_sums(row) == Catch::Approx(1.0).margin(1e-12));
    }
}

}  // namespace

TEST_CASE("api: radiative view factors end to end", "[api][radiative][gpu]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_demo_model();

    // The GMM entry points that feed the raytracer.
    std::vector<rad::ScenePart> parts = model->mesh_parts();
    rad::MaterialTable materials = model->material_table();

    // Device discovery is safe everywhere; construction is the one gate.
    rad::Device device = rad::Device::create();
    REQUIRE(device.info().ray_tracing);
    REQUIRE(device.memory_budget() > 0);

    // Build once, trace many.
    rad::RadiativeScene scene(device, std::move(parts), std::move(materials));
    REQUIRE(scene.num_face_slots() == 4);
    REQUIRE(scene.face_areas().size() == 4);
    require_memory_estimates(scene);

    // Batch-additive accumulation.
    rad::TraceSettings settings;
    settings.rays_per_face = 1'000;
    rad::VfAccumulator accumulator(scene);
    scene.accumulate_vf(accumulator, settings);
    require_result_shape(accumulator.result());
}
