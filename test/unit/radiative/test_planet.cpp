#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/scene_part.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "scene_fixtures.hpp"

namespace rad = pycanha::radiative;
using radiative_fixtures::csr_value;
using radiative_fixtures::gray_row;
using radiative_fixtures::make_materials;

namespace {

// A unit plate (faces 0/1, side 1 up) under a full sphere of radius 1
// centered 3 above it, grouped as "planet" so mesh_parts() splits it into
// its own rigid part.
[[nodiscard]] std::unique_ptr<pycanha::gmm::GeometryModel>
make_plate_under_planet() {
    using pycanha::gmm::GeometryGroup;
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::Sphere;
    using pycanha::gmm::ThermalMesh;
    auto model = std::make_unique<pycanha::gmm::GeometryModel>("orbit");
    model->add(std::make_shared<GeometryItem>(
        "plate", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{}));
    model->add(std::make_shared<GeometryGroup>("planet"));
    model->add(std::make_shared<GeometryItem>(
                   "planet_sphere",
                   Sphere({0.5, 0.5, 3.0}, {0.5, 0.5, 4.0}, {1.5, 0.5, 3.0},
                          1.0, -1.0, 1.0, 0.0, 2.0 * std::numbers::pi),
                   ThermalMesh{}),
               "planet");
    return model;
}

}  // namespace

TEST_CASE("radiative planet: a celestial part blocks, scores and never emits",
          "[radiative][gpu][scene]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_plate_under_planet();
    const std::vector<std::string> split{"planet"};
    std::vector<rad::ScenePart> parts = model->mesh_parts(split);
    REQUIRE(parts.size() == 2);
    // mesh_parts marks split groups Articulated; a planet is the caller's
    // decision.
    parts[1].kind = rad::PartKind::CelestialBody;
    const auto planet_first_face =
        static_cast<std::int32_t>(parts[0].mesh.nf());

    const auto num_pairs = static_cast<std::size_t>(model->mesh().nf()) / 2;
    const std::array<std::array<float, 6>, 1> rows{gray_row(0.5F)};
    const std::vector<int> pair_rows(num_pairs, 0);
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, std::move(parts),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 10'000;
    settings.seed = 91;

    // Planet faces never emit: the default emitter list is only the
    // plate's two sides.
    rad::ExchangeAccumulator acc(scene, rad::Band::IR);
    scene.accumulate_exchange(acc, settings);
    const rad::ExchangeResult result = acc.result();
    REQUIRE(result.stats.total_rays == 2 * settings.rays_per_face);

    // The planet absorbs EVERYTHING it intercepts (its gray material is
    // ignored), so the plate row deposits full ray energy on planet
    // columns; the sphere at distance 3 subtends ~sin^2(asin(1/3)) of the
    // upward hemisphere.
    double to_planet_extensive = 0.0;
    const auto faces = static_cast<std::int32_t>(scene.num_faces());
    for (std::int32_t col = planet_first_face; col < faces; ++col) {
        to_planet_extensive += csr_value(result.factors, 0, col);
    }
    // The stored value is the extensive A_i eps_i B_ij; dividing the
    // emissive area back out recovers the intensive share.
    const double to_planet =
        to_planet_extensive / (scene.face_areas()[0] * 0.5);
    REQUIRE(to_planet == Catch::Approx(1.0 / 9.0).margin(0.02));
    REQUIRE(acc.conservation_error() == 0);

    // The vf kernel sees the same geometry (planet faces are ordinary
    // first-hit targets for view factors). Untriangulated and divided by the
    // plate area, its stored entries are the same intensive estimate the
    // exchange factors above are.
    rad::VfAccumulator vf_acc(
        scene, rad::AccumConfig{
                   .layout = rad::AccumLayout::Dense,
                   .tile_rows = 0,
                   .sparse_threshold = 0.0,
                   .triangulation = {.mode = rad::TriangulationMode::None}});
    scene.accumulate_vf(vf_acc, settings);
    const rad::VfResult vf = vf_acc.result();
    double vf_to_planet = 0.0;
    for (std::int32_t col = planet_first_face; col < faces; ++col) {
        vf_to_planet += csr_value(vf.vf, 0, col);
    }
    REQUIRE(vf_to_planet / scene.face_areas()[0] ==
            Catch::Approx(to_planet).margin(0.01));
}
