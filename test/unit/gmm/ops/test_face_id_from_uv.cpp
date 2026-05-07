#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <numbers>
#include <stdexcept>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/ops/face_id_from_uv.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"

namespace {

using pycanha::Point3D;
using pycanha::gmm::Cube;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Side;
using pycanha::gmm::ThermalMesh;
namespace gmm_ops = pycanha::gmm::ops;

}  // namespace

TEST_CASE("face_id_from_uv maps uv fractions to thermal cells", "[gmm][ops]") {
    const ThermalMesh mesh({0.0, 0.5, 1.0}, {0.0, 0.5, 1.0});
    const Rectangle rectangle({0.0, 0.0, 0.0}, {4.0, 0.0, 0.0},
                              {0.0, 2.0, 0.0});
    const Cylinder cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {2.0, 0.0, 0.0},
                            2.0, 0.0, std::numbers::pi);

    REQUIRE(static_cast<std::uint64_t>(gmm_ops::face_id_from_uv(
                rectangle, mesh, rectangle.to_uv(Point3D(1.0, 1.5, 0.0)))) ==
            2U);
    REQUIRE(static_cast<std::uint64_t>(gmm_ops::face_id_from_uv(
                cylinder, mesh, cylinder.to_uv(Point3D(0.0, 2.0, 1.5)),
                Side::Back)) == 7U);
}

TEST_CASE("face_id_from_uv rejects cube until topology is defined",
          "[gmm][ops]") {
    const ThermalMesh mesh;
    const Cube cube({0.0, 0.0, 0.0}, {2.0, 2.0, 2.0});

    REQUIRE_THROWS_AS(
        gmm_ops::face_id_from_uv(cube, mesh, cube.to_uv({1.0, 0.0, 0.0})),
        std::logic_error);
}
