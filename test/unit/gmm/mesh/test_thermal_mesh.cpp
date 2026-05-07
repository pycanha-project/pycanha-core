#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <stdexcept>

#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"

namespace {

using pycanha::gmm::Side;
using pycanha::gmm::ThermalMesh;

}  // namespace

TEST_CASE("ThermalMesh exposes ordered cuts and face ids", "[gmm][mesh]") {
    const ThermalMesh mesh({0.0, 0.25, 1.0}, {0.0, 0.5, 1.0});

    REQUIRE(mesh.dir1_cuts().size() == 3);
    REQUIRE(mesh.dir2_cuts().size() == 3);
    REQUIRE(mesh.num_faces_per_side() == 4);
    REQUIRE(static_cast<std::uint64_t>(mesh.face_id(0, 0, Side::Front)) == 0U);
    REQUIRE(static_cast<std::uint64_t>(mesh.face_id(0, 0, Side::Back)) == 1U);
    REQUIRE(static_cast<std::uint64_t>(mesh.face_id(1, 1, Side::Front)) == 6U);
}

TEST_CASE("ThermalMesh rejects invalid cut definitions", "[gmm][mesh]") {
    REQUIRE_THROWS_AS(ThermalMesh({0.0, 0.9, 0.2, 1.0}, {0.0, 1.0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(ThermalMesh({0.1, 1.0}, {0.0, 1.0}),
                      std::invalid_argument);
}
