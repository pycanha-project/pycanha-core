#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/uv_mesher.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"

namespace {

using pycanha::gmm::Cube;
using pycanha::gmm::MeshOptions;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::UvMesher;

}  // namespace

TEST_CASE("UvMesher rejects cube primitives", "[gmm][mesh]") {
    const Cube cube({0.0, 0.0, 0.0}, {2.0, 2.0, 2.0});
    const ThermalMesh thermal_mesh;
    const UvMesher mesher;

    REQUIRE_THROWS_AS(mesher.mesh(cube, thermal_mesh, MeshOptions{}),
                      std::logic_error);
}
