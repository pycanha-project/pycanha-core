#include <manifold/manifold.h>

#include <catch2/catch_test_macros.hpp>
#include <cstddef>

#include "pycanha-core/gmm/cutting/proxy_shell.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/uv_mesher.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"

TEST_CASE("Proxy shell builds a valid manifold from a rectangle mesh",
          "[gmm][cutting]") {
    const pycanha::gmm::UvMesher mesher;
    const auto mesh =
        mesher.mesh(pycanha::gmm::Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0}),
                    pycanha::gmm::ThermalMesh{}, pycanha::gmm::MeshOptions{});

    const auto proxy = pycanha::gmm::cutting::build_primitive_proxy(
        mesh, {pycanha::gmm::GeometryId{}, 1.0e-3,
               manifold::Manifold::ReserveIDs(3U)});

    REQUIRE(proxy.Status() == manifold::Manifold::Error::NoError);
    REQUIRE(proxy.NumTri() > static_cast<std::size_t>(mesh.triangles.rows()));
    REQUIRE(proxy.Volume() > 0.0);
}
