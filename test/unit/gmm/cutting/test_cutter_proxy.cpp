#include <catch2/catch_test_macros.hpp>
#include <numbers>

#include "pycanha-core/gmm/cutting/cutter_proxy.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"

namespace {

using pycanha::gmm::Cone;
using pycanha::gmm::Cube;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Sphere;
using pycanha::gmm::cutting::build_cutter;

}  // namespace

TEST_CASE("Cutter proxies build closed solids", "[gmm][cutting]") {
    const auto sphere =
        build_cutter(Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0},
                            1.0, -1.0, 1.0, 0.0, 2.0 * std::numbers::pi));
    const auto cylinder = build_cutter(
        Cylinder({0.0, 0.0, -1.0}, {0.0, 0.0, 1.0}, {0.5, 0.0, -1.0}, 0.5, 0.0,
                 2.0 * std::numbers::pi));
    const auto cone =
        build_cutter(Cone({0.0, 0.0, -1.0}, {0.0, 0.0, 1.0}, {0.75, 0.0, -1.0},
                          0.75, 0.25, 0.0, 2.0 * std::numbers::pi));
    const auto cube = build_cutter(Cube({0.0, 0.0, 0.0}, {1.0, 2.0, 3.0}));

    REQUIRE(sphere.Status() == manifold::Manifold::Error::NoError);
    REQUIRE(cylinder.Status() == manifold::Manifold::Error::NoError);
    REQUIRE(cone.Status() == manifold::Manifold::Error::NoError);
    REQUIRE(cube.Status() == manifold::Manifold::Error::NoError);
    REQUIRE(sphere.Volume() > 0.0);
    REQUIRE(cylinder.Volume() > 0.0);
    REQUIRE(cone.Volume() > 0.0);
    REQUIRE(cube.Volume() > 0.0);
}
