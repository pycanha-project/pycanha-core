#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "pycanha-core/gmm/ops/distance.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"

namespace {

using pycanha::gmm::Cone;
using pycanha::gmm::Cube;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Disc;
using pycanha::gmm::Paraboloid;
using pycanha::gmm::Quadrilateral;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Sphere;
using pycanha::gmm::Triangle;
namespace gmm_ops = pycanha::gmm::ops;

}  // namespace

TEST_CASE("Primitive distance handles representative points", "[gmm][ops]") {
    REQUIRE(gmm_ops::distance(
                Triangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}),
                {0.5, 0.5, 1.0}) == Catch::Approx(1.0));
    REQUIRE(gmm_ops::distance(
                Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 3.0, 0.0}),
                {1.0, 1.5, 2.0}) == Catch::Approx(2.0));
    REQUIRE(gmm_ops::distance(Quadrilateral({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
                                            {2.0, 2.0, 0.0}, {0.0, 2.0, 0.0}),
                              {1.0, 1.0, 1.5}) == Catch::Approx(1.5));
    REQUIRE(gmm_ops::distance(
                Disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 0.0,
                     2.0, 0.0, 2.0 * std::numbers::pi),
                {0.0, 0.0, 3.0}) == Catch::Approx(3.0));
    REQUIRE(gmm_ops::distance(
                Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {2.0, 0.0, 0.0}, 2.0,
                         0.0, 2.0 * std::numbers::pi),
                {0.0, 0.0, 1.0}) == Catch::Approx(2.0));
    REQUIRE(gmm_ops::distance(
                Cone({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {1.0, 0.0, 0.0}, 1.0,
                     2.0, 0.0, 2.0 * std::numbers::pi),
                {1.5, 0.0, 0.0}) == Catch::Approx(std::sqrt(0.2)));
    REQUIRE(gmm_ops::distance(
                Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 2.0,
                       -2.0, 2.0, 0.0, 2.0 * std::numbers::pi),
                {0.0, 0.0, 0.0}) == Catch::Approx(2.0));
    REQUIRE(gmm_ops::distance(
                Paraboloid({0.0, 0.0, 0.0}, {0.0, 0.0, 4.0}, {1.0, 0.0, 0.0},
                           2.0, 0.0, 2.0 * std::numbers::pi),
                {0.0, 0.0, 1.0}) == Catch::Approx(0.8660254037844386));
    REQUIRE(gmm_ops::distance(Cube({0.0, 0.0, 0.0}, {2.0, 2.0, 2.0}),
                              {0.0, 0.0, 0.0}) == Catch::Approx(1.0));
}
