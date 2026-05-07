#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ops/transform.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace {

using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cube;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Triangle;
namespace gmm_ops = pycanha::gmm::ops;

}  // namespace

TEST_CASE("Primitive transform applies rigid motions", "[gmm][ops]") {
    const CoordinateTransformation translation =
        CoordinateTransformation::from_translation({1.0, 2.0, 3.0});
    const CoordinateTransformation rotation =
        CoordinateTransformation::from_rotation(
            Eigen::AngleAxisd(std::numbers::pi / 2.0, Vector3D::UnitZ()));
    const CoordinateTransformation rigid = rotation.compose(translation);

    const Triangle triangle({1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {1.0, 1.0, 0.0});
    const Rectangle rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
                              {0.0, 1.0, 0.0});

    const Triangle transformed_triangle =
        std::get<Triangle>(gmm_ops::transform(triangle, rigid));
    const Rectangle transformed_rectangle =
        std::get<Rectangle>(gmm_ops::transform(rectangle, rigid));

    REQUIRE(transformed_triangle.p1().isApprox(Point3D(1.0, 3.0, 3.0)));
    REQUIRE(transformed_triangle.p2().isApprox(Point3D(1.0, 4.0, 3.0)));
    REQUIRE(transformed_rectangle.p3().isApprox(Point3D(0.0, 2.0, 3.0)));

    const Triangle round_tripped = std::get<Triangle>(
        gmm_ops::transform(transformed_triangle, rigid.inverse()));
    REQUIRE(round_tripped.p1().isApprox(triangle.p1()));
    REQUIRE(round_tripped.p2().isApprox(triangle.p2()));
    REQUIRE(round_tripped.p3().isApprox(triangle.p3()));
}

TEST_CASE("Cube transform preserves rigid rotation", "[gmm][ops]") {
    const Cube cube({0.0, 0.0, 0.0}, {2.0, 4.0, 6.0});
    const CoordinateTransformation rotation =
        CoordinateTransformation::from_rotation(
            Eigen::AngleAxisd(std::numbers::pi / 4.0, Vector3D::UnitZ()));
    const Cube transformed_cube =
        std::get<Cube>(gmm_ops::transform(cube, rotation));

    REQUIRE(transformed_cube.center().isApprox(Point3D(0.0, 0.0, 0.0)));
    REQUIRE(transformed_cube.to_cartesian({0.5, 0.5})
                .isApprox(Point3D(std::sqrt(0.5), std::sqrt(0.5), 0.0)));
    REQUIRE(transformed_cube.normal_at_uv({0.5, 0.5})
                .isApprox(Vector3D(std::sqrt(0.5), std::sqrt(0.5), 0.0)));
}
