#include <catch2/catch_test_macros.hpp>
#include <numbers>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace {

using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::CoordinateTransformation;

}  // namespace

TEST_CASE("CoordinateTransformation supports rigid motion composition",
          "[gmm][scene]") {
    const CoordinateTransformation identity;
    REQUIRE(identity.is_identity());
    REQUIRE(identity.apply(Point3D(1.0, 2.0, 3.0))
                .isApprox(Point3D(1.0, 2.0, 3.0)));

    const auto rotation = CoordinateTransformation::from_rotation(
        Eigen::AngleAxisd(std::numbers::pi / 2.0, Vector3D::UnitZ()));
    const CoordinateTransformation motion({1.0, 2.0, 3.0}, rotation.rotation());

    REQUIRE(motion.translation().isApprox(Vector3D(1.0, 2.0, 3.0)));
    REQUIRE(
        motion.apply(Point3D(1.0, 0.0, 0.0)).isApprox(Point3D(1.0, 3.0, 3.0)));
    REQUIRE(motion.apply_normal(Vector3D(1.0, 0.0, 0.0))
                .isApprox(Vector3D(0.0, 1.0, 0.0)));

    const auto inverse = motion.inverse();
    REQUIRE(inverse.apply(motion.apply(Point3D(2.0, 1.0, 0.0)))
                .isApprox(Point3D(2.0, 1.0, 0.0)));
    REQUIRE(inverse.apply_normal(motion.apply_normal(Vector3D(0.0, 1.0, 0.0)))
                .isApprox(Vector3D(0.0, 1.0, 0.0)));

    const auto translation =
        CoordinateTransformation::from_translation({1.0, 2.0, 0.0});
    const auto rigid = rotation.compose(translation);
    REQUIRE(
        rigid.apply(Point3D(1.0, 0.0, 0.0)).isApprox(Point3D(1.0, 3.0, 0.0)));

    auto from_euler = CoordinateTransformation::from_euler(
        {0.5, -0.5, 1.0}, {0.0, 0.0, std::numbers::pi / 2.0});
    REQUIRE(from_euler.apply(Point3D(1.0, 0.0, 0.0))
                .isApprox(Point3D(0.5, 0.5, 1.0)));

    from_euler.set_translation({0.0, 0.0, 0.0});
    from_euler.set_rotation(Eigen::Matrix3d::Identity());
    REQUIRE(from_euler.is_identity());
}
