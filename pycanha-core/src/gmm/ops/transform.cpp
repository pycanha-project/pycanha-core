#include "pycanha-core/gmm/ops/transform.hpp"

#include <variant>

#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::gmm::ops {
namespace {

[[nodiscard]] Primitive transform(
    const Triangle& triangle, const CoordinateTransformation& transformation) {
    return Triangle(transformation.apply(triangle.p1()),
                    transformation.apply(triangle.p2()),
                    transformation.apply(triangle.p3()));
}

[[nodiscard]] Primitive transform(
    const Rectangle& rectangle,
    const CoordinateTransformation& transformation) {
    return Rectangle(transformation.apply(rectangle.p1()),
                     transformation.apply(rectangle.p2()),
                     transformation.apply(rectangle.p3()));
}

[[nodiscard]] Primitive transform(
    const Quadrilateral& quadrilateral,
    const CoordinateTransformation& transformation) {
    return Quadrilateral(transformation.apply(quadrilateral.p1()),
                         transformation.apply(quadrilateral.p2()),
                         transformation.apply(quadrilateral.p3()),
                         transformation.apply(quadrilateral.p4()));
}

[[nodiscard]] Primitive transform(
    const Disc& disc, const CoordinateTransformation& transformation) {
    return Disc(transformation.apply(disc.p1()),
                transformation.apply(disc.p2()),
                transformation.apply(disc.p3()), disc.inner_radius(),
                disc.outer_radius(), disc.start_angle(), disc.end_angle());
}

[[nodiscard]] Primitive transform(
    const Cylinder& cylinder, const CoordinateTransformation& transformation) {
    return Cylinder(transformation.apply(cylinder.p1()),
                    transformation.apply(cylinder.p2()),
                    transformation.apply(cylinder.p3()), cylinder.radius(),
                    cylinder.start_angle(), cylinder.end_angle());
}

[[nodiscard]] Primitive transform(
    const Cone& cone, const CoordinateTransformation& transformation) {
    return Cone(transformation.apply(cone.p1()),
                transformation.apply(cone.p2()),
                transformation.apply(cone.p3()), cone.radius1(), cone.radius2(),
                cone.start_angle(), cone.end_angle());
}

[[nodiscard]] Primitive transform(
    const Sphere& sphere, const CoordinateTransformation& transformation) {
    return Sphere(transformation.apply(sphere.p1()),
                  transformation.apply(sphere.p2()),
                  transformation.apply(sphere.p3()), sphere.radius(),
                  sphere.base_truncation(), sphere.apex_truncation(),
                  sphere.start_angle(), sphere.end_angle());
}

[[nodiscard]] Primitive transform(
    const Paraboloid& paraboloid,
    const CoordinateTransformation& transformation) {
    return Paraboloid(transformation.apply(paraboloid.p1()),
                      transformation.apply(paraboloid.p2()),
                      transformation.apply(paraboloid.p3()),
                      paraboloid.radius(), paraboloid.start_angle(),
                      paraboloid.end_angle());
}

[[nodiscard]] Primitive transform(
    const Cube& cube, const CoordinateTransformation& transformation) {
    const Eigen::Quaterniond rotation(transformation.linear());
    return Cube(transformation.apply(cube.center()), cube.extent(),
                rotation * cube.orientation());
}

}  // namespace

Primitive transform(const Primitive& primitive,
                    const CoordinateTransformation& transformation) {
    return std::visit(
        [&transformation](const auto& concrete_primitive) {
            // NOLINTNEXTLINE(build/include_what_you_use)
            return transform(concrete_primitive, transformation);
        },
        primitive);
}

}  // namespace pycanha::gmm::ops
