#pragma once

#include <type_traits>
#include <variant>

#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"

namespace pycanha::gmm {

using Primitive = std::variant<Triangle, Rectangle, Quadrilateral, Disc,
                               Cylinder, Cone, Sphere, Paraboloid, Cube>;

template <class Visitor>
using primitive_visitor_t =
    std::common_type_t<std::invoke_result_t<Visitor, const Triangle&>,
                       std::invoke_result_t<Visitor, const Rectangle&>,
                       std::invoke_result_t<Visitor, const Quadrilateral&>,
                       std::invoke_result_t<Visitor, const Disc&>,
                       std::invoke_result_t<Visitor, const Cylinder&>,
                       std::invoke_result_t<Visitor, const Cone&>,
                       std::invoke_result_t<Visitor, const Sphere&>,
                       std::invoke_result_t<Visitor, const Paraboloid&>,
                       std::invoke_result_t<Visitor, const Cube&>>;

}  // namespace pycanha::gmm
