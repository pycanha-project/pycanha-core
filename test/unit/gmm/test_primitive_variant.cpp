#include <array>
#include <catch2/catch_test_macros.hpp>
#include <numbers>
#include <variant>

#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::gmm {

TEST_CASE("Primitive variant stores every rewritten value type",
          "[gmm][primitive][variant]") {
    using std::numbers::pi;

    const std::array<Primitive, 9> primitives = {
        Primitive(Triangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 3.0, 0.0})),
        Primitive(Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 3.0, 0.0})),
        Primitive(Quadrilateral({0.0, 0.0, 0.0}, {2.0, 1.0, 0.0},
                                {3.0, 3.0, 0.0}, {1.0, 2.0, 0.0})),
        Primitive(Disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {2.0, 0.0, 0.0}, 1.0,
                       3.0, 0.0, pi)),
        Primitive(Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 4.0}, {1.0, 0.0, 0.0},
                           2.0, 0.0, 2.0 * pi)),
        Primitive(Cone({0.0, 0.0, 0.0}, {0.0, 0.0, 4.0}, {1.0, 0.0, 0.0}, 1.0,
                       3.0, 0.0, 2.0 * pi)),
        Primitive(Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 2.0,
                         -1.0, 1.0, 0.0, 2.0 * pi)),
        Primitive(Paraboloid({0.0, 0.0, 0.0}, {0.0, 0.0, 4.0}, {1.0, 0.0, 0.0},
                             2.0, 0.0, 2.0 * pi)),
        Primitive(Cube({0.0, 0.0, 0.0}, {2.0, 4.0, 6.0})),
    };

    for (const Primitive& primitive : primitives) {
        const double area = std::visit(
            [](const auto& value) { return value.surface_area(); }, primitive);
        REQUIRE(area > 0.0);
    }
}

}  // namespace pycanha::gmm
