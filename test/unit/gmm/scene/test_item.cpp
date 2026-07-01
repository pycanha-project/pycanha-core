#include <catch2/catch_test_macros.hpp>
#include <variant>

#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::GeometryId;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::MeshOptions;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::Triangle;

}  // namespace

TEST_CASE(
    "GeometryItem stores name, primitive, thermal mesh, transform, override",
    "[gmm][scene]") {
    GeometryItem item(
        "panel", Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{},
        CoordinateTransformation::from_translation({1.0, 2.0, 3.0}));

    REQUIRE(item.name() == "panel");
    // Id is assigned at construction; registration state is separate.
    REQUIRE(item.id() != GeometryId{0});
    REQUIRE(item.owning_model() == nullptr);  // unregistered until model.add()
    REQUIRE(std::holds_alternative<Rectangle>(item.primitive()));
    REQUIRE(item.transform()
                .apply({0.0, 0.0, 0.0})
                .isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));
    REQUIRE_FALSE(item.mesh_options_override().has_value());

    item.set_primitive(
        Triangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
    item.set_thermal_mesh(ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 1.0}});
    item.set_transform(
        CoordinateTransformation::from_translation({-1.0, -2.0, -3.0}));
    item.set_mesh_options_override(MeshOptions{1.0e-5});

    REQUIRE(std::holds_alternative<Triangle>(item.primitive()));
    REQUIRE(item.thermal_mesh().get_number_of_pair_faces() == 2U);
    REQUIRE(item.transform()
                .apply({0.0, 0.0, 0.0})
                .isApprox(Eigen::Vector3d(-1.0, -2.0, -3.0)));
    const auto mesh_options_override = item.mesh_options_override();
    REQUIRE(mesh_options_override.has_value());
    REQUIRE(mesh_options_override.value_or(MeshOptions{}).deviation_tolerance ==
            1.0e-5);
}

TEST_CASE("GeometryItem builds and caches its own mesh", "[gmm][scene]") {
    GeometryItem item(
        "panel", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{});

    REQUIRE(item.children().empty());
    REQUIRE(item.mesh().triangles.rows() > 0);
    // Cached: same object returned on the second call.
    REQUIRE(&item.mesh() == &item.mesh());
}
