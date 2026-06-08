#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <numbers>
#include <vector>

#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Geometry;
using pycanha::gmm::GeometryGroup;
using pycanha::gmm::GeometryGroupCutted;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Sphere;
using pycanha::gmm::ThermalMesh;

[[nodiscard]] std::shared_ptr<GeometryItem> make_panel(const char* name) {
    return std::make_shared<GeometryItem>(
        name, Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{});
}

[[nodiscard]] std::shared_ptr<GeometryItem> make_sphere_cutter(
    const char* name) {
    using std::numbers::pi;
    return std::make_shared<GeometryItem>(
        name,
        Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 1.0, -1.0,
               1.0, 0.0, 2.0 * pi),
        ThermalMesh{});
}

}  // namespace

TEST_CASE("GeometryGroup starts empty and preserves its transform",
          "[gmm][scene]") {
    GeometryGroup group(
        "rig", {}, CoordinateTransformation::from_translation({3.0, 2.0, 1.0}));

    REQUIRE(group.name() == "rig");
    REQUIRE(group.transform()
                .apply({0.0, 0.0, 0.0})
                .isApprox(Eigen::Vector3d(3.0, 2.0, 1.0)));
    REQUIRE(group.children().empty());

    group.add(make_panel("panel"));
    REQUIRE(group.children().size() == 1U);
    REQUIRE(group.children()[0]->name() == "panel");
}

TEST_CASE("GeometryGroup rejects null and duplicate children", "[gmm][scene]") {
    GeometryGroup group("rig");
    auto panel = make_panel("panel");
    group.add(panel);
    REQUIRE_THROWS(group.add(nullptr));
    REQUIRE_THROWS(group.add(panel));
}

TEST_CASE("GeometryGroupCutted only accepts closed solid cutters",
          "[gmm][scene]") {
    auto target = make_panel("target");
    GeometryGroupCutted cut_group(
        "trimmed",
        std::vector<std::shared_ptr<Geometry>>{target},
        std::vector<std::shared_ptr<GeometryItem>>{});

    REQUIRE(cut_group.name() == "trimmed");
    REQUIRE(cut_group.targets().size() == 1U);
    REQUIRE(cut_group.cutters().empty());

    REQUIRE_NOTHROW(cut_group.cut_with(make_sphere_cutter("sphere_cutter")));
    REQUIRE(cut_group.cutters().size() == 1U);

    auto flat = make_panel("flat_cutter");
    REQUIRE_THROWS(cut_group.cut_with(flat));
}
