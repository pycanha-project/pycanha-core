#include <catch2/catch_test_macros.hpp>
#include <numbers>

#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/cut_group.hpp"
#include "pycanha-core/gmm/scene/group.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::CutGroup;
using pycanha::gmm::Group;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Sphere;

}  // namespace

TEST_CASE("Group starts empty and preserves its transform", "[gmm][scene]") {
    const Group group(
        CoordinateTransformation::from_translation({3.0, 2.0, 1.0}));

    REQUIRE(group.transform()
                .apply({0.0, 0.0, 0.0})
                .isApprox(Eigen::Vector3d(3.0, 2.0, 1.0)));
    REQUIRE(group.child_item_indices().empty());
    REQUIRE(group.child_group_indices().empty());
    REQUIRE(group.child_cut_group_indices().empty());
}

TEST_CASE("CutGroup only accepts closed solid cutters", "[gmm][scene]") {
    using std::numbers::pi;

    CutGroup cut_group;
    REQUIRE_NOTHROW(cut_group.add_cutter(
        Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 1.0, -1.0,
               1.0, 0.0, 2.0 * pi)));
    REQUIRE(cut_group.cutters().size() == 1U);

    REQUIRE_THROWS(cut_group.add_cutter(
        Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0})));
}
