#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "pycanha-core/conduction/links.hpp"
#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"

namespace {

using pycanha::conduction::FacePairLink;
using pycanha::conduction::intra_primitive_links;
using pycanha::conduction::through_thickness_conductance;
using pycanha::conduction::TmmBuildOptions;
using pycanha::gmm::ActiveSide;
using pycanha::gmm::BulkMaterial;
using pycanha::gmm::Primitive;
using pycanha::gmm::Quadrilateral;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;

// A shell whose side 1 has conductivity * thickness == 1, so a conductance
// reads directly as the geometric shape factor.
[[nodiscard]] ThermalMesh unit_shell(std::vector<double> dir1,
                                     std::vector<double> dir2,
                                     double conductance_thickness = 1.0) {
    ThermalMesh mesh(std::move(dir1), std::move(dir2));
    mesh.set_side1_material(std::make_shared<BulkMaterial>(
        "unit", 1.0, conductance_thickness, 1.0));
    mesh.set_side1_thick(1.0);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    return mesh;
}

[[nodiscard]] double link_value(const std::vector<FacePairLink>& links,
                                pycanha::MeshIndex face_pair_a,
                                pycanha::MeshIndex face_pair_b,
                                unsigned side = 1U) {
    double total = 0.0;
    for (const auto& link : links) {
        const bool matches = (link.face_pair_a == face_pair_a &&
                              link.face_pair_b == face_pair_b) ||
                             (link.face_pair_a == face_pair_b &&
                              link.face_pair_b == face_pair_a);
        if (matches && link.side == side) {
            total += link.conductance;
        }
    }
    return total;
}

// A symmetric trapezoid: the p1-p2 edge is 4 m long, the p4-p3 edge 2 m, and
// they sit 2 m apart. Its bilinear faces are trapezoids too, so both the
// shared edge and the reference distances have to come from the patch itself.
// Every number in the tests below is worked out by hand from
//   P(u, v) = (4u - 2uv + v,  2v,  0)
// which is the bilinear map on these four corners.
[[nodiscard]] Primitive make_trapezoid() {
    return Quadrilateral({0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {3.0, 2.0, 0.0},
                         {1.0, 2.0, 0.0});
}

}  // namespace

TEST_CASE("planar links: a split rectangle gives k t L / x",
          "[conduction][links]") {
    // 3 m along direction 1, 2 m along direction 2, cut once in the middle of
    // direction 1: the two face pairs are 1.5 m apart and share a 2 m edge.
    const Primitive rectangle =
        Rectangle({0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {0.0, 2.0, 0.0});
    const ThermalMesh mesh = unit_shell({0.0, 0.5, 1.0}, {0.0, 1.0});

    const auto links =
        intra_primitive_links(rectangle, mesh, TmmBuildOptions{});
    REQUIRE(links.size() == 1U);
    REQUIRE(link_value(links, 0U, 1U) == Catch::Approx(2.0 / 1.5));
}

TEST_CASE("planar links: a uniform grid conducts in both directions",
          "[conduction][links]") {
    const Primitive rectangle =
        Rectangle({0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {0.0, 3.0, 0.0});
    // 4 x 3 face pairs of 1 m x 1 m.
    const ThermalMesh mesh = unit_shell({0.0, 0.25, 0.5, 0.75, 1.0},
                                        {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0});

    const auto links =
        intra_primitive_links(rectangle, mesh, TmmBuildOptions{});
    // 3 direction-1 links per row (3 rows) + 2 direction-2 links per column
    // (4 columns).
    REQUIRE(links.size() == (3U * 3U) + (2U * 4U));
    for (const auto& link : links) {
        REQUIRE(link.conductance == Catch::Approx(1.0));
        REQUIRE(link.side == 1U);
    }
    // k = 1 W/(m K), t = 1 m, edge 1 m, distance 1 m -> 1 W/K.
    REQUIRE(link_value(links, 0U, 1U) == Catch::Approx(1.0));
    REQUIRE(link_value(links, 0U, 4U) == Catch::Approx(1.0));
}

TEST_CASE("planar links: non-uniform cuts use the reference midpoints",
          "[conduction][links]") {
    const Primitive rectangle =
        Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    // Face pairs [0, 0.2] and [0.2, 1.0]: references at 0.1 and 0.6, so 0.5
    // apart.
    const ThermalMesh mesh = unit_shell({0.0, 0.2, 1.0}, {0.0, 1.0});

    const auto links =
        intra_primitive_links(rectangle, mesh, TmmBuildOptions{});
    REQUIRE(link_value(links, 0U, 1U) == Catch::Approx(1.0 / 0.5));
}

TEST_CASE("planar links: a trapezoid conducts across its own shared edges",
          "[conduction][links]") {
    SECTION("split along direction 1") {
        // The shared edge runs from P(0.5, 0) = (2, 0) to P(0.5, 1) = (2, 2),
        // so it is 2 m long. The reference points are P(0.25, 0.5) = (1.25, 1)
        // and P(0.75, 0.5) = (2.75, 1), each 0.75 m from the edge midpoint
        // (2, 1). With k t = 1 the conductance is 2 / (0.75 + 0.75).
        const auto links = intra_primitive_links(
            make_trapezoid(), unit_shell({0.0, 0.5, 1.0}, {0.0, 1.0}),
            TmmBuildOptions{});
        REQUIRE(link_value(links, 0U, 1U) == Catch::Approx(2.0 / 1.5));
    }

    SECTION("split along direction 2") {
        // The shared edge runs from P(0, 0.5) = (0.5, 1) to P(1, 0.5) =
        // (3.5, 1): 3 m long, the trapezoid's mid-height width. The reference
        // points are (2, 0.5) and (2, 1.5), each 0.5 m from the edge midpoint.
        const auto links = intra_primitive_links(
            make_trapezoid(), unit_shell({0.0, 1.0}, {0.0, 0.5, 1.0}),
            TmmBuildOptions{});
        REQUIRE(link_value(links, 0U, 1U) == Catch::Approx(3.0 / 1.0));
    }
}

TEST_CASE("planar links: a trapezoid is not its equivalent rectangle",
          "[conduction][links]") {
    // The rule this replaces spanned p2 - p1 and the orthogonal part of
    // p4 - p1, which for this trapezoid is a 4 m x 2 m rectangle -- a third
    // more area than the shape has, and a direction-2 conductance to match.
    const ThermalMesh mesh = unit_shell({0.0, 1.0}, {0.0, 0.5, 1.0});
    const Primitive equivalent_rectangle =
        Rectangle({0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {0.0, 2.0, 0.0});

    const auto rectangle_links =
        intra_primitive_links(equivalent_rectangle, mesh, TmmBuildOptions{});
    const auto trapezoid_links =
        intra_primitive_links(make_trapezoid(), mesh, TmmBuildOptions{});

    REQUIRE(link_value(rectangle_links, 0U, 1U) == Catch::Approx(4.0));
    REQUIRE(link_value(trapezoid_links, 0U, 1U) == Catch::Approx(3.0));
}

TEST_CASE("planar links: the two sides are independent sheets",
          "[conduction][links]") {
    const Primitive rectangle =
        Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    ThermalMesh mesh({0.0, 0.5, 1.0}, {0.0, 1.0});
    mesh.set_side1_material(std::make_shared<BulkMaterial>("a", 1.0, 4.0, 1.0));
    mesh.set_side1_thick(0.5);  // k t = 2
    mesh.set_side2_material(std::make_shared<BulkMaterial>("b", 1.0, 1.0, 1.0));
    mesh.set_side2_thick(3.0);  // k t = 3

    const auto links =
        intra_primitive_links(rectangle, mesh, TmmBuildOptions{});
    REQUIRE(links.size() == 2U);
    // Shape factor: 1 m edge over a 1 m reference distance.
    REQUIRE(link_value(links, 0U, 1U, 1U) == Catch::Approx(2.0));
    REQUIRE(link_value(links, 0U, 1U, 2U) == Catch::Approx(3.0));
}

TEST_CASE("planar links: an inactive or material-less side conducts nothing",
          "[conduction][links]") {
    const Primitive rectangle =
        Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.0, 0.0});

    ThermalMesh no_material({0.0, 0.5, 1.0}, {0.0, 1.0});
    REQUIRE(intra_primitive_links(rectangle, no_material, TmmBuildOptions{})
                .empty());

    ThermalMesh no_thickness = no_material;
    no_thickness.set_side1_material(
        std::make_shared<BulkMaterial>("m", 1.0, 100.0, 1.0));
    REQUIRE(intra_primitive_links(rectangle, no_thickness, TmmBuildOptions{})
                .empty());

    ThermalMesh inactive = no_thickness;
    inactive.set_side1_thick(1.0);
    inactive.set_conductive_active_side(ActiveSide::None);
    REQUIRE(
        intra_primitive_links(rectangle, inactive, TmmBuildOptions{}).empty());
}

TEST_CASE("through-thickness: the two half-slabs are in series",
          "[conduction][links]") {
    ThermalMesh mesh;
    mesh.set_side1_material(std::make_shared<BulkMaterial>("a", 1.0, 2.0, 1.0));
    mesh.set_side1_thick(0.1);
    mesh.set_side2_material(std::make_shared<BulkMaterial>("b", 1.0, 5.0, 1.0));
    mesh.set_side2_thick(0.2);

    // A / (t1/k1 + t2/k2) = 3 / (0.05 + 0.04).
    REQUIRE(through_thickness_conductance(mesh, 3.0) ==
            Catch::Approx(3.0 / 0.09));

    // Zero thickness on both sides leaves no resistance and no conductor.
    ThermalMesh massless = mesh;
    massless.set_side1_thick(0.0);
    massless.set_side2_thick(0.0);
    REQUIRE(through_thickness_conductance(massless, 3.0) == 0.0);

    // A conductively inactive side breaks the path.
    ThermalMesh half_inactive = mesh;
    half_inactive.set_conductive_active_side(ActiveSide::Side1);
    REQUIRE(through_thickness_conductance(half_inactive, 3.0) == 0.0);
}
