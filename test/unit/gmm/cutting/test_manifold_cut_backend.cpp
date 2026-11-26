#include <Eigen/Dense>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>
#include <span>
#include <utility>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/cutting/cutter.hpp"
#include "pycanha-core/gmm/cutting/manifold_cut_backend.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/ops/validate.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangular_prism.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::gmm::Cone;
using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cube;
using pycanha::gmm::CutSense;
using pycanha::gmm::Cylinder;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::Primitive;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Sphere;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriangularPrism;
namespace cutting = pycanha::gmm::cutting;
namespace mesh_ops = pycanha::gmm::mesh::ops;

// A 2 m x 2 m panel in the z = 0 plane, split into four face pairs.
[[nodiscard]] GeometryItem make_panel() {
    return {"panel",
            Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}),
            ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}};
}

// A cylinder across the z = 0 plane, with its axis at (x, y).
[[nodiscard]] Cylinder make_bar(double x, double y, double radius) {
    return {{x, y, -1.0}, {x, y, 1.0}, {x + radius, y, -1.0},
            radius,       0.0,         2.0 * std::numbers::pi};
}

[[nodiscard]] cutting::Cutter hole(Primitive solid) {
    return {.solid = std::move(solid)};
}

[[nodiscard]] cutting::Cutter trim(Primitive solid) {
    return {.solid = std::move(solid), .sense = CutSense::KeepInside};
}

[[nodiscard]] double cut_area(const GeometryItem& target,
                              std::span<const cutting::Cutter> cutters) {
    const cutting::ManifoldCutBackend backend;
    const auto mesh = backend.cut(target, cutters, CoordinateTransformation{},
                                  pycanha::gmm::MeshOptions{});
    return mesh_ops::compute_areas(mesh).sum();
}

// What each sense of `solid` leaves of `target` on its own.
struct SplitAreas {
    double inside = 0.0;
    double outside = 0.0;
};

[[nodiscard]] SplitAreas split_areas(const GeometryItem& target,
                                     const Primitive& solid) {
    const std::array<cutting::Cutter, 1> keeps_inside{trim(solid)};
    const std::array<cutting::Cutter, 1> keeps_outside{hole(solid)};
    return {.inside = cut_area(target, keeps_inside),
            .outside = cut_area(target, keeps_outside)};
}

// A cylinder cutter is a 64-sided prism inscribed in its circle.
[[nodiscard]] double polygon_area(double radius) {
    return 32.0 * radius * radius * std::sin(std::numbers::pi / 32.0);
}

}  // namespace

TEST_CASE("ManifoldCutBackend cuts a rectangle with a cylinder",
          "[gmm][cutting]") {
    const cutting::ManifoldCutBackend backend;
    const GeometryItem panel = make_panel();
    const std::array<cutting::Cutter, 1> cutters{
        hole(make_bar(1.0, 1.0, 0.35))};

    const auto mesh = backend.cut(panel, cutters, CoordinateTransformation{},
                                  pycanha::gmm::MeshOptions{});

    REQUIRE(mesh.vertices.rows() > 0);
    REQUIRE(mesh.triangles.rows() > 0);
    REQUIRE_FALSE(mesh_ops::is_watertight(mesh));
    REQUIRE(mesh_ops::has_consistent_face_ids(mesh));
    REQUIRE(mesh_ops::compute_areas(mesh).sum() < 4.0);
}

TEST_CASE("A cutter that removes its outside leaves what it encloses",
          "[gmm][cutting]") {
    const GeometryItem panel = make_panel();

    const SplitAreas middle = split_areas(panel, make_bar(1.0, 1.0, 0.35));
    CHECK(middle.inside == Catch::Approx(polygon_area(0.35)).epsilon(2e-6));
    CHECK(middle.inside + middle.outside == Catch::Approx(4.0).epsilon(2e-6));

    // About a corner of the panel: a quarter of the circle is on it.
    const SplitAreas corner = split_areas(panel, make_bar(0.0, 0.0, 1.0));
    CHECK(corner.inside ==
          Catch::Approx(0.25 * polygon_area(1.0)).epsilon(2e-6));
    CHECK(corner.inside + corner.outside == Catch::Approx(4.0).epsilon(2e-6));

    // Enclosing the panel leaves all of it, or none of it.
    const SplitAreas all = split_areas(panel, make_bar(1.0, 1.0, 5.0));
    CHECK(all.inside == Catch::Approx(4.0).epsilon(2e-6));
    CHECK(all.outside == 0.0);

    // Away from the panel: the other way round.
    const SplitAreas away = split_areas(panel, make_bar(9.0, 9.0, 0.8));
    CHECK(away.inside == 0.0);
    CHECK(away.outside == Catch::Approx(4.0).epsilon(2e-6));
}

TEST_CASE("A panel that is trimmed away keeps its faces", "[gmm][cutting]") {
    const cutting::ManifoldCutBackend backend;
    const GeometryItem panel = make_panel();
    const std::array<cutting::Cutter, 1> cutters{trim(make_bar(9.0, 9.0, 0.8))};
    const auto mesh = backend.cut(panel, cutters, CoordinateTransformation{},
                                  pycanha::gmm::MeshOptions{});
    REQUIRE(mesh.triangles.rows() == 0);
    REQUIRE(mesh.nf() == 8U);
}

TEST_CASE("Trims and holes combine in any order", "[gmm][cutting]") {
    const GeometryItem panel = make_panel();
    const cutting::Cutter wide = trim(make_bar(1.0, 1.0, 0.8));
    const cutting::Cutter narrow = trim(make_bar(1.0, 1.0, 0.6));
    const cutting::Cutter bore = hole(make_bar(1.0, 1.0, 0.2));
    const cutting::Cutter edge = hole(make_bar(2.0, 1.0, 0.3));

    // The hole at the panel's edge takes nothing that the trims leave.
    const double expected = polygon_area(0.6) - polygon_area(0.2);
    const std::array<cutting::Cutter, 4> trims_first{wide, narrow, bore, edge};
    const std::array<cutting::Cutter, 4> holes_first{edge, bore, narrow, wide};
    const std::array<cutting::Cutter, 4> mixed{bore, wide, edge, narrow};
    CHECK(cut_area(panel, trims_first) ==
          Catch::Approx(expected).epsilon(2e-6));
    CHECK(cut_area(panel, holes_first) ==
          Catch::Approx(expected).epsilon(2e-6));
    CHECK(cut_area(panel, mixed) == Catch::Approx(expected).epsilon(2e-6));
}

TEST_CASE("A trim that touches the edges of the panel", "[gmm][cutting]") {
    // A 2.7 m x 2.4 m panel and a circle of radius 1.35 m about its centre:
    // tangent to two edges, across the other two.
    const double radius = 1.35;
    const double half_height = 1.2;
    const GeometryItem panel(
        "panel",
        Rectangle({-radius, -half_height, 0.0}, {radius, -half_height, 0.0},
                  {-radius, half_height, 0.0}),
        ThermalMesh{{0.0, 0.25, 0.5, 0.75, 1.0}, {0.0, 0.25, 0.5, 0.75, 1.0}});
    const std::array<cutting::Cutter, 1> cutters{
        trim(make_bar(0.0, 0.0, radius))};

    // The circle minus its two caps beyond the panel.
    const double cap = (radius * radius * std::acos(half_height / radius)) -
                       (half_height * std::sqrt((radius * radius) -
                                                (half_height * half_height)));
    const double exact = (std::numbers::pi * radius * radius) - (2.0 * cap);
    CHECK(cut_area(panel, cutters) == Catch::Approx(exact).epsilon(2e-3));
}

TEST_CASE("Every closed solid can trim", "[gmm][cutting]") {
    using std::numbers::pi;
    const GeometryItem panel = make_panel();

    // Section of each solid with the panel's plane, and how far below it the
    // solid's own tessellation may fall.
    const auto check = [&panel](const Primitive& solid, double section,
                                double tolerance) {
        const SplitAreas areas = split_areas(panel, solid);
        CHECK(areas.inside <= section * (1.0 + 2e-6));
        CHECK(areas.inside >= section * (1.0 - tolerance));
        CHECK(areas.inside + areas.outside == Catch::Approx(4.0).epsilon(2e-6));
    };

    // Radius 0.6 m at mid-height. The surface that is cut is that of a thin
    // shell about the panel, a fraction of a millimetre above it, where the
    // cone is already a little narrower.
    check(Cone({1.0, 1.0, -1.0}, {1.0, 1.0, 1.0}, {1.8, 1.0, -1.0}, 0.8, 0.4,
               0.0, 2.0 * pi),
          polygon_area(0.6), 2e-4);
    check(Sphere({1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {2.0, 1.0, 0.0}, 0.7, -0.7,
                 0.7, 0.0, 2.0 * pi),
          pi * 0.7 * 0.7, 2e-3);
    check(Cube({1.0, 1.0, 0.0}, {1.2, 0.5, 1.0}), 0.6, 2e-6);
    // A box turned 45 degrees about z: a diamond of half-diagonal 1.2 m, less
    // its four corners, which reach beyond the panel's edges.
    check(Cube({1.0, 1.0, 0.0},
               {1.2 * std::numbers::sqrt2, 1.2 * std::numbers::sqrt2, 1.0},
               Eigen::Quaterniond(
                   Eigen::AngleAxisd(0.25 * pi, pycanha::Vector3D::UnitZ()))),
          (2.0 * 1.2 * 1.2) - (4.0 * 0.2 * 0.2), 2e-6);
    check(TriangularPrism({0.5, 0.5, -1.0}, {1.5, 0.5, -1.0}, {0.5, 1.5, -1.0},
                          {0.5, 0.5, 1.0}),
          0.5, 2e-6);
}

TEST_CASE("A curved target is trimmed as well as a flat one",
          "[gmm][cutting]") {
    using std::numbers::pi;
    // A cylinder shell inside a wider coaxial cylinder over half its length.
    const GeometryItem shell("shell",
                             Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0},
                                      {1.0, 0.0, 0.0}, 1.0, 0.0, 2.0 * pi),
                             ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}});
    const Cylinder sleeve({0.0, 0.0, -1.0}, {0.0, 0.0, 1.0}, {1.5, 0.0, -1.0},
                          1.5, 0.0, 2.0 * pi);
    const double whole = cut_area(shell, {});
    const SplitAreas areas = split_areas(shell, sleeve);
    CHECK(areas.inside == Catch::Approx(0.5 * whole).epsilon(1e-3));
    CHECK(areas.inside + areas.outside == Catch::Approx(whole).epsilon(1e-3));
}
