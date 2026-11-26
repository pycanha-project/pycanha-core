// Cuts whose result has a closed form, in both senses: what is left inside the
// cutter and what is left outside it.

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/cutting/cutter.hpp"
#include "pycanha-core/gmm/cutting/manifold_cut_backend.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"
#include "pycanha-core/gmm/primitives/triangular_prism.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::Vector3D;
using pycanha::gmm::Cone;
using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cube;
using pycanha::gmm::CutSense;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Disc;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::Paraboloid;
using pycanha::gmm::Primitive;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Sphere;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::Triangle;
using pycanha::gmm::TriangularPrism;
using pycanha::gmm::TriMeshD;
namespace cutting = pycanha::gmm::cutting;
namespace mesh_ops = pycanha::gmm::mesh::ops;

constexpr double pi = std::numbers::pi;

// Relative tolerances. A straight-sided cutter is exact, up to the thin shell
// the cut goes through; a curved one is a polyhedron inscribed in its surface,
// of 64 sides for a cylinder (1.6e-3 of its section) and 96 for a sphere.
constexpr double flat_tolerance = 2e-6;
constexpr double curved_tolerance = 3e-3;
// The target is also a polygon, or the cutter is a sphere.
constexpr double coarse_tolerance = 5e-3;

[[nodiscard]] cutting::Cutter hole(Primitive solid) {
    return {.solid = std::move(solid)};
}

[[nodiscard]] cutting::Cutter trim(Primitive solid) {
    return {.solid = std::move(solid), .sense = CutSense::KeepInside};
}

[[nodiscard]] TriMeshD cut_mesh(const Primitive& target,
                                const std::vector<cutting::Cutter>& with) {
    const cutting::ManifoldCutBackend backend;
    return backend.cut(
        GeometryItem("target", target,
                     ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}),
        with, CoordinateTransformation{}, pycanha::gmm::MeshOptions{});
}

[[nodiscard]] double cut_area(const Primitive& target,
                              const std::vector<cutting::Cutter>& with) {
    return mesh_ops::compute_areas(cut_mesh(target, with)).sum();
}

[[nodiscard]] Vector3D centroid(const TriMeshD& mesh) {
    const auto areas = mesh_ops::compute_areas(mesh);
    Vector3D moment = Vector3D::Zero();
    for (Eigen::Index row = 0; row < mesh.triangles.rows(); ++row) {
        Vector3D corners = Vector3D::Zero();
        for (Eigen::Index corner = 0; corner < 3; ++corner) {
            corners +=
                mesh.vertices.row(mesh.triangles(row, corner)).transpose();
        }
        moment += areas(row) * corners / 3.0;
    }
    return moment / areas.sum();
}

// The area that `solid` encloses of `target`, and all the rest, each obtained
// with the sense that leaves it.
void check_both_senses(const Primitive& target, const Primitive& solid,
                       double whole, double enclosed, double tolerance) {
    CHECK(cut_area(target, {trim(solid)}) ==
          Catch::Approx(enclosed).epsilon(tolerance));
    CHECK(cut_area(target, {hole(solid)}) ==
          Catch::Approx(whole - enclosed).epsilon(tolerance));
}

// A 4 m x 4 m panel in the z = 0 plane, centred at the origin.
[[nodiscard]] Rectangle make_panel() {
    return {{-2.0, -2.0, 0.0}, {2.0, -2.0, 0.0}, {-2.0, 2.0, 0.0}};
}

// A cylinder across the z = 0 plane, with its axis along z at (x, 0).
[[nodiscard]] Cylinder make_bar(double x, double radius) {
    return {{x, 0.0, -3.0}, {x, 0.0, 3.0}, {x + radius, 0.0, -3.0},
            radius,         0.0,           2.0 * pi};
}

// A box of the given extent along z, beyond every target in x and y.
[[nodiscard]] Cube make_slab(double z_min, double z_max) {
    return {{0.0, 0.0, 0.5 * (z_min + z_max)}, {8.0, 8.0, z_max - z_min}};
}

[[nodiscard]] Sphere make_sphere(double x, double radius) {
    return {{x, 0.0, 0.0}, {x, 0.0, radius}, {x + radius, 0.0, 0.0},
            radius,        -radius,          radius,
            0.0,           2.0 * pi};
}

// Area common to two circles whose centres are `distance` apart.
[[nodiscard]] double lens_area(double first, double second, double distance) {
    const double to_chord =
        ((distance * distance) + (first * first) - (second * second)) /
        (2.0 * distance);
    const auto segment = [](double radius, double height) {
        return (radius * radius * std::acos(height / radius)) -
               (height * std::sqrt((radius * radius) - (height * height)));
    };
    return segment(first, to_chord) + segment(second, distance - to_chord);
}

// Area of z = h (r / R)^2 from its apex up to the height z.
[[nodiscard]] double paraboloid_area(double radius, double height, double z) {
    const double slope = 2.0 * height / (radius * radius);
    const double reach = radius * std::sqrt(z / height);
    return (2.0 * pi / (3.0 * slope * slope)) *
           (std::pow(1.0 + (slope * slope * reach * reach), 1.5) - 1.0);
}

}  // namespace

TEST_CASE("Circle on a panel, in both senses", "[gmm][cutting][analytical]") {
    check_both_senses(make_panel(), make_bar(0.0, 0.8), 16.0, pi * 0.64,
                      curved_tolerance);
    // Half of it is beyond the edge of the panel.
    check_both_senses(make_panel(), make_bar(2.0, 0.8), 16.0, 0.5 * pi * 0.64,
                      curved_tolerance);
}

TEST_CASE("Oblique cylinder on a panel, in both senses",
          "[gmm][cutting][analytical]") {
    // The axis is 60 degrees off the panel's normal: the section is an ellipse
    // whose semi-axes are the radius and twice the radius.
    const double radius = 0.5;
    const Vector3D axis(std::sin(pi / 3.0), 0.0, std::cos(pi / 3.0));
    const Vector3D base = -4.0 * axis;
    const Cylinder bar(base, 4.0 * axis, base + (radius * Vector3D::UnitY()),
                       radius, 0.0, 2.0 * pi);
    check_both_senses(make_panel(), bar, 16.0, pi * radius * (2.0 * radius),
                      curved_tolerance);
}

TEST_CASE("Sphere through a panel, in both senses",
          "[gmm][cutting][analytical]") {
    // Centred on the panel, and then 0.6 m above it.
    check_both_senses(make_panel(), make_sphere(0.0, 1.0), 16.0, pi,
                      coarse_tolerance);
    const Sphere above({0.0, 0.0, 0.6}, {0.0, 0.0, 1.6}, {1.0, 0.0, 0.6}, 1.0,
                       -1.0, 1.0, 0.0, 2.0 * pi);
    check_both_senses(make_panel(), above, 16.0, pi * (1.0 - 0.36),
                      coarse_tolerance);
}

TEST_CASE("Box and prism on a panel, in both senses",
          "[gmm][cutting][analytical]") {
    check_both_senses(make_panel(), Cube({0.3, -0.2, 0.0}, {1.5, 0.8, 1.0}),
                      16.0, 1.2, flat_tolerance);
    // The same box, turned about the panel's normal.
    check_both_senses(
        make_panel(),
        Cube({0.3, -0.2, 0.0}, {1.5, 0.8, 1.0},
             Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Vector3D::UnitZ()))),
        16.0, 1.2, flat_tolerance);
    check_both_senses(make_panel(),
                      TriangularPrism({0.0, 0.0, -1.0}, {1.5, 0.0, -1.0},
                                      {0.0, 1.0, -1.0}, {0.0, 0.0, 1.0}),
                      16.0, 0.75, flat_tolerance);
}

TEST_CASE("Incircle of a triangle, in both senses",
          "[gmm][cutting][analytical]") {
    // Legs of 3 m and 4 m: the incircle has a radius of 1 m.
    const Triangle target({-1.0, -1.0, 0.0}, {3.0, -1.0, 0.0},
                          {-1.0, 2.0, 0.0});
    check_both_senses(target, make_bar(0.0, 1.0), 6.0, pi, curved_tolerance);
}

TEST_CASE("Circle on a disc, in both senses", "[gmm][cutting][analytical]") {
    const Disc target({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 0.0,
                      1.0, 0.0, 2.0 * pi);
    check_both_senses(target, make_bar(0.0, 0.6), pi, 0.36 * pi,
                      coarse_tolerance);
    // Off centre, across the rim: a lens is left, or a lune.
    check_both_senses(target, make_bar(1.1, 0.7), pi, lens_area(1.0, 0.7, 1.1),
                      coarse_tolerance);
}

TEST_CASE("Slab and half space of a cylinder, in both senses",
          "[gmm][cutting][analytical]") {
    const Cylinder target({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {1.0, 0.0, 0.0},
                          1.0, 0.0, 2.0 * pi);
    const double whole = 4.0 * pi;
    check_both_senses(target, make_slab(0.3, 0.8), whole, 2.0 * pi * 0.5,
                      curved_tolerance);
    // x > 0.5 is a third of the turn.
    check_both_senses(target, Cube({4.5, 0.0, 0.0}, {8.0, 8.0, 8.0}), whole,
                      whole / 3.0, curved_tolerance);
}

TEST_CASE("Zone and caps of a sphere, in both senses",
          "[gmm][cutting][analytical]") {
    const Sphere target = make_sphere(0.0, 1.0);
    const double whole = 4.0 * pi;
    // A zone has the area of the cylinder around it.
    check_both_senses(target, make_slab(-0.3, 0.4), whole, 2.0 * pi * 0.7,
                      curved_tolerance);
    check_both_senses(target, make_slab(0.5, 3.0), whole, 2.0 * pi * 0.5,
                      curved_tolerance);
    // A bore through the centre takes a cap off either end.
    check_both_senses(target, make_bar(0.0, 0.6), whole, 2.0 * 2.0 * pi * 0.2,
                      curved_tolerance);
    // Two unit spheres 1 m apart meet halfway: a cap 0.5 m high.
    check_both_senses(target, make_sphere(1.0, 1.0), whole, 2.0 * pi * 0.5,
                      coarse_tolerance);
}

TEST_CASE("Frustum of a cone, in both senses", "[gmm][cutting][analytical]") {
    // From a radius of 1 m down to 0.5 m over a height of 1 m.
    const Cone target({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 1.0,
                      0.5, 0.0, 2.0 * pi);
    const double slant = std::sqrt(1.25);
    check_both_senses(target, make_slab(0.25, 0.75), pi * 1.5 * slant,
                      pi * (0.875 + 0.625) * 0.5 * slant, curved_tolerance);
}

TEST_CASE("Bowl of a paraboloid, in both senses",
          "[gmm][cutting][analytical]") {
    const Paraboloid target({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0},
                            1.0, 0.0, 2.0 * pi);
    check_both_senses(target, make_slab(-1.0, 0.4),
                      paraboloid_area(1.0, 1.0, 1.0),
                      paraboloid_area(1.0, 1.0, 0.4), curved_tolerance);
}

TEST_CASE("Two circles on a panel, in every combination of senses",
          "[gmm][cutting][analytical]") {
    const Cylinder left = make_bar(0.0, 1.0);
    const Cylinder right = make_bar(0.9, 0.7);
    const double left_area = pi;
    const double right_area = 0.49 * pi;
    const double lens = lens_area(1.0, 0.7, 0.9);

    CHECK(cut_area(make_panel(), {trim(left), trim(right)}) ==
          Catch::Approx(lens).epsilon(curved_tolerance));
    CHECK(cut_area(make_panel(), {trim(left), hole(right)}) ==
          Catch::Approx(left_area - lens).epsilon(curved_tolerance));
    CHECK(cut_area(make_panel(), {hole(left), trim(right)}) ==
          Catch::Approx(right_area - lens).epsilon(curved_tolerance));
    CHECK(cut_area(make_panel(), {hole(left), hole(right)}) ==
          Catch::Approx(16.0 - left_area - right_area + lens)
              .epsilon(curved_tolerance));
}

TEST_CASE("Ring and frame cut out of a panel", "[gmm][cutting][analytical]") {
    // The hole need not be concentric.
    CHECK(cut_area(make_panel(),
                   {trim(make_bar(0.0, 1.2)), hole(make_bar(0.4, 0.5))}) ==
          Catch::Approx(pi * (1.44 - 0.25)).epsilon(curved_tolerance));
    // A square outline with a round hole, and a round one with a square hole.
    const Cube square({0.0, 0.0, 0.0}, {1.4, 1.4, 1.0});
    CHECK(cut_area(make_panel(), {trim(square), hole(make_bar(0.0, 0.5))}) ==
          Catch::Approx(1.96 - (0.25 * pi)).epsilon(curved_tolerance));
    CHECK(cut_area(make_panel(), {trim(make_bar(0.0, 1.5)), hole(square)}) ==
          Catch::Approx((2.25 * pi) - 1.96).epsilon(curved_tolerance));
}

// Equal areas could still be the wrong part of the panel.
TEST_CASE("The part left is where the cutters say",
          "[gmm][cutting][analytical]") {
    // A half disc of radius 0.9 m on the +x side of the origin.
    const Cube right_half({2.0, 0.0, 0.0}, {4.0, 8.0, 1.0});
    const TriMeshD half =
        cut_mesh(make_panel(), {trim(make_bar(0.0, 0.9)), trim(right_half)});
    const Vector3D centre = centroid(half);
    CHECK(centre.x() ==
          Catch::Approx(4.0 * 0.9 / (3.0 * pi)).epsilon(curved_tolerance));
    CHECK(centre.y() == Catch::Approx(0.0).margin(1e-6));

    // Nothing of it reaches beyond the circle.
    const double reach = std::sqrt(
        half.vertices.leftCols<2>().rowwise().squaredNorm().maxCoeff());
    CHECK(reach == Catch::Approx(0.9).epsilon(1e-6));

    // A lune is what the far circle leaves of the near one, on its own side.
    const TriMeshD lune = cut_mesh(
        make_panel(), {trim(make_bar(0.0, 0.9)), hole(make_bar(0.8, 0.9))});
    const double lens = lens_area(0.9, 0.9, 0.8);
    const double expected = -(lens * 0.4) / ((pi * 0.81) - lens);
    CHECK(centroid(lune).x() ==
          Catch::Approx(expected).epsilon(curved_tolerance));
}
