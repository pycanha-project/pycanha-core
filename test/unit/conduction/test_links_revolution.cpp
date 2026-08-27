#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/conduction/links.hpp"
#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/conduction/profile.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"

namespace {

using pycanha::conduction::FacePairLink;
using pycanha::conduction::intra_primitive_links;
using pycanha::conduction::MeridianProfile;
using pycanha::conduction::profile_of;
using pycanha::conduction::TmmBuildOptions;
using pycanha::gmm::ActiveSide;
using pycanha::gmm::BulkMaterial;
using pycanha::gmm::Cone;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Disc;
using pycanha::gmm::Paraboloid;
using pycanha::gmm::Primitive;
using pycanha::gmm::Sphere;
using pycanha::gmm::ThermalMesh;

constexpr double pi = std::numbers::pi;

[[nodiscard]] ThermalMesh unit_shell(std::vector<double> dir1,
                                     std::vector<double> dir2) {
    ThermalMesh mesh(std::move(dir1), std::move(dir2));
    mesh.set_side1_material(
        std::make_shared<BulkMaterial>("unit", 1.0, 1.0, 1.0));
    mesh.set_side1_thick(1.0);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    return mesh;
}

[[nodiscard]] double link_value(const std::vector<FacePairLink>& links,
                                pycanha::MeshIndex face_pair_a,
                                pycanha::MeshIndex face_pair_b) {
    double total = 0.0;
    for (const auto& link : links) {
        const bool matches = (link.face_pair_a == face_pair_a &&
                              link.face_pair_b == face_pair_b) ||
                             (link.face_pair_a == face_pair_b &&
                              link.face_pair_b == face_pair_a);
        if (matches) {
            total += link.conductance;
        }
    }
    return total;
}

}  // namespace

TEST_CASE("revolution links: disc azimuthal conductance",
          "[conduction][links]") {
    const double inner = 0.5;
    const double outer = 2.0;
    const double angle = pi / 2.0;
    const Primitive disc = Disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
                                {outer, 0.0, 0.0}, inner, outer, 0.0, angle);
    // Two angular face pairs over the quadrant, one radial band.
    const ThermalMesh mesh = unit_shell({0.0, 0.5, 1.0}, {0.0, 1.0});

    const auto links = intra_primitive_links(disc, mesh, TmmBuildOptions{});
    REQUIRE(links.size() == 1U);
    // The strips at different radii are parallel resistors, so the band's
    // log-radius span multiplies and the angular distance divides.
    REQUIRE(link_value(links, 0U, 1U) ==
            Catch::Approx(std::log(outer / inner) / (angle / 2.0)));
}

TEST_CASE("revolution links: a full annulus radial conductor",
          "[conduction][links]") {
    const double inner = 0.5;
    const double outer = 2.0;
    const Primitive disc = Disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
                                {outer, 0.0, 0.0}, inner, outer, 0.0, 2.0 * pi);
    // One angular face pair spanning the whole revolution, two radial bands.
    const ThermalMesh mesh = unit_shell({0.0, 1.0}, {0.0, 0.5, 1.0});

    const auto links = intra_primitive_links(disc, mesh, TmmBuildOptions{});
    REQUIRE(links.size() == 1U);
    // Between the two band reference radii, which sit at 1/4 and 3/4.
    const double reference_inner = inner + (0.25 * (outer - inner));
    const double reference_outer = inner + (0.75 * (outer - inner));
    REQUIRE(
        link_value(links, 0U, 1U) ==
        Catch::Approx(2.0 * pi / std::log(reference_outer / reference_inner)));
}

TEST_CASE("revolution links: cylinder axial and circumferential",
          "[conduction][links]") {
    const double radius = 1.5;
    const double height = 4.0;
    const Primitive cylinder = Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, height},
                                        {radius, 0.0, 0.0}, radius, 0.0, pi);
    const ThermalMesh mesh = unit_shell({0.0, 0.5, 1.0}, {0.0, 0.5, 1.0});

    const auto links = intra_primitive_links(cylinder, mesh, TmmBuildOptions{});
    // 1 circumferential link per row (2 rows) + 1 axial link per column
    // (2 columns).
    REQUIRE(links.size() == 4U);

    // Circumferential: an edge of height/2 over an arc of radius * pi / 2.
    REQUIRE(link_value(links, 0U, 1U) ==
            Catch::Approx((height / 2.0) / (radius * pi / 2.0)));
    // Axial: an edge of radius * pi / 2 over a distance of height / 2.
    REQUIRE(link_value(links, 0U, 2U) ==
            Catch::Approx((radius * pi / 2.0) / (height / 2.0)));
}

TEST_CASE("revolution links: a cone of constant radius equals a cylinder",
          "[conduction][links]") {
    const double radius = 0.9;
    const double height = 2.2;
    const Primitive cone = Cone({0.0, 0.0, 0.0}, {0.0, 0.0, height},
                                {radius, 0.0, 0.0}, radius, radius, 0.0, pi);
    const Primitive cylinder = Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, height},
                                        {radius, 0.0, 0.0}, radius, 0.0, pi);
    const ThermalMesh mesh = unit_shell({0.0, 0.4, 1.0}, {0.0, 0.3, 1.0});

    const auto cone_links =
        intra_primitive_links(cone, mesh, TmmBuildOptions{});
    const auto cylinder_links =
        intra_primitive_links(cylinder, mesh, TmmBuildOptions{});
    REQUIRE(cone_links.size() == cylinder_links.size());
    for (std::size_t index = 0; index < cone_links.size(); ++index) {
        REQUIRE(
            cone_links[index].conductance ==
            Catch::Approx(cylinder_links[index].conductance).epsilon(1e-12));
    }
}

TEST_CASE("revolution links: a narrow sphere band tends to a cylinder",
          "[conduction][links]") {
    const double radius = 1.0;
    // A thin band straddling the equator, where the sphere is locally a
    // cylinder of the same radius.
    const double half_height = 1e-4;
    const Primitive sphere =
        Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, radius}, {radius, 0.0, 0.0}, radius,
               -half_height, half_height, 0.0, pi);
    const ThermalMesh mesh = unit_shell({0.0, 0.5, 1.0}, {0.0, 1.0});

    const auto links = intra_primitive_links(sphere, mesh, TmmBuildOptions{});
    REQUIRE(links.size() == 1U);
    // The band height is 2 * half_height and the arc is radius * pi / 2.
    REQUIRE(
        link_value(links, 0U, 1U) ==
        Catch::Approx((2.0 * half_height) / (radius * pi / 2.0)).epsilon(1e-6));
}

TEST_CASE("revolution links: the paraboloid agrees with its own profile",
          "[conduction][links]") {
    const double radius = 1.2;
    const double height = 2.4;
    const Primitive paraboloid =
        Paraboloid({0.0, 0.0, 0.0}, {0.0, 0.0, height}, {radius, 0.0, 0.0},
                   radius, 0.0, pi / 2.0);
    const ThermalMesh mesh = unit_shell({0.0, 1.0}, {0.0, 0.4, 1.0});
    const std::optional<MeridianProfile> profile = profile_of(paraboloid);
    if (!profile.has_value()) {
        throw std::logic_error("the paraboloid must have a closed form");
    }

    const auto links =
        intra_primitive_links(paraboloid, mesh, TmmBuildOptions{});
    REQUIRE(links.size() == 1U);
    const double expected =
        (pi / 2.0) / (profile->potential(0.7) - profile->potential(0.2));
    REQUIRE(link_value(links, 0U, 1U) == Catch::Approx(expected));
}

TEST_CASE(
    "revolution links: a disc reaching the centre uses the near-axis form",
    "[conduction][links]") {
    // A full disc down to r = 0. The plain integral assumes the angular
    // temperature difference is the same at every radius of the band, which
    // fails at the centre: the field is analytic there, so the difference
    // vanishes linearly with r. Imposing that gives the band's radial extent
    // over its own reference radius, dr / r_mid.
    const Primitive disc = Disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
                                {1.0, 0.0, 0.0}, 0.0, 1.0, 0.0, 2.0 * pi);
    const ThermalMesh mesh =
        unit_shell({0.0, 0.25, 0.5, 0.75, 1.0}, {0.0, 1.0});

    const auto links = intra_primitive_links(disc, mesh, TmmBuildOptions{});
    for (const auto& link : links) {
        REQUIRE(std::isfinite(link.conductance));
        REQUIRE(link.conductance > 0.0);
    }
    // dr / r_mid = 1 / 0.5 = 2, over a quarter turn.
    REQUIRE(link_value(links, 0U, 1U) == Catch::Approx(2.0 / (pi / 2.0)));
}

TEST_CASE("revolution links: the near-axis conductance is mesh independent",
          "[conduction][links]") {
    // The node sits at the middle of the band, so dr / r_mid is 2 whatever the
    // first radial cut is: refining towards the centre cannot inflate it.
    const Primitive disc = Disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
                                {1.0, 0.0, 0.0}, 0.0, 1.0, 0.0, 2.0 * pi);
    for (const double first_cut : {0.5, 0.1, 0.01}) {
        // Four angular face pairs, so face pairs 0 and 1 meet at one seam only;
        // with two the wrap would join the same pair a second time.
        const ThermalMesh mesh =
            unit_shell({0.0, 0.25, 0.5, 0.75, 1.0}, {0.0, first_cut, 1.0});
        const auto links = intra_primitive_links(disc, mesh, TmmBuildOptions{});
        REQUIRE(link_value(links, 0U, 1U) == Catch::Approx(2.0 / (pi / 2.0)));
    }
}

TEST_CASE("revolution links: every apex and pole stays finite",
          "[conduction][links]") {
    const ThermalMesh mesh = unit_shell({0.0, 0.5, 1.0}, {0.0, 0.5, 1.0});
    const std::vector<Primitive> primitives{
        // Cone apex at radius 0.
        Cone({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {1.5, 0.0, 0.0}, 0.0, 1.5, 0.0,
             2.0 * pi),
        // Sphere truncated at neither pole.
        Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 1.0, -1.0,
               1.0, 0.0, 2.0 * pi),
        // Paraboloid, whose apex is always on the axis.
        Paraboloid({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {1.0, 0.0, 0.0}, 1.0, 0.0,
                   2.0 * pi)};

    for (const auto& primitive : primitives) {
        const auto links =
            intra_primitive_links(primitive, mesh, TmmBuildOptions{});
        REQUIRE_FALSE(links.empty());
        for (const auto& link : links) {
            REQUIRE(std::isfinite(link.conductance));
            REQUIRE(link.conductance > 0.0);
        }
    }
}

TEST_CASE("revolution links: a sphere polar cap over its reference latitude",
          "[conduction][links]") {
    // The cap spans the upper half of the latitude range, up to the north
    // pole; its near-axis extent is the meridian arc over the parallel radius
    // of the cap's own reference latitude.
    const double radius = 1.0;
    const Primitive sphere =
        Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, radius}, {radius, 0.0, 0.0}, radius,
               0.0, radius, 0.0, 2.0 * pi);
    const ThermalMesh mesh =
        unit_shell({0.0, 0.25, 0.5, 0.75, 1.0}, {0.0, 0.5, 1.0});

    const auto links = intra_primitive_links(sphere, mesh, TmmBuildOptions{});
    // Latitudes run 0 to pi/2, so the polar band is [pi/4, pi/2] with its
    // reference at 3*pi/8. Face pairs 4 and 5 are the first two of that band.
    const double latitude_span = pi / 4.0;
    const double expected =
        (latitude_span / std::cos(3.0 * pi / 8.0)) / (pi / 2.0);
    REQUIRE(link_value(links, 4U, 5U) == Catch::Approx(expected));
}

TEST_CASE("revolution links: a full revolution closes the ring",
          "[conduction][links]") {
    const Primitive cylinder = Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
                                        {1.0, 0.0, 0.0}, 1.0, 0.0, 2.0 * pi);
    const ThermalMesh mesh =
        unit_shell({0.0, 0.25, 0.5, 0.75, 1.0}, {0.0, 1.0});

    const auto closed =
        intra_primitive_links(cylinder, mesh, TmmBuildOptions{});
    REQUIRE(closed.size() == 4U);  // three interior seams plus the wrap
    REQUIRE(link_value(closed, 3U, 0U) ==
            Catch::Approx(link_value(closed, 0U, 1U)));

    TmmBuildOptions open_options;
    open_options.close_full_revolution = false;
    const auto open = intra_primitive_links(cylinder, mesh, open_options);
    REQUIRE(open.size() == 3U);
    REQUIRE(link_value(open, 3U, 0U) == 0.0);
}

TEST_CASE("revolution links: a partial revolution never closes the ring",
          "[conduction][links]") {
    const Primitive cylinder = Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
                                        {1.0, 0.0, 0.0}, 1.0, 0.0, pi);
    const ThermalMesh mesh = unit_shell({0.0, 0.5, 1.0}, {0.0, 1.0});
    const auto links = intra_primitive_links(cylinder, mesh, TmmBuildOptions{});
    REQUIRE(links.size() == 1U);
}
