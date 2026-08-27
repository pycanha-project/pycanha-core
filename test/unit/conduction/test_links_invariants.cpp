#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <numbers>
#include <numeric>
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
#include "pycanha-core/gmm/primitives/rectangle.hpp"
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
using pycanha::gmm::Rectangle;
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

// Resistance of the whole chain of direction-2 links of the single column of
// face pairs, walked in order.
[[nodiscard]] double series_resistance(const std::vector<FacePairLink>& links) {
    return std::transform_reduce(
        links.begin(), links.end(), 0.0, std::plus<>{},
        [](const FacePairLink& link) { return 1.0 / link.conductance; });
}

// The chain must telescope: the half-resistances that meet at each interior
// edge add up to the analytic resistance between the two end reference lines,
// whatever the cut vector in between. This is the property that makes a
// refined mesh give the same end-to-end answer as a coarse one.
void check_telescoping(const Primitive& primitive,
                       const std::vector<double>& dir2_cuts) {
    const ThermalMesh mesh = unit_shell({0.0, 1.0}, dir2_cuts);
    const auto links =
        intra_primitive_links(primitive, mesh, TmmBuildOptions{});
    REQUIRE(links.size() == dir2_cuts.size() - 2U);

    const std::optional<MeridianProfile> profile = profile_of(primitive);
    if (!profile.has_value()) {
        throw std::logic_error(
            "primitive has no closed-form conduction profile");
    }
    const double first_reference = 0.5 * (dir2_cuts[0] + dir2_cuts[1]);
    const double last_reference =
        0.5 * (dir2_cuts[dir2_cuts.size() - 2U] + dir2_cuts.back());
    const double expected = (profile.value().potential(last_reference) -
                             profile.value().potential(first_reference)) /
                            profile.value().dir1_extent();

    REQUIRE(series_resistance(links) == Catch::Approx(expected).epsilon(1e-12));
}

}  // namespace

TEST_CASE("link invariants: a rectangle chain telescopes",
          "[conduction][links]") {
    const Primitive rectangle =
        Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 5.0, 0.0});
    check_telescoping(rectangle, {0.0, 0.5, 1.0});
    check_telescoping(rectangle, {0.0, 0.05, 0.4, 0.42, 0.9, 1.0});
}

TEST_CASE("link invariants: every revolution primitive telescopes",
          "[conduction][links]") {
    const std::vector<double> skewed{0.0, 0.03, 0.31, 0.33, 0.78, 1.0};
    const std::vector<double> uniform{0.0, 0.25, 0.5, 0.75, 1.0};

    const Primitive disc = Disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
                                {2.0, 0.0, 0.0}, 0.3, 2.0, 0.0, pi / 3.0);
    check_telescoping(disc, skewed);
    check_telescoping(disc, uniform);

    const Primitive cylinder = Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 3.0},
                                        {1.1, 0.0, 0.0}, 1.1, 0.0, pi);
    check_telescoping(cylinder, skewed);

    const Primitive cone = Cone({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0},
                                {1.5, 0.0, 0.0}, 0.4, 1.5, 0.0, pi / 2.0);
    check_telescoping(cone, skewed);

    const Primitive sphere = Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
                                    {1.0, 0.0, 0.0}, 1.0, -0.6, 0.7, 0.0, pi);
    check_telescoping(sphere, skewed);

    const Primitive paraboloid = Paraboloid(
        {0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {1.0, 0.0, 0.0}, 1.0, 0.0, pi / 4.0);
    check_telescoping(paraboloid, skewed);
}

TEST_CASE("link invariants: refining a mesh keeps the end-to-end resistance",
          "[conduction][links]") {
    const Primitive cone = Cone({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0},
                                {1.5, 0.0, 0.0}, 0.4, 1.5, 0.0, pi / 2.0);

    // Same first and last face pairs, different subdivision in between: the
    // resistance between the two end reference lines cannot move.
    const std::vector<double> coarse{0.0, 0.2, 0.8, 1.0};
    const std::vector<double> fine{0.0, 0.2, 0.35, 0.5, 0.62, 0.8, 1.0};

    const auto coarse_links = intra_primitive_links(
        cone, unit_shell({0.0, 1.0}, coarse), TmmBuildOptions{});
    const auto fine_links = intra_primitive_links(
        cone, unit_shell({0.0, 1.0}, fine), TmmBuildOptions{});
    REQUIRE(series_resistance(coarse_links) ==
            Catch::Approx(series_resistance(fine_links)).epsilon(1e-12));
}

TEST_CASE("link invariants: mirroring the mesh mirrors the conductors",
          "[conduction][links]") {
    const Primitive rectangle =
        Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0});
    const std::vector<double> cuts{0.0, 0.1, 0.4, 1.0};
    std::vector<double> mirrored;
    mirrored.reserve(cuts.size());
    for (std::size_t index = cuts.size(); index > 0U; --index) {
        mirrored.push_back(1.0 - cuts[index - 1U]);
    }

    const auto links = intra_primitive_links(
        rectangle, unit_shell(cuts, {0.0, 1.0}), TmmBuildOptions{});
    const auto mirrored_links = intra_primitive_links(
        rectangle, unit_shell(mirrored, {0.0, 1.0}), TmmBuildOptions{});
    REQUIRE(links.size() == mirrored_links.size());
    for (std::size_t index = 0; index < links.size(); ++index) {
        REQUIRE(links[index].conductance ==
                Catch::Approx(
                    mirrored_links[links.size() - 1U - index].conductance));
    }
}

TEST_CASE("link invariants: a degenerate face pair carries no conductor",
          "[conduction][links]") {
    const Primitive rectangle =
        Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    // A repeated cut leaves a zero-width face pair; a zero distance would
    // otherwise make the conductance unbounded.
    const auto links = intra_primitive_links(
        rectangle, unit_shell({0.0, 0.5, 0.5, 1.0}, {0.0, 1.0}),
        TmmBuildOptions{});
    for (const auto& link : links) {
        REQUIRE(std::isfinite(link.conductance));
    }
}
