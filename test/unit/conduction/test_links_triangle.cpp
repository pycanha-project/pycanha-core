#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "pycanha-core/conduction/links.hpp"
#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"

namespace {

using pycanha::conduction::FacePairLink;
using pycanha::conduction::intra_primitive_links;
using pycanha::conduction::TmmBuildOptions;
using pycanha::gmm::ActiveSide;
using pycanha::gmm::BulkMaterial;
using pycanha::gmm::Primitive;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::Triangle;

[[nodiscard]] ThermalMesh shell(std::vector<double> dir1,
                                std::vector<double> dir2,
                                double conductivity = 1.0) {
    ThermalMesh mesh(std::move(dir1), std::move(dir2));
    mesh.set_side1_material(
        std::make_shared<BulkMaterial>("m", 1.0, conductivity, 1.0));
    mesh.set_side1_thick(1.0);
    mesh.set_conductive_active_side(ActiveSide::Side1);
    return mesh;
}

[[nodiscard]] double link_value(const std::vector<FacePairLink>& links,
                                pycanha::MeshIndex face_pair_a,
                                pycanha::MeshIndex face_pair_b) {
    for (const auto& link : links) {
        const bool matches = (link.face_pair_a == face_pair_a &&
                              link.face_pair_b == face_pair_b) ||
                             (link.face_pair_a == face_pair_b &&
                              link.face_pair_b == face_pair_a);
        if (matches) {
            return link.conductance;
        }
    }
    return 0.0;
}

}  // namespace

TEST_CASE("triangle links: the fan fallback produces the full grid",
          "[conduction][links]") {
    const Primitive triangle =
        Triangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    const auto links = intra_primitive_links(
        triangle, shell({0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}), TmmBuildOptions{});

    // 2 x 2 face pairs: one link per row along direction 1, one per column
    // along direction 2.
    REQUIRE(links.size() == 4U);
    for (const auto& link : links) {
        REQUIRE(std::isfinite(link.conductance));
        REQUIRE(link.conductance > 0.0);
    }
}

TEST_CASE("triangle links: a symmetric triangle gives symmetric conductors",
          "[conduction][links]") {
    // Swapping the two edges of this right isoceles triangle maps the blend
    // parameter w onto 1 - w, so mirrored face pairs must come out equal.
    const Primitive triangle =
        Triangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    const auto links = intra_primitive_links(
        triangle, shell({0.0, 0.4, 1.0}, {0.0, 0.5, 1.0}), TmmBuildOptions{});

    // Face pair k = i + j * 2: the two fan links (0, 1) and (2, 3) are mirror
    // images, and so are the two blend links (0, 2) and (1, 3) of each column.
    REQUIRE(link_value(links, 0U, 1U) ==
            Catch::Approx(link_value(links, 2U, 3U)));
}

TEST_CASE("triangle links: the apex row conducts outward",
          "[conduction][links]") {
    const Primitive triangle =
        Triangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0});
    // The row at fan parameter 0 collapses to the apex, so the innermost face
    // pairs are triangles rather than quadrilaterals; the discrete rule handles
    // them like any other face pair.
    const auto links = intra_primitive_links(
        triangle, shell({0.0, 0.25, 1.0}, {0.0, 1.0}), TmmBuildOptions{});
    REQUIRE(links.size() == 1U);
    REQUIRE(link_value(links, 0U, 1U) > 0.0);
}

TEST_CASE(
    "triangle links: conductance scales with conductivity times thickness",
    "[conduction][links]") {
    const Primitive triangle =
        Triangle({0.0, 0.0, 0.0}, {1.5, 0.0, 0.0}, {0.0, 1.0, 0.0});
    const auto base = intra_primitive_links(
        triangle, shell({0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}, 1.0),
        TmmBuildOptions{});
    const auto scaled = intra_primitive_links(
        triangle, shell({0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}, 3.0),
        TmmBuildOptions{});

    REQUIRE(base.size() == scaled.size());
    for (std::size_t index = 0; index < base.size(); ++index) {
        REQUIRE(scaled[index].conductance ==
                Catch::Approx(3.0 * base[index].conductance));
    }
}

TEST_CASE("triangle links: refining the mesh raises the total conductance",
          "[conduction][links]") {
    const Primitive triangle =
        Triangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    // Halving the distance between references while keeping the same shared
    // edge doubles the conductance across it.
    const auto wide = intra_primitive_links(
        triangle, shell({0.0, 0.5, 1.0}, {0.0, 1.0}), TmmBuildOptions{});
    const auto narrow = intra_primitive_links(
        triangle, shell({0.0, 0.25, 0.5, 0.75, 1.0}, {0.0, 1.0}),
        TmmBuildOptions{});
    REQUIRE(link_value(narrow, 1U, 2U) > link_value(wide, 0U, 1U));
}
