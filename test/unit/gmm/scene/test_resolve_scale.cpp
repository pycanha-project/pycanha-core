#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::gmm::Cube;
using pycanha::gmm::Geometry;
using pycanha::gmm::GeometryGroup;
using pycanha::gmm::GeometryGroupCutted;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriMeshD;

// `count` uncut plates side by side, so the only cost that grows is the
// assembly of their meshes into one.
[[nodiscard]] GeometryGroup make_row_of_plates(std::size_t count) {
    std::vector<std::shared_ptr<Geometry>> children;
    children.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const double origin = static_cast<double>(index) * 2.0;
        children.push_back(std::make_shared<GeometryItem>(
            "plate_" + std::to_string(index),
            Rectangle({origin, 0.0, 0.0}, {origin + 1.0, 0.0, 0.0},
                      {origin, 1.0, 0.0}),
            ThermalMesh{}));
    }
    return GeometryGroup{"row", std::move(children)};
}

}  // namespace

// Eigen has no capacity concept, so growing an array with conservativeResize
// reallocates and copies everything already in it. Appending child meshes one
// at a time therefore moved O(n^2) bytes to assemble n children -- worst
// exactly where it hurts, on the large refined models the raytracer is for.
// Sizing the arrays once from a first pass makes it linear.
//
// This asserts the RESULT of assembling many pieces, not its speed. A
// wall-clock guard was tried twice and abandoned, and the measurements are
// recorded here so nobody re-adds one naively:
//
//   - Comparing 500 -> 2000 items read 5.6x in Debug but 13.8x in Release, on
//     code that is provably linear: at 500 items a Release build spends
//     0.2 ms, so the "baseline" was cache noise.
//   - Moving both sizes past the cache cliff (2000 -> 16000) gave a stable
//     ~8x in Debug AND in a plain Release build -- and then 30x inside the
//     packaging build, where LTO makes 2000 items take 0.62 ms instead of
//     12.7 ms and drops the baseline back under the cliff.
//
// The cliff is at a fixed problem SIZE, but the time that size costs moves by
// more than an order of magnitude with build flags, so no fixed pair of sizes
// is safe. Separating linear from quadratic needs a spread wide enough that
// cache effects cannot masquerade as growth (~32x), which costs more suite
// time than the guard is worth. If this regression needs a real guard, count
// allocations rather than seconds.
TEST_CASE("Mesh assembly of many items is correct and single-pass",
          "[gmm][scene][resolve]") {
    constexpr std::size_t count = 16000;
    GeometryGroup row = make_row_of_plates(count);
    const TriMeshD& mesh = row.mesh();

    // One pair per plate, two triangles and four vertices each, all assembled
    // in one sizing pass.
    REQUIRE(mesh.nf() == static_cast<pycanha::MeshIndex>(count) * 2U);
    REQUIRE(mesh.triangles.rows() == static_cast<Eigen::Index>(count) * 2);
    REQUIRE(mesh.vertices.rows() == static_cast<Eigen::Index>(count) * 4);
    REQUIRE(mesh.face_ids.rows() == mesh.triangles.rows());
    REQUIRE(mesh.node_numbers.rows() == static_cast<Eigen::Index>(mesh.nf()));
    // Face ids are contiguous and even across every piece.
    REQUIRE(mesh.face_ids.minCoeff() == 0U);
    REQUIRE(mesh.face_ids.maxCoeff() ==
            static_cast<pycanha::MeshIndex>(count - 1U) * 2U);
}

// The boolean cuts run on several threads, each writing only its own face, so
// the assembled mesh must not depend on how the work was handed out.
TEST_CASE("Resolving the same subtree twice gives the same mesh",
          "[gmm][scene][resolve]") {
    std::vector<std::shared_ptr<Geometry>> plates;
    std::vector<std::shared_ptr<GeometryItem>> cutters;
    for (std::size_t index = 0; index < 16U; ++index) {
        const double origin = static_cast<double>(index) * 4.0;
        plates.push_back(std::make_shared<GeometryItem>(
            "plate_" + std::to_string(index),
            Rectangle({origin, 0.0, 0.0}, {origin + 3.0, 0.0, 0.0},
                      {origin, 1.0, 0.0}),
            ThermalMesh{}));
        cutters.push_back(std::make_shared<GeometryItem>(
            "cut_" + std::to_string(index),
            Cube({origin + 1.5, 0.5, 0.0}, {1.0, 2.0, 2.0}), ThermalMesh{}));
    }
    GeometryGroupCutted cut("cut", std::move(plates), std::move(cutters));

    // Deliberate copies: mesh() hands back a reference to storage the next
    // call overwrites, so holding references would compare a result to itself.
    TriMeshD first = cut.mesh();
    TriMeshD second = cut.mesh();

    REQUIRE(first.vertices.rows() == second.vertices.rows());
    REQUIRE(first.nf() == second.nf());
    REQUIRE(first.vertices == second.vertices);
    REQUIRE(first.triangles == second.triangles);
    REQUIRE(first.face_ids == second.face_ids);
}
