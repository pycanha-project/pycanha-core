#include <Eigen/Dense>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
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

[[nodiscard]] double seconds_to_assemble(std::size_t count) {
    GeometryGroup row = make_row_of_plates(count);
    row.create_mesh();  // warm the per-item caches; time only the assembly
    const auto start = std::chrono::steady_clock::now();
    const Eigen::Index triangles = row.mesh().triangles.rows();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    REQUIRE(triangles == static_cast<Eigen::Index>(count) * 2);
    return std::chrono::duration<double>(elapsed).count();
}

}  // namespace

// Eigen has no capacity concept, so growing an array with conservativeResize
// reallocates and copies everything already in it. Appending child meshes one
// at a time therefore moved O(n^2) bytes to assemble n children -- worst
// exactly where it hurts, on the large refined models the raytracer is for.
// Sizing the arrays once from a first pass makes it linear.
TEST_CASE("Mesh assembly stays linear in the number of items",
          "[gmm][scene][resolve]") {
    constexpr std::size_t small = 500;
    constexpr std::size_t large = 4 * small;

    // Discard a first run: it pays for whatever the allocator has to warm up.
    static_cast<void>(seconds_to_assemble(small));
    const double small_seconds = seconds_to_assemble(small);
    const double large_seconds = seconds_to_assemble(large);

    // Quadratic growth would be ~16x for 4x the items; linear measures ~5x
    // here, the extra coming from cache behaviour rather than from the
    // algorithm. The bound sits between the two with room for a loaded
    // machine: this is a shape check on a timing measurement, not a benchmark.
    INFO("small: " << small_seconds << " s, large: " << large_seconds << " s");
    REQUIRE(large_seconds < 10.0 * small_seconds);
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
