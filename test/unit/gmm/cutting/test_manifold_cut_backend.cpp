#include <array>
#include <catch2/catch_test_macros.hpp>
#include <numbers>
#include <span>

#include "pycanha-core/gmm/cutting/manifold_cut_backend.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/ops/validate.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cylinder;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
namespace cutting = pycanha::gmm::cutting;
namespace mesh_ops = pycanha::gmm::mesh::ops;

}  // namespace

TEST_CASE("ManifoldCutBackend cuts a rectangle with a cylinder",
          "[gmm][cutting]") {
    const cutting::ManifoldCutBackend backend;
    const GeometryItem panel(
        "panel", Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}),
        ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}});
    const Cylinder cutter({1.0, 1.0, -1.0}, {1.0, 1.0, 1.0}, {1.35, 1.0, -1.0},
                          0.35, 0.0, 2.0 * std::numbers::pi);
    const std::array<pycanha::gmm::Primitive, 1> cutters{cutter};

    const auto mesh =
        backend.cut(panel, std::span<const pycanha::gmm::Primitive>{cutters},
                    CoordinateTransformation{}, pycanha::gmm::MeshOptions{});

    REQUIRE(mesh.vertices.rows() > 0);
    REQUIRE(mesh.triangles.rows() > 0);
    REQUIRE_FALSE(mesh_ops::is_watertight(mesh));
    REQUIRE(mesh_ops::has_consistent_face_ids(mesh));
    REQUIRE(mesh_ops::compute_areas(mesh).sum() < 4.0);
}
