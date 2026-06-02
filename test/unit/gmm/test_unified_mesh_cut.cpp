#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <numbers>
#include <utility>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/ops/boundary_edges.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/cut_group.hpp"
#include "pycanha-core/gmm/scene/item.hpp"

TEST_CASE("GeometryModel applies CutGroup cutters during unified mesh build",
          "[gmm][geometrymodel][cutting]") {
    pycanha::gmm::CutGroup trim;
    trim.add_cutter(pycanha::gmm::Cylinder({1.0, 1.0, -1.0}, {1.0, 1.0, 1.0},
                                           {1.35, 1.0, -1.0}, 0.35, 0.0,
                                           2.0 * std::numbers::pi));

    pycanha::gmm::GeometryModel model("scene");
    model.add_cut_group("trim", std::move(trim));
    const auto panel_id = model.add_item(
        "panel",
        pycanha::gmm::Item(
            pycanha::gmm::Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
                                    {0.0, 2.0, 0.0}),
            pycanha::gmm::ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}),
        "trim");

    const auto& mesh = model.unified_mesh();
    const pycanha::gmm::TriMesh tri_mesh{mesh.vertices, mesh.triangles,
                                         mesh.face_ids};
    const auto loops = pycanha::gmm::mesh::ops::boundary_edge_loops(tri_mesh);

    REQUIRE(mesh.vertices.rows() > 0);
    REQUIRE(mesh.triangles.rows() > 0);
    REQUIRE_FALSE(loops.empty());
    REQUIRE(std::all_of(mesh.geometry_ids.begin(), mesh.geometry_ids.end(),
                        [panel_id](std::uint64_t geometry_id) {
                            return geometry_id ==
                                   pycanha::gmm::to_raw(panel_id);
                        }));
    REQUIRE(pycanha::gmm::mesh::ops::compute_areas(tri_mesh).sum() < 4.0);
}
