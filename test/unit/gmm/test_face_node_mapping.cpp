#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::gmm::FaceId;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;

[[nodiscard]] std::vector<std::uint32_t> as_raw(std::span<const FaceId> faces) {
    std::vector<std::uint32_t> values;
    values.reserve(faces.size());
    std::ranges::transform(faces, std::back_inserter(values),
                           [](const FaceId face_id) {
                               return static_cast<std::uint32_t>(face_id);
                           });
    std::ranges::sort(values);
    return values;
}

// Two dir1 cells, one dir2 cell. side1 node = 100 + k, side2 node = 7 + k.
[[nodiscard]] std::shared_ptr<GeometryItem> make_panel() {
    ThermalMesh thermal_mesh{{0.0, 0.5, 1.0}, {0.0, 1.0}};
    thermal_mesh.set_node1_start(100);
    thermal_mesh.set_node1_step(1);
    thermal_mesh.set_node2_start(7);
    thermal_mesh.set_node2_step(1);
    return std::make_shared<GeometryItem>(
        "panel", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        std::move(thermal_mesh));
}

}  // namespace

TEST_CASE("GeometryModel reverse node -> face_ids from node_numbers",
          "[gmm][geometrymodel][mapping]") {
    GeometryModel model("scene");
    model.add(make_panel());
    model.create_mesh();

    // Cell k=0 -> face_id 0 (side1) / 1 (side2); cell k=1 -> 2 / 3.
    REQUIRE(as_raw(model.faces_of_node(100)) == std::vector<std::uint32_t>{0U});
    REQUIRE(as_raw(model.faces_of_node(101)) == std::vector<std::uint32_t>{2U});
    REQUIRE(as_raw(model.faces_of_node(7)) == std::vector<std::uint32_t>{1U});
    REQUIRE(as_raw(model.faces_of_node(8)) == std::vector<std::uint32_t>{3U});
    REQUIRE(model.faces_of_node(99).empty());
}

TEST_CASE("GeometryModel reverse mapping refreshes after structural change",
          "[gmm][geometrymodel][mapping]") {
    GeometryModel model("scene");
    model.add(make_panel());
    model.create_mesh();
    REQUIRE_FALSE(model.faces_of_node(100).empty());

    model.remove("panel");
    model.create_mesh();
    REQUIRE(model.faces_of_node(100).empty());
}
