#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/item.hpp"

namespace {

using pycanha::gmm::FaceId;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::Item;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;

[[nodiscard]] Item make_item() {
    return {Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
            ThermalMesh{}};
}

[[nodiscard]] std::vector<std::uint64_t> as_raw(std::span<const FaceId> faces) {
    std::vector<std::uint64_t> values;
    values.reserve(faces.size());
    std::transform(
        faces.begin(), faces.end(), std::back_inserter(values),
        [](const FaceId face_id) { return pycanha::gmm::to_raw(face_id); });
    std::sort(values.begin(), values.end());
    return values;
}

}  // namespace

TEST_CASE("GeometryModel face-to-node mapping round-trips",
          "[gmm][geometrymodel][mapping]") {
    GeometryModel model("scene");
    const auto face_x = static_cast<FaceId>(2U);
    const auto face_y = static_cast<FaceId>(4U);
    const auto face_z = static_cast<FaceId>(6U);

    model.assign_face_to_node(face_x, 42);
    model.assign_face_to_node(face_y, 42);
    model.assign_face_to_node(face_z, 7);

    REQUIRE(model.face_to_node(face_x) ==
            std::optional<pycanha::NodeNum>{42});
    REQUIRE(model.face_to_node(face_y) ==
            std::optional<pycanha::NodeNum>{42});
    REQUIRE(model.face_to_node(face_z) ==
            std::optional<pycanha::NodeNum>{7});
    REQUIRE_FALSE(model.face_to_node(static_cast<FaceId>(8U)).has_value());

    REQUIRE(as_raw(model.faces_of_node(42)) ==
            std::vector<std::uint64_t>{2U, 4U});
    REQUIRE(as_raw(model.faces_of_node(7)) == std::vector<std::uint64_t>{6U});
    REQUIRE(model.faces_of_node(99).empty());
}

TEST_CASE("GeometryModel face-to-node reverse cache refreshes on mutation",
          "[gmm][geometrymodel][mapping]") {
    GeometryModel model("scene");
    model.assign_face_to_node(static_cast<FaceId>(2U), 42);
    REQUIRE(as_raw(model.faces_of_node(42)) == std::vector<std::uint64_t>{2U});

    model.assign_face_to_node(static_cast<FaceId>(4U), 42);
    REQUIRE(as_raw(model.faces_of_node(42)) ==
            std::vector<std::uint64_t>{2U, 4U});

    model.add_item("panel", make_item());
    REQUIRE(as_raw(model.faces_of_node(42)) ==
            std::vector<std::uint64_t>{2U, 4U});
}
