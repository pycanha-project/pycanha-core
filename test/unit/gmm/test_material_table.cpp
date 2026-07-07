#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/materials/optical_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/materials.hpp"

namespace {

using pycanha::gmm::GeometryItem;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::OpticalMaterial;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;

[[nodiscard]] std::shared_ptr<GeometryItem> make_panel(
    const std::string& name, ThermalMesh thermal_mesh) {
    return std::make_shared<GeometryItem>(
        name, Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        std::move(thermal_mesh));
}

// panel: side1 = white_paint, side2 = black (side2 inactive);
// shared: white_paint on both sides; bare: no optical material.
struct Fixture {
    std::shared_ptr<OpticalMaterial> white_paint =
        std::make_shared<OpticalMaterial>(
            "white_paint",
            OpticalMaterial::Properties{0.9, 0.05, 0.0, 0.2, 0.1, 0.0});
    std::shared_ptr<OpticalMaterial> black =
        std::make_shared<OpticalMaterial>("black", 0.95, 0.95);
    GeometryModel model{"scene"};

    Fixture() {
        ThermalMesh panel_mesh;  // one face pair -> slots 0/1
        panel_mesh.set_side1_optical(white_paint);
        panel_mesh.set_side2_optical(black);
        panel_mesh.set_side2_activity(/*activity=*/false);
        model.add(make_panel("panel", std::move(panel_mesh)));

        ThermalMesh shared_mesh;  // slots 2/3
        shared_mesh.set_side1_optical(white_paint);
        shared_mesh.set_side2_optical(white_paint);
        model.add(make_panel("shared", std::move(shared_mesh)));

        model.add(make_panel("bare", ThermalMesh{}));  // slots 4/5
    }
};

}  // namespace

TEST_CASE("material_table: unique materials deduplicated", "[gmm][materials]") {
    const Fixture fixture;
    const auto table = fixture.model.material_table();

    REQUIRE(table.num_materials() == 2);
    // The shared material maps every user to the same row.
    REQUIRE(table.face_material[0] == table.face_material[2]);
    REQUIRE(table.face_material[2] == table.face_material[3]);
}

TEST_CASE("material_table: row layout matches OpticalMaterial",
          "[gmm][materials]") {
    const Fixture fixture;
    const auto table = fixture.model.material_table();

    const auto white_row = table.face_material[0];
    const auto& expected = fixture.white_paint->get_th_optical_properties();
    for (int dof = 0; dof < 6; ++dof) {
        REQUIRE(table.properties(white_row, dof) ==
                Catch::Approx(expected.at(static_cast<std::size_t>(dof))));
    }
}

TEST_CASE("material_table: per-face-slot indices", "[gmm][materials]") {
    const Fixture fixture;
    const auto table = fixture.model.material_table();

    const auto white_row = table.face_material[0];
    const auto black_row = table.face_material[1];
    REQUIRE(white_row >= 0);
    REQUIRE(black_row >= 0);
    REQUIRE(white_row != black_row);
    REQUIRE(table.properties(white_row, 0) == Catch::Approx(0.9));
    REQUIRE(table.properties(black_row, 0) == Catch::Approx(0.95));

    // Missing material -> -1 (and a logged warning).
    REQUIRE(table.face_material[4] == -1);
    REQUIRE(table.face_material[5] == -1);
}

TEST_CASE("material_table: activity flags", "[gmm][materials]") {
    const Fixture fixture;
    const auto table = fixture.model.material_table();

    REQUIRE(table.face_active[0]);
    REQUIRE_FALSE(table.face_active[1]);  // side2_activity = false
    REQUIRE(table.face_active[2]);
    REQUIRE(table.face_active[3]);
}

TEST_CASE("material_table: table tracks the mesh", "[gmm][materials]") {
    const Fixture fixture;
    const auto table = fixture.model.material_table();
    const auto num_slots = static_cast<Eigen::Index>(fixture.model.mesh().nf());

    REQUIRE(table.face_material.rows() == num_slots);
    REQUIRE(table.face_active.rows() == num_slots);
    REQUIRE(num_slots == 6);
}
