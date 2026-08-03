#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>

#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"

namespace {

using pycanha::gmm::ActiveSide;
using pycanha::gmm::BulkMaterial;
using pycanha::gmm::ThermalMesh;

}  // namespace

TEST_CASE("ThermalMesh default is a valid unit square", "[gmm][mesh]") {
    const ThermalMesh mesh;
    REQUIRE(mesh.is_valid());
    REQUIRE(mesh.get_dir1_mesh().size() == 2);
    REQUIRE(mesh.get_dir2_mesh().size() == 2);
    REQUIRE(mesh.get_number_of_pair_faces() == 1U);
}

TEST_CASE("ThermalMesh exposes ordered cuts and pair faces", "[gmm][mesh]") {
    ThermalMesh mesh({0.0, 0.25, 1.0}, {0.0, 0.5, 1.0});
    REQUIRE(mesh.get_dir1_mesh().size() == 3);
    REQUIRE(mesh.get_dir2_mesh().size() == 3);
    REQUIRE(mesh.get_number_of_pair_faces() == 4U);

    mesh.set_dir2_mesh({0.0, 1.0});
    REQUIRE(mesh.get_number_of_pair_faces() == 2U);
}

TEST_CASE("ThermalMesh rejects invalid cut definitions", "[gmm][mesh]") {
    REQUIRE_THROWS_AS(ThermalMesh({0.0, 0.9, 0.2, 1.0}, {0.0, 1.0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(ThermalMesh({0.1, 1.0}, {0.0, 1.0}),
                      std::invalid_argument);

    ThermalMesh mesh;
    REQUIRE_THROWS_AS(mesh.set_dir1_mesh({0.0}), std::invalid_argument);
    // A throwing setter leaves the mesh unchanged.
    REQUIRE(mesh.get_dir1_mesh().size() == 2);
    REQUIRE_THROWS_AS(mesh.set_side1_thick(-1.0), std::invalid_argument);
}

TEST_CASE("ThermalMesh node_of follows start + k * step", "[gmm][mesh]") {
    ThermalMesh mesh({0.0, 0.5, 1.0}, {0.0, 0.5, 1.0});  // 2x2 cells

    // Defaults: every face maps to NO_NODE (unassigned).
    REQUIRE(mesh.node_of(0U, 0U, 1U) == pycanha::gmm::NO_NODE);
    REQUIRE(mesh.node_of(1U, 1U, 2U) == pycanha::gmm::NO_NODE);

    mesh.set_node1_start(100);
    mesh.set_node1_step(1);
    mesh.set_node2_start(200);
    mesh.set_node2_step(0);

    // Direction 1 varies fastest: k = i + j * (n1 - 1), n1 - 1 == 2.
    REQUIRE(mesh.node_of(0U, 0U, 1U) == 100);  // k = 0
    REQUIRE(mesh.node_of(1U, 0U, 1U) == 101);  // k = 1
    REQUIRE(mesh.node_of(0U, 1U, 1U) == 102);  // k = 2
    REQUIRE(mesh.node_of(1U, 1U, 1U) == 103);  // k = 3
    // Side 2 with step 0: every face shares node 200.
    REQUIRE(mesh.node_of(0U, 0U, 2U) == 200);
    REQUIRE(mesh.node_of(1U, 1U, 2U) == 200);
}

TEST_CASE("ThermalMesh node_of validates side and cell range", "[gmm][mesh]") {
    const ThermalMesh mesh({0.0, 0.5, 1.0}, {0.0, 1.0});  // 2x1 cells
    REQUIRE_THROWS_AS(mesh.node_of(0U, 0U, 0U), std::invalid_argument);
    REQUIRE_THROWS_AS(mesh.node_of(0U, 0U, 3U), std::invalid_argument);
    REQUIRE_THROWS_AS(mesh.node_of(2U, 0U, 1U), std::invalid_argument);
    REQUIRE_THROWS_AS(mesh.node_of(0U, 1U, 1U), std::invalid_argument);
}

TEST_CASE("ThermalMesh materials default to nullptr", "[gmm][mesh]") {
    const ThermalMesh mesh;
    REQUIRE(mesh.get_side1_material() == nullptr);
    REQUIRE(mesh.get_side2_material() == nullptr);
    REQUIRE(mesh.get_side1_optical() == nullptr);
    REQUIRE(mesh.get_side2_optical() == nullptr);
}

TEST_CASE("ThermalMesh materials are shared by reference", "[gmm][mesh]") {
    auto aluminum =
        std::make_shared<BulkMaterial>("aluminum", 2700.0, 167.0, 896.0);
    ThermalMesh mesh;
    mesh.set_side1_material(aluminum);
    mesh.set_side2_material(aluminum);

    REQUIRE(mesh.get_side1_material() == aluminum);
    REQUIRE(mesh.get_side1_material() == mesh.get_side2_material());
    REQUIRE(mesh.get_side1_material()->get_density() == 2700.0);
}

TEST_CASE("ThermalMesh activity defaults to both sides in both physics",
          "[gmm][mesh]") {
    const ThermalMesh mesh;
    REQUIRE(mesh.get_radiative_active_side() == ActiveSide::Both);
    REQUIRE(mesh.get_conductive_active_side() == ActiveSide::Both);
    REQUIRE(mesh.is_radiative_active(1U));
    REQUIRE(mesh.is_radiative_active(2U));
    REQUIRE(mesh.is_conductive_active(1U));
    REQUIRE(mesh.is_conductive_active(2U));
    REQUIRE(mesh.is_side_active(1U));
    REQUIRE(mesh.is_side_active(2U));
}

TEST_CASE("ThermalMesh active-side selector covers the four states",
          "[gmm][mesh]") {
    ThermalMesh mesh;

    mesh.set_radiative_active_side(ActiveSide::None);
    REQUIRE_FALSE(mesh.is_radiative_active(1U));
    REQUIRE_FALSE(mesh.is_radiative_active(2U));

    mesh.set_radiative_active_side(ActiveSide::Side1);
    REQUIRE(mesh.is_radiative_active(1U));
    REQUIRE_FALSE(mesh.is_radiative_active(2U));

    mesh.set_radiative_active_side(ActiveSide::Side2);
    REQUIRE_FALSE(mesh.is_radiative_active(1U));
    REQUIRE(mesh.is_radiative_active(2U));

    mesh.set_radiative_active_side(ActiveSide::Both);
    REQUIRE(mesh.is_radiative_active(1U));
    REQUIRE(mesh.is_radiative_active(2U));
}

TEST_CASE("ThermalMesh radiative and conductive activity are independent",
          "[gmm][mesh]") {
    ThermalMesh mesh;
    // The two ESATAN states a single selector could not express: side 1 only
    // radiates, side 2 only conducts.
    mesh.set_radiative_active_side(ActiveSide::Side1);
    mesh.set_conductive_active_side(ActiveSide::Side2);

    REQUIRE(mesh.is_radiative_active(1U));
    REQUIRE_FALSE(mesh.is_conductive_active(1U));
    REQUIRE_FALSE(mesh.is_radiative_active(2U));
    REQUIRE(mesh.is_conductive_active(2U));
    REQUIRE(mesh.is_side_active(1U));
    REQUIRE(mesh.is_side_active(2U));

    mesh.set_radiative_active_side(ActiveSide::Side2);
    mesh.set_conductive_active_side(ActiveSide::Side2);
    REQUIRE_FALSE(mesh.is_side_active(1U));
    REQUIRE(mesh.is_side_active(2U));
}

TEST_CASE("ThermalMesh activity predicates validate the side", "[gmm][mesh]") {
    const ThermalMesh mesh;
    REQUIRE_THROWS_AS(mesh.is_radiative_active(0U), std::invalid_argument);
    REQUIRE_THROWS_AS(mesh.is_conductive_active(3U), std::invalid_argument);
    REQUIRE_THROWS_AS(mesh.is_side_active(0U), std::invalid_argument);
}
