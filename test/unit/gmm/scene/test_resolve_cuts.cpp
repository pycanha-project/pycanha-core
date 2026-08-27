#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/ops/validate.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cube;
using pycanha::gmm::Geometry;
using pycanha::gmm::GeometryGroup;
using pycanha::gmm::GeometryGroupCutted;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriMeshD;
namespace mesh_ops = pycanha::gmm::mesh::ops;

// A 4 m x 1 m plate along +x, meshed as one face pair.
[[nodiscard]] std::shared_ptr<GeometryItem> make_plate() {
    return std::make_shared<GeometryItem>(
        "plate", Rectangle({0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{});
}

// A box tall enough to cut clean through the plate, 1 m wide in x, centred on
// `x_center`. Each one removes exactly 1 m2.
[[nodiscard]] std::shared_ptr<GeometryItem> make_cutter(const char* name,
                                                        double x_center) {
    return std::make_shared<GeometryItem>(
        name, Cube({x_center, 0.5, 0.0}, {1.0, 2.0, 2.0}), ThermalMesh{});
}

[[nodiscard]] double area_of(const Geometry& node) {
    return mesh_ops::compute_areas(node.mesh()).sum();
}

}  // namespace

// The failure this whole design exists for. A cut result is a triangle soup
// with no primitive, and the backend re-classifies every surviving triangle
// back onto the target's primitive -- so a cut of a cut is impossible by
// construction. Resolving from the top instead turns the chain into one
// operation on the original primitive.
TEST_CASE("A chain of cuts resolves as one cut of the primitive",
          "[gmm][scene][resolve]") {
    auto plate = make_plate();
    auto inner = std::make_shared<GeometryGroupCutted>(
        "inner", std::vector<std::shared_ptr<Geometry>>{plate},
        std::vector<std::shared_ptr<GeometryItem>>{make_cutter("cut_1", 0.5)});
    auto outer = std::make_shared<GeometryGroupCutted>(
        "outer", std::vector<std::shared_ptr<Geometry>>{inner},
        std::vector<std::shared_ptr<GeometryItem>>{make_cutter("cut_2", 2.5)});

    REQUIRE(area_of(*outer) == Catch::Approx(2.0).margin(1e-6));
    REQUIRE(mesh_ops::has_consistent_face_ids(outer->mesh()));
}

// node.mesh() means "this subtree resolved with node as the resolution root",
// so only cutters INSIDE the subtree apply. That is what makes a submodel
// meaningful: the same plate legitimately resolves differently under two
// different roots.
TEST_CASE("Only cutters inside the subtree apply", "[gmm][scene][resolve]") {
    auto plate = make_plate();
    auto inner = std::make_shared<GeometryGroupCutted>(
        "inner", std::vector<std::shared_ptr<Geometry>>{plate},
        std::vector<std::shared_ptr<GeometryItem>>{make_cutter("cut_1", 0.5)});
    auto outer = std::make_shared<GeometryGroupCutted>(
        "outer", std::vector<std::shared_ptr<Geometry>>{inner},
        std::vector<std::shared_ptr<GeometryItem>>{make_cutter("cut_2", 2.5)});

    // Resolved at `inner`, cut_2 is out of scope: only 1 m2 is gone.
    REQUIRE(area_of(*inner) == Catch::Approx(3.0).margin(1e-6));
    // Resolved at `outer`, both apply.
    REQUIRE(area_of(*outer) == Catch::Approx(2.0).margin(1e-6));
    // So a parent's mesh is NOT the concatenation of its children's meshes.
    REQUIRE(area_of(*outer) != Catch::Approx(area_of(*inner)));
}

// Two cutters on one group and the same two spread over nested groups describe
// the same shape, so they must produce the same mesh.
TEST_CASE("Nesting depth does not change the result", "[gmm][scene][resolve]") {
    auto flat_plate = make_plate();
    auto flat = std::make_shared<GeometryGroupCutted>(
        "flat", std::vector<std::shared_ptr<Geometry>>{flat_plate},
        std::vector<std::shared_ptr<GeometryItem>>{make_cutter("c1", 0.5),
                                                   make_cutter("c2", 2.5)});

    auto nested_plate = make_plate();
    auto nested_inner = std::make_shared<GeometryGroupCutted>(
        "nested_inner", std::vector<std::shared_ptr<Geometry>>{nested_plate},
        std::vector<std::shared_ptr<GeometryItem>>{make_cutter("c3", 0.5)});
    auto nested = std::make_shared<GeometryGroupCutted>(
        "nested", std::vector<std::shared_ptr<Geometry>>{nested_inner},
        std::vector<std::shared_ptr<GeometryItem>>{make_cutter("c4", 2.5)});

    const TriMeshD& flat_mesh = flat->mesh();
    const TriMeshD& nested_mesh = nested->mesh();
    REQUIRE(flat_mesh.vertices.rows() == nested_mesh.vertices.rows());
    REQUIRE(flat_mesh.triangles.rows() == nested_mesh.triangles.rows());
    REQUIRE(flat_mesh.nf() == nested_mesh.nf());
    REQUIRE(flat_mesh.vertices.isApprox(nested_mesh.vertices));
}

// A cut group may now hold groups, not just items: the restriction was never a
// rule, only the shape of a backend call that needed a primitive to cut.
TEST_CASE("A cut group can hold a group of items", "[gmm][scene][resolve]") {
    auto left = std::make_shared<GeometryItem>(
        "left", Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{});
    auto right = std::make_shared<GeometryItem>(
        "right", Rectangle({2.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {2.0, 1.0, 0.0}),
        ThermalMesh{});
    auto both = std::make_shared<GeometryGroup>(
        "both", std::vector<std::shared_ptr<Geometry>>{left, right});

    auto cut = std::make_shared<GeometryGroupCutted>(
        "cut", std::vector<std::shared_ptr<Geometry>>{both},
        // Straddles the seam, so it takes 0.5 m2 from each plate.
        std::vector<std::shared_ptr<GeometryItem>>{make_cutter("c", 2.0)});

    REQUIRE(area_of(*cut) == Catch::Approx(3.0).margin(1e-6));
    // Both items still contribute their own faces, in tree order.
    REQUIRE(cut->mesh().primitives.size() == 2U);
    REQUIRE(cut->mesh().primitives[0].geometry_id == left->id());
    REQUIRE(cut->mesh().primitives[1].geometry_id == right->id());
}

// The cutter's placement is the composition of every transform between it and
// the resolution root, and so is the target's. Moving an ancestor must move
// both together.
TEST_CASE("A cut under a transformed ancestor lands where the transform says",
          "[gmm][scene][resolve]") {
    auto plate = make_plate();
    auto cut = std::make_shared<GeometryGroupCutted>(
        "cut", std::vector<std::shared_ptr<Geometry>>{plate},
        std::vector<std::shared_ptr<GeometryItem>>{make_cutter("c", 0.5)});
    // The whole cut group is shifted; the cutter goes with it, so the hole
    // stays in the same place ON the plate and the area is unchanged.
    GeometryGroup rig(
        "rig", {cut},
        CoordinateTransformation::from_translation({10.0, 0.0, 0.0}));

    REQUIRE(area_of(rig) == Catch::Approx(3.0).margin(1e-6));
    // The cutter takes local x in [0, 1], so the survivor starts at local
    // x = 1 -- 11 once the rig's translation is applied. A cutter left behind
    // in the untranslated frame would have cut nothing and left 10 here.
    REQUIRE(rig.mesh().vertices.col(0).minCoeff() ==
            Catch::Approx(11.0).margin(1e-9));
    REQUIRE(rig.mesh().vertices.col(0).maxCoeff() ==
            Catch::Approx(14.0).margin(1e-9));
}
