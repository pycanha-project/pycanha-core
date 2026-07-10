#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/ops/validate.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace {

using pycanha::gmm::TriMeshD;
namespace mesh_ops = pycanha::gmm::mesh::ops;

[[nodiscard]] TriMeshD make_square_mesh() {
    TriMeshD mesh;
    mesh.vertices.resize(4, 3);
    mesh.vertices << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0;
    mesh.triangles.resize(2, 3);
    mesh.triangles << 0, 1, 2, 0, 2, 3;
    mesh.face_ids.resize(2);
    mesh.face_ids << 0U, 0U;
    return mesh;
}

}  // namespace

TEST_CASE("TriMeshD mesh ops compute geometry metrics", "[gmm][mesh]") {
    const TriMeshD mesh = make_square_mesh();

    const auto areas = mesh_ops::compute_areas(mesh);
    const auto centroids = mesh_ops::compute_centroids(mesh);
    const auto normals = mesh_ops::compute_face_normals(mesh);
    const auto bbox = mesh_ops::bounding_box(mesh);

    REQUIRE(areas.size() == 2);
    REQUIRE(areas[0] == Catch::Approx(0.5));
    REQUIRE(areas[1] == Catch::Approx(0.5));
    REQUIRE(centroids.row(0).isApprox(
        Eigen::RowVector3d(2.0 / 3.0, 1.0 / 3.0, 0.0)));
    REQUIRE(normals.row(0).isApprox(Eigen::RowVector3d(0.0, 0.0, 1.0)));
    REQUIRE(bbox.min().isApprox(Eigen::Vector3d(0.0, 0.0, 0.0)));
    REQUIRE(bbox.max().isApprox(Eigen::Vector3d(1.0, 1.0, 0.0)));
}

TEST_CASE("TriMeshD mesh ops compute per-face-slot areas", "[gmm][mesh]") {
    TriMeshD mesh = make_square_mesh();

    const auto slot_areas = mesh_ops::compute_face_slot_areas(mesh);

    // One face pair (slots 0/1): both sides share the full pair area.
    REQUIRE(slot_areas.size() == 2);
    REQUIRE(slot_areas[0] == Catch::Approx(1.0));
    REQUIRE(slot_areas[1] == Catch::Approx(1.0));

    // A gap slot pair (face id 2 unused after a cut) stays 0.
    mesh.face_ids << 0U, 4U;
    const auto gapped = mesh_ops::compute_face_slot_areas(mesh);
    REQUIRE(gapped.size() == 6);
    REQUIRE(gapped[0] == Catch::Approx(0.5));
    REQUIRE(gapped[1] == Catch::Approx(0.5));
    REQUIRE(gapped[2] == 0.0);
    REQUIRE(gapped[3] == 0.0);
    REQUIRE(gapped[4] == Catch::Approx(0.5));
    REQUIRE(gapped[5] == Catch::Approx(0.5));
}

TEST_CASE("TriMeshD mesh ops validate open manifold meshes", "[gmm][mesh]") {
    const TriMeshD mesh = make_square_mesh();

    REQUIRE_FALSE(mesh_ops::is_watertight(mesh));
    REQUIRE(mesh_ops::has_consistent_face_ids(mesh));
}
