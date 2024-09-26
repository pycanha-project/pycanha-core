// NOLINTBEGIN(misc-include-cleaner)
// #include <Eigen/Core>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <numbers>
#include <set>
#include <vector>

#include "pycanha-core/gmm/primitives.hpp"
#include "pycanha-core/gmm/thermalmesh.hpp"
#include "pycanha-core/gmm/trimesh.hpp"
#include "pycanha-core/parameters.hpp"

TEST_CASE("Create rectangular mesh", "[gmm][trimesh]") {
    using pycanha::gmm::Edges;
    using pycanha::gmm::TriMesh;
    using pycanha::gmm::VerticesList;
    using pycanha::gmm::trimesher::create_2d_rectangular_mesh;

    SECTION("Check points coordinates") {
        const int dir1_size = 3;
        const int dir2_size = 2;
        const double max_length_dir1 = 1.5;
        const double max_length_dir2 = 0.9;

        Eigen::VectorXd dir1_mesh(dir1_size);
        dir1_mesh << 0.0, 7.0, 9.0;

        Eigen::VectorXd dir2_mesh(dir2_size);
        dir2_mesh << 0.0, 1.0;

        TriMesh trimesh = create_2d_rectangular_mesh(
            dir1_mesh, dir2_mesh, max_length_dir1, max_length_dir2);

        // First check size
        // 8 points per row, 3 rows
        REQUIRE(trimesh.get_vertices().rows() == 24);

        // Check coordinates
        VerticesList expected_points(24, 3);
        expected_points << 0.0, 0.0, 0.0,  // The first row
            1.4, 0.0, 0.0,                 // The first row
            2.8, 0.0, 0.0,                 // The first row
            4.2, 0.0, 0.0,                 // The first row
            5.6, 0.0, 0.0,                 // The first row
            7.0, 0.0, 0.0,                 // The first row
            8.0, 0.0, 0.0,                 // The first row
            9.0, 0.0, 0.0,                 // The first row
            0.0, 0.5, 0.0,                 // The second row
            1.4, 0.5, 0.0,                 // The second row
            2.8, 0.5, 0.0,                 // The second row
            4.2, 0.5, 0.0,                 // The second row
            5.6, 0.5, 0.0,                 // The second row
            7.0, 0.5, 0.0,                 // The second row
            8.0, 0.5, 0.0,                 // The second row
            9.0, 0.5, 0.0,                 // The second row
            0.0, 1.0, 0.0,                 // The third row
            1.4, 1.0, 0.0,                 // The third row
            2.8, 1.0, 0.0,                 // The third row
            4.2, 1.0, 0.0,                 // The third row
            5.6, 1.0, 0.0,                 // The third row
            7.0, 1.0, 0.0,                 // The third row
            8.0, 1.0, 0.0,                 // The third row
            9.0, 1.0, 0.0;                 // The third row

        // First row
        REQUIRE(trimesh.get_vertices().isApprox(expected_points, LENGTH_TOL));
    }

    SECTION("Check a general 2D mesh") {
        Eigen::VectorXd dir1_mesh(4);
        dir1_mesh << 0.0, 3.0, 4.0, 8.0;

        Eigen::VectorXd dir2_mesh(3);
        dir2_mesh << 0.0, 3.0, 7.0;

        TriMesh trimesh =
            create_2d_rectangular_mesh(dir1_mesh, dir2_mesh, 2.5, 3.5);

        // First check size
        REQUIRE(trimesh.get_vertices().rows() == 24);
        REQUIRE(trimesh.get_edges().size() == 17);
        REQUIRE(trimesh.get_perimeter_edges().size() == 10);
        REQUIRE(trimesh.get_faces_edges().size() == 6);

        // The values are from the function documentation

        // Check the edges are correct std::vector<pycanha::MeshIndex>{0, 1, 2}

        // Check the edges are correct
        REQUIRE(trimesh.get_edges()[0] == (Edges(3) << 0, 1, 2).finished());
        REQUIRE(trimesh.get_edges()[1] == (Edges(2) << 2, 3).finished());
        REQUIRE(trimesh.get_edges()[2] == (Edges(3) << 3, 4, 5).finished());
        REQUIRE(trimesh.get_edges()[3] == (Edges(3) << 6, 7, 8).finished());
        REQUIRE(trimesh.get_edges()[4] == (Edges(2) << 8, 9).finished());
        REQUIRE(trimesh.get_edges()[5] == (Edges(3) << 9, 10, 11).finished());
        REQUIRE(trimesh.get_edges()[6] == (Edges(3) << 18, 19, 20).finished());
        REQUIRE(trimesh.get_edges()[7] == (Edges(2) << 20, 21).finished());
        REQUIRE(trimesh.get_edges()[8] == (Edges(3) << 21, 22, 23).finished());

        REQUIRE(trimesh.get_edges()[9] == (Edges(2) << 0, 6).finished());
        REQUIRE(trimesh.get_edges()[10] == (Edges(3) << 6, 12, 18).finished());
        REQUIRE(trimesh.get_edges()[11] == (Edges(2) << 2, 8).finished());
        REQUIRE(trimesh.get_edges()[12] == (Edges(3) << 8, 14, 20).finished());
        REQUIRE(trimesh.get_edges()[13] == (Edges(2) << 3, 9).finished());
        REQUIRE(trimesh.get_edges()[14] == (Edges(3) << 9, 15, 21).finished());
        REQUIRE(trimesh.get_edges()[15] == (Edges(2) << 5, 11).finished());
        REQUIRE(trimesh.get_edges()[16] == (Edges(3) << 11, 17, 23).finished());

        // Check the perimeter edges are correct
        REQUIRE(trimesh.get_perimeter_edges() ==
                (Edges(10) << 0, 1, 2, 15, 16, 8, 7, 6, 10, 9).finished());

        // Check the faces edges are correct
        REQUIRE(trimesh.get_faces_edges()[0] ==
                (Edges(4) << 0, 11, 3, 9).finished());
        REQUIRE(trimesh.get_faces_edges()[1] ==
                (Edges(4) << 1, 13, 4, 11).finished());
        REQUIRE(trimesh.get_faces_edges()[2] ==
                (Edges(4) << 2, 15, 5, 13).finished());
        REQUIRE(trimesh.get_faces_edges()[3] ==
                (Edges(4) << 3, 12, 6, 10).finished());
        REQUIRE(trimesh.get_faces_edges()[4] ==
                (Edges(4) << 4, 14, 7, 12).finished());
        REQUIRE(trimesh.get_faces_edges()[5] ==
                (Edges(4) << 5, 16, 8, 14).finished());
    }
}

TEST_CASE("Create quadrilateral 2D mesh", "[gmm][trimesh]") {
    using pycanha::gmm::TriMesh;
    using pycanha::gmm::VerticesList;
    using pycanha::gmm::trimesher::create_2d_quadrilateral_mesh;

    SECTION("Check points coordinates") {
        const Point2D p1 = {0.0, 0.0};
        const Point2D p2 = {2.0, 0.0};
        const Point2D p3 = {2.0, 2.0};
        const Point2D p4 = {0.0, 2.0};

        const int dir1_size = 3;
        const int dir2_size = 3;
        const double max_length_dir1 = 10.0;
        const double max_length_dir2 = 10.0;

        Eigen::VectorXd dir1_mesh(dir1_size);
        dir1_mesh << 0.0, 0.5, 1.0;

        Eigen::VectorXd dir2_mesh(dir2_size);
        dir2_mesh << 0.0, 0.5, 1.0;

        TriMesh trimesh =
            create_2d_quadrilateral_mesh(dir1_mesh, dir2_mesh, p1, p2, p3, p4,
                                         max_length_dir1, max_length_dir2);

        // First check size
        // 3 points per row, 3 rows
        REQUIRE(trimesh.get_vertices().rows() == 9);

        // Check coordinates
        VerticesList expected_points(9, 3);
        expected_points << 0.0, 0.0, 0.0,  //
            1.0, 0.0, 0.0,                 //
            2.0, 0.0, 0.0,                 //
            0.0, 1.0, 0.0,                 //
            1.0, 1.0, 0.0,                 //
            2.0, 1.0, 0.0,                 //
            0.0, 2.0, 0.0,                 //
            1.0, 2.0, 0.0,                 //
            2.0, 2.0, 0.0;                 //

        // First row
        REQUIRE(trimesh.get_vertices().isApprox(expected_points, LENGTH_TOL));
    }

    SECTION("Check a general 2D mesh") {
        const Point2D p1 = {0.0, 0.0};
        const Point2D p2 = {2.0, 0.0};
        const Point2D p3 = {1.5, 2.0};
        const Point2D p4 = {0.5, 2.0};

        const int dir1_size = 3;
        const int dir2_size = 3;
        const double max_length_dir1 = 0.49;
        const double max_length_dir2 = 0.9;

        Eigen::VectorXd dir1_mesh(dir1_size);
        dir1_mesh << 0.0, 0.5, 1.0;

        Eigen::VectorXd dir2_mesh(dir2_size);
        dir2_mesh << 0.0, 0.5, 1.0;

        const TriMesh trimesh =
            create_2d_quadrilateral_mesh(dir1_mesh, dir2_mesh, p1, p2, p3, p4,
                                         max_length_dir1, max_length_dir2);

        // First check size
        // 3 points per row, 3 rows
        // REQUIRE(trimesh.get_vertices().rows() == 9);

        // Check coordinates
        // VerticesList expected_points(9, 3);
        // expected_points << 0.0, 0.0, 0.0,  //
        //    1.0, 0.0, 0.0,                 //
        //    2.0, 0.0, 0.0,                 //
        //    0.0, 1.0, 0.0,                 //
        //    1.0, 1.0, 0.0,                 //
        //    2.0, 1.0, 0.0,                 //
        //    0.0, 2.0, 0.0,                 //
        //    1.0, 2.0, 0.0,                 //
        //    2.0, 2.0, 0.0;                 //

        //// First row
        // REQUIRE(trimesh.get_vertices().isApprox(expected_points,
        // LENGTH_TOL));
    }
}

TEST_CASE("Create triangle only 2D mesh", "[gmm][trimesh]") {
    using pycanha::gmm::TriMesh;
    using pycanha::gmm::VerticesList;
    using pycanha::gmm::trimesher::create_2d_triangular_only_mesh;

    SECTION("Check points coordinates") {
        const Point2D p1 = {0.0, 0.0};
        const Point2D p2 = {1.0, 0.0};
        const Point2D p3 = {1.0, 1.0};

        const int dir2_size = 3;
        const double max_length_dir1 = 0.5;
        const double max_length_dir2 = 0.5;

        Eigen::VectorXd dir2_mesh(dir2_size);
        dir2_mesh << 0.0, 0.5, 1.0;

        const TriMesh trimesh = create_2d_triangular_only_mesh(
            dir2_mesh, p1, p2, p3, max_length_dir1, max_length_dir2);

        // First check size
        // 3 points per row, 3 rows
    }
}

// NOLINTBEGIN(readability-function-cognitive-complexity)
TEST_CASE("Mesh a triangle", "[gmm][trimesh]") {
    using pycanha::gmm::Edges;
    using pycanha::gmm::TriMesh;
    using pycanha::gmm::VerticesList;
    using pycanha::gmm::trimesher::create_2d_triangular_mesh;
    using pycanha::gmm::trimesher::print_point3d;

    SECTION("Check points coordinates") {
        const Point2D p1 = {0.0, 0.0};
        const Point2D p2 = {2.0, 0.0};
        const Point2D p3 = {2.0, 2.0};

        const int dir1_size = 3;
        const int dir2_size = 3;
        const double max_length_dir1 = 0.7;
        const double max_length_dir2 = 0.7;

        Eigen::VectorXd dir1_mesh(dir1_size);
        dir1_mesh << 0.0, 0.5, 1.0;

        Eigen::VectorXd dir2_mesh(dir2_size);
        dir2_mesh << 0.0, 0.5, 1.0;

        TriMesh trimesh = create_2d_triangular_mesh(
            dir1_mesh, dir2_mesh, p1, p2, p3, max_length_dir1, max_length_dir2);

        // First check size
        // 3 points per row, 3 rows
        REQUIRE(trimesh.get_vertices().rows() == 20);

        // Check coordinates
        VerticesList expected_points(20, 3);
        expected_points << 0., 0., 0., 0.5, 0., 0., 1., 0., 0., 0.5, 0.25, 0.,
            1., 0.5, 0., 0.333333, 0.333333, 0., 0.666667, 0.666667, 0., 1., 1.,
            0., 1.5, 0., 0., 2., 0., 0., 1.5, 0.75, 0., 2., 1., 0., 1.33333,
            1.33333, 0., 1.66667, 1.66667, 0., 2., 2., 0., 2., 0.5, 0., 2., 1.5,
            0., 1.5, 0.375, 0., 1.33333, 1., 0., 1.66667, 1.25, 0.;

        // First row
        REQUIRE(trimesh.get_vertices().isApprox(expected_points, 1.0E-5));

        // First check size
        REQUIRE(trimesh.get_edges().size() == 10);
        REQUIRE(trimesh.get_perimeter_edges().size() == 6);
        REQUIRE(trimesh.get_faces_edges().size() == 4);

        // The values are from the function documentation

        // Check the edges are correct
        REQUIRE(trimesh.get_edges()[0] == (Edges(3) << 0, 1, 2).finished());
        REQUIRE(trimesh.get_edges()[1] == (Edges(3) << 0, 3, 4).finished());
        REQUIRE(trimesh.get_edges()[2] == (Edges(4) << 0, 5, 6, 7).finished());
        REQUIRE(trimesh.get_edges()[3] == (Edges(2) << 2, 4).finished());
        REQUIRE(trimesh.get_edges()[4] == (Edges(2) << 4, 7).finished());
        REQUIRE(trimesh.get_edges()[5] == (Edges(3) << 2, 8, 9).finished());
        REQUIRE(trimesh.get_edges()[6] == (Edges(3) << 4, 10, 11).finished());
        REQUIRE(trimesh.get_edges()[7] ==
                (Edges(4) << 7, 12, 13, 14).finished());
        REQUIRE(trimesh.get_edges()[8] == (Edges(3) << 9, 15, 11).finished());
        REQUIRE(trimesh.get_edges()[9] == (Edges(3) << 11, 16, 14).finished());

        // Check the perimeter edges are correct
        REQUIRE(trimesh.get_perimeter_edges() ==
                (Edges(6) << 0, 5, 8, 9, 7, 2).finished());

        // Check the faces edges are correct
        REQUIRE(trimesh.get_faces_edges()[0] ==
                (Edges(3) << 0, 3, 1).finished());
        REQUIRE(trimesh.get_faces_edges()[1] ==
                (Edges(3) << 1, 4, 2).finished());
        REQUIRE(trimesh.get_faces_edges()[2] ==
                (Edges(4) << 5, 8, 6, 3).finished());
        REQUIRE(trimesh.get_faces_edges()[3] ==
                (Edges(4) << 6, 9, 7, 4).finished());
    }
}
// NOLINTEND(readability-function-cognitive-complexity)

TEST_CASE("Mesh a rectangle", "[gmm][trimesh][rectangle]") {
    using pycanha::Point3D;
    using pycanha::gmm::Edges;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    using pycanha::gmm::TriMesh;

    SECTION("Check the created mesh is correct") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(5.0, 0.0, 0.0);
        const Point3D p3(0.0, 2.0, 0.0);
        const Rectangle rect(p1, p2, p3);

        ThermalMesh th_mesh;
        th_mesh.set_dir2_mesh(std::vector<double>{0.0, 0.5, 1.0});

        TriMesh trimesh = rect.create_mesh(th_mesh, 0.0);

        REQUIRE(trimesh.get_vertices().rows() == 6);
        REQUIRE(trimesh.get_triangles().rows() == 4);
        REQUIRE(trimesh.get_face_ids().size() == 4);
        REQUIRE(trimesh.get_perimeter_edges() ==
                (Edges(6) << 0, 5, 6, 2, 4, 3).finished());
        REQUIRE(trimesh.get_faces_edges().size() == 2);
        REQUIRE(trimesh.get_faces_edges()[0] ==
                (Edges(4) << 0, 5, 1, 3).finished());
        REQUIRE(trimesh.get_faces_edges()[1] ==
                (Edges(4) << 1, 6, 2, 4).finished());

        // Check face ids. Triangles should be sorted by face id,
        // so the rectangle should have two triangles with face id 0
        // and two triangles with face id 2
        REQUIRE(trimesh.get_face_ids() == (Edges(4) << 0, 0, 2, 2).finished());
    }
}

// NOLINTBEGIN(readability-function-cognitive-complexity)
TEST_CASE("Mesh a cylinder", "[gmm][trimesh][cylinder]") {
    using pycanha::Point3D;
    using pycanha::gmm::Cylinder;
    using pycanha::gmm::ThermalMesh;
    using pycanha::gmm::TriMesh;
    using std::numbers::pi;

    SECTION("Check the created mesh is correct") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(0.0, 0.0, 1.0);
        const Point3D p3(1.0, 0.0, 0.0);
        const double radius = 1.0;
        const double start_angle = 0.0;
        const double end_angle = 2 * pi;
        const Cylinder cyl(p1, p2, p3, radius, start_angle, end_angle);

        ThermalMesh th_mesh;

        th_mesh.set_dir1_mesh(std::vector<double>{0.0, 0.5, 1.0});
        th_mesh.set_dir2_mesh(std::vector<double>{0.0, 0.5, 1.0});

        auto trimesh = std::make_shared<TriMesh>(cyl.create_mesh(th_mesh, 0.5));

        // TODO: add checks

        // Check face ids. Triangles should be sorted by face id.
        // The exact number of triangles depend on the resolution.
        // But we can check that there are at least one triangle with
        // the expected ids
        REQUIRE(std::is_sorted(trimesh->get_face_ids().begin(),
                               trimesh->get_face_ids().end()));
        auto expected_face_ids = std::set<pycanha::MeshIndex>{0, 2, 4, 6};
        auto unique_face_ids = std::set<pycanha::MeshIndex>(
            trimesh->get_face_ids().begin(), trimesh->get_face_ids().end());
        REQUIRE(expected_face_ids == unique_face_ids);

        // Quick dirty check for the reconstruct_face_edges_2d
        // Create a vector of sets of edges
        std::vector<std::set<pycanha::MeshIndex>> expected_face_edges(
            trimesh->get_faces_edges().size());
        for (pycanha::VectorIndex i = 0; i < trimesh->get_faces_edges().size();
             ++i) {
            for (auto edge_id : trimesh->get_faces_edges()[i]) {
                expected_face_edges[i].insert(edge_id);
            }
        }
        // Transform points to 2D and reconstruct edges
        for (int i = 0; i < trimesh->get_vertices().rows(); i++) {
            Point2D p = cyl.from_3d_to_2d(trimesh->get_vertices().row(i));
            trimesh->get_vertices().row(i)(0) = p(0);
            trimesh->get_vertices().row(i)(1) = p(1);
        }

        cyl.reconstruct_face_edges_2d(trimesh, th_mesh);
        // Create another vector of sets with the new faces edges
        std::vector<std::set<pycanha::MeshIndex>> reconstructed_face_edges(
            trimesh->get_faces_edges().size());
        for (pycanha::VectorIndex i = 0; i < trimesh->get_faces_edges().size();
             ++i) {
            for (auto edge_id : trimesh->get_faces_edges()[i]) {
                reconstructed_face_edges[i].insert(edge_id);
            }
        }
        // Compare reconstructed_face_edges and expected_face_edges face by face
        for (pycanha::VectorIndex i = 0; i < reconstructed_face_edges.size();
             ++i) {
            REQUIRE(reconstructed_face_edges[i] == expected_face_edges[i]);
        }
    }
}
// NOLINTEND(readability-function-cognitive-complexity)

// NOLINTBEGIN(readability-function-cognitive-complexity)
TEST_CASE("Mesh a disc", "[gmm][trimesh][disc]") {
    using pycanha::Point3D;
    using pycanha::gmm::Disc;
    using pycanha::gmm::Edges;
    using pycanha::gmm::FaceIdsList;
    using pycanha::gmm::ThermalMesh;
    using pycanha::gmm::TriMesh;
    using pycanha::gmm::VerticesList;
    using std::numbers::pi;

    SECTION("Check the created mesh is correct for a complete disc") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(0.0, 0.0, 1.0);
        const Point3D p3(1.0, 0.0, 0.0);
        const double inner_radius = 0.0;
        const double outer_radius = 1.41421;
        const double start_angle = 0.0;
        const double end_angle = 2 * pi;
        const Disc disc(p1, p2, p3, inner_radius, outer_radius, start_angle,
                        end_angle);

        ThermalMesh th_mesh;

        th_mesh.set_dir1_mesh(std::vector<double>{0.0, 0.5, 1.0});
        th_mesh.set_dir2_mesh(std::vector<double>{0.0, 0.25, 0.5, 0.75, 1.0});

        TriMesh trimesh = disc.create_mesh(th_mesh, 0.4);

        // TODO: add checks

        // Check face ids. Triangles should be sorted by face id.
        // The exact number of triangles depend on the resolution.
        // But we can check that there are at least one triangle with
        // the expected ids
        REQUIRE(std::is_sorted(trimesh.get_face_ids().begin(),
                               trimesh.get_face_ids().end()));
        FaceIdsList expected_face_ids = (FaceIdsList(16) << 0, 2, 2, 2, 4, 6, 6,
                                         6, 8, 10, 10, 10, 12, 14, 14, 14)
                                            .finished();
        auto unique_face_ids = trimesh.get_face_ids();
        REQUIRE(expected_face_ids == unique_face_ids);

        const VerticesList expected_points =
            (VerticesList(13, 3) << 0.0, 0.0, 0.0, 0.707105, 0.0, 0.0, 1.41421,
             0.0, 0.0, 0.0, 0.707105, 0.0, 0.0, 1.41421, 0.0, -0.707105, 0.0,
             0.0, -1.41421, 0.0, 0.0, 0.0, -0.707105, 0.0, 0.0, -1.41421, 0.0,
             0.999997, 0.999997, 0.0, -0.999997, 0.999997, 0.0, -0.999997,
             -0.999997, 0.0, 0.999997, -0.999997, 0)
                .finished();
        auto unique_points = trimesh.get_vertices();
        REQUIRE(expected_points.isApprox(unique_points, 1.0E-5));

        // Check the edges are correct
        REQUIRE(trimesh.get_edges().size() == 16);
        REQUIRE(trimesh.get_edges()[0] == (Edges(2) << 0, 1).finished());
        REQUIRE(trimesh.get_edges()[1] == (Edges(2) << 1, 2).finished());
        REQUIRE(trimesh.get_edges()[2] == (Edges(2) << 0, 3).finished());
        REQUIRE(trimesh.get_edges()[3] == (Edges(2) << 3, 4).finished());
        REQUIRE(trimesh.get_edges()[4] == (Edges(2) << 0, 5).finished());
        REQUIRE(trimesh.get_edges()[5] == (Edges(2) << 5, 6).finished());
        REQUIRE(trimesh.get_edges()[14] == (Edges(3) << 6, 11, 8).finished());
        REQUIRE(trimesh.get_edges()[15] == (Edges(3) << 8, 12, 2).finished());

        // Check the perimeter edges are correct
        REQUIRE(trimesh.get_perimeter_edges() ==
                (Edges(4) << 12, 13, 14, 15).finished());
    }

    SECTION("Check the created mesh is correct for a partial disc") {
        const Point3D p1(0.0, 0.0, 0.0);
        const Point3D p2(0.0, 0.0, 1.0);
        const Point3D p3(1.0, 0.0, 0.0);
        const double inner_radius = 0.5;
        const double outer_radius = 1.0;
        const double start_angle = pi / 2.0;
        const double end_angle = 3.0 / 2.0 * pi;
        const Disc disc(p1, p2, p3, inner_radius, outer_radius, start_angle,
                        end_angle);

        ThermalMesh th_mesh;

        th_mesh.set_dir1_mesh(std::vector<double>{0.0, 0.25, 0.5, 0.75, 1.0});
        th_mesh.set_dir2_mesh(std::vector<double>{0.0, 0.25, 0.5, 0.75, 1.0});

        TriMesh trimesh = disc.create_mesh(th_mesh, 0.1);

        // TODO: add checks

        // Check face ids. Triangles should be sorted by face id.
        // The exact number of triangles depend on the resolution.
        // But we can check that there are at least one triangle with
        // the expected ids
        REQUIRE(std::is_sorted(trimesh.get_face_ids().begin(),
                               trimesh.get_face_ids().end()));
        FaceIdsList expected_face_ids =
            (FaceIdsList(32) << 0, 0, 0, 2, 2, 4, 4, 6, 8, 8, 8, 10, 10, 12, 12,
             14, 16, 16, 16, 18, 18, 20, 20, 22, 24, 24, 24, 26, 26, 28, 28, 30)
                .finished();
        auto unique_face_ids = trimesh.get_face_ids();
        REQUIRE(expected_face_ids == unique_face_ids);

        auto unique_points = trimesh.get_vertices();
        const VerticesList expected_points =
            (VerticesList(25, 3) << 0.0, 0.5, 0.0, 0.0, 0.625, 0.0, 0.0, 0.75,
             0.0, 0.0, 0.875, 0.0, 0.0, 1.0, 0.0, -0.353553, 0.353553, 0.0,
             -0.441942, 0.441942, 0.0, -0.53033, 0.53033, 0.0, -0.618718,
             0.618718, 0.0, -0.707107, 0.707107, 0.0, -0.5, 0.0, 0.0, -0.625,
             0.0, 0.0, -0.75, 0.0, 0.0, -0.875, 0.0, 0.0, -1.0, 0.0, 0.0,
             -0.353553, -0.353553, 0.0, -0.441942, -0.441942, 0.0, -0.53033,
             -0.53033, 0.0, -0.618718, -0.618718, 0.0, -0.707107, -0.707107,
             0.0, 0.0, -0.5, 0.0, 0.0, -0.625, 0.0, 0.0, -0.75, 0.0, 0.0,
             -0.875, 0.0, 0.0, -1.0, 0.0)
                .finished();
        REQUIRE(expected_points.isApprox(unique_points, 1.0E-5));

        // Check the edges are correct
        REQUIRE(trimesh.get_edges().size() == 40);
        REQUIRE(trimesh.get_edges()[0] == (Edges(2) << 0, 1).finished());
        REQUIRE(trimesh.get_edges()[1] == (Edges(2) << 1, 2).finished());
        REQUIRE(trimesh.get_edges()[2] == (Edges(2) << 2, 3).finished());
        REQUIRE(trimesh.get_edges()[27] == (Edges(2) << 16, 21).finished());
        REQUIRE(trimesh.get_edges()[39] == (Edges(2) << 19, 24).finished());

        // Check the perimeter edges are correct
        REQUIRE(trimesh.get_perimeter_edges() == (Edges(16) << 0, 1, 2, 3, 16,
                                                  17, 18, 19, 20, 21, 22, 23,
                                                  36, 37, 38, 39)
                                                     .finished());
    }
}
// NOLINTEND(readability-function-cognitive-complexity)

TEST_CASE("Test the model mesh", "[gmm][trimesh][model]") {
    using pycanha::Point3D;
    using pycanha::gmm::Cylinder;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    using pycanha::gmm::TriMesh;
    using pycanha::gmm::TriMeshModel;
    using std::numbers::pi;

    const Point3D p1_rect(0.0, 0.0, 0.0);
    const Point3D p2_rect(5.0, 0.0, 0.0);
    const Point3D p3_rect(0.0, 2.0, 0.0);
    const Rectangle rect(p1_rect, p2_rect, p3_rect);

    ThermalMesh th_mesh_rect;
    th_mesh_rect.set_dir2_mesh(std::vector<double>{0.0, 0.5, 1.0});

    TriMesh trimesh_rect = rect.create_mesh(th_mesh_rect, 0.0);

    const Point3D p1_cyl(0.0, 0.0, 0.0);
    const Point3D p2_cyl(0.0, 0.0, 1.0);
    const Point3D p3_cyl(1.0, 0.0, 0.0);
    const double radius = 1.0;
    const double start_angle = 0.0;
    const double end_angle = 2 * pi;
    const Cylinder cyl(p1_cyl, p2_cyl, p3_cyl, radius, start_angle, end_angle);

    ThermalMesh th_mesh_cyl;

    th_mesh_cyl.set_dir1_mesh(std::vector<double>{0.0, 0.5, 1.0});
    th_mesh_cyl.set_dir2_mesh(std::vector<double>{0.0, 0.5, 1.0});

    TriMesh trimesh_cyl = cyl.create_mesh(th_mesh_cyl, 0.5);

    SECTION("Check set/get methods of TriMeshModel") {
        // TODO
    }

    SECTION("Check we can add meshes") {
        auto model_trimesh = TriMeshModel();
        model_trimesh.add_mesh(trimesh_rect, 0);
        model_trimesh.add_mesh(trimesh_cyl, 1);

        // Check number of points
        REQUIRE(model_trimesh.get_vertices().rows() ==
                trimesh_rect.get_vertices().rows() +
                    trimesh_cyl.get_vertices().rows());
        // Check number of triangles
        REQUIRE(model_trimesh.get_triangles().rows() ==
                trimesh_rect.get_triangles().rows() +
                    trimesh_cyl.get_triangles().rows());

        // Check we can retrieve the meshes
        auto submesh_rect = model_trimesh.get_geometry_mesh(0);
        auto submesh_cyl = model_trimesh.get_geometry_mesh(1);

        REQUIRE(submesh_rect.get_vertices().isApprox(
            trimesh_rect.get_vertices(), LENGTH_TOL * 10));
        REQUIRE(submesh_rect.get_triangles() == trimesh_rect.get_triangles());
        REQUIRE(submesh_rect.get_edges() == trimesh_rect.get_edges());
        REQUIRE(submesh_rect.get_perimeter_edges() ==
                trimesh_rect.get_perimeter_edges());
        REQUIRE(submesh_rect.get_faces_edges() ==
                trimesh_rect.get_faces_edges());
        REQUIRE(submesh_rect.get_surface1_color() ==
                trimesh_rect.get_surface1_color());
        REQUIRE(submesh_rect.get_surface2_color() ==
                trimesh_rect.get_surface2_color());

        // We need to relax the LENGHT_TOL because the TriMeshModel uses float
        // (32 bits) and the TriMesh uses double (64 bits)
        REQUIRE(submesh_cyl.get_vertices().isApprox(trimesh_cyl.get_vertices(),
                                                    LENGTH_TOL * 10));
        REQUIRE(submesh_cyl.get_triangles() == trimesh_cyl.get_triangles());
        REQUIRE(submesh_cyl.get_edges() == trimesh_cyl.get_edges());
        REQUIRE(submesh_cyl.get_perimeter_edges() ==
                trimesh_cyl.get_perimeter_edges());
        REQUIRE(submesh_cyl.get_faces_edges() == trimesh_cyl.get_faces_edges());
        REQUIRE(submesh_cyl.get_surface1_color() ==
                trimesh_cyl.get_surface1_color());
        REQUIRE(submesh_cyl.get_surface2_color() ==
                trimesh_cyl.get_surface2_color());

        // We can compare face ids of the first geometry added. But not the
        // second and later geometries.
        REQUIRE(submesh_rect.get_face_ids() == trimesh_rect.get_face_ids());
        // REQUIRE(submesh_cyl.get_face_ids() == trimesh_cyl.get_face_ids());
    }
}
// NOLINTEND(misc-include-cleaner)
