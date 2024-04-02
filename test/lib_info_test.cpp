// #include <filesystem>
// #include <fstream>
// #include <iostream>

// #include "pycanha-core/gmm/geometrymodel.hpp"
// #include "pycanha-core/gmm/primitives.hpp"
#include "pycanha-core/pycanha-core.hpp"  // NOLINT (misc-include-cleaner)

// using pycanha::Point3D;
// using pycanha::gmm::Disc;
// using pycanha::gmm::Geometry;
// using pycanha::gmm::GeometryGroup;
// using pycanha::gmm::GeometryItem;
// using pycanha::gmm::GeometryMeshedItem;
// using pycanha::gmm::GeometryModel;
// using pycanha::gmm::Quadrilateral;
using pycanha::gmm::ThermalMesh;
// using pycanha::gmm::Triangle;
// using pycanha::gmm::TriMesh;
// using pycanha::gmm::VerticesList;
// using pycanha::gmm::trimesher::cdt_trimesher;
// using pycanha::gmm::trimesher::create_2d_disc_mesh;
// using pycanha::gmm::trimesher::create_2d_triangular_mesh;
// using pycanha::gmm::trimesher::create_2d_triangular_only_mesh;
// using pycanha::gmm::trimesher::print_point2d;
// using pycanha::gmm::trimesher::print_point3d;
// using std::cout;
// using std::endl;
// using std::numbers::pi;

using pycanha::gmm::Edges;
using pycanha::gmm::TriMesh;
using pycanha::gmm::VerticesList;
using pycanha::gmm::trimesher::create_2d_triangular_mesh;
using pycanha::gmm::trimesher::create_2d_disc_mesh;
using pycanha::gmm::trimesher::print_point3d;
using pycanha::gmm::Sphere;

// Supress clang-tidy warning that main should'n throw exceptions
//  NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    print_package_info();  // NOLINT (misc-include-cleaner)
    // const Point2D p1 = {0.0, 0.0};
    // const Point2D p2 = {2.0, 0.0};
    // const Point2D p3 = {2.0, 2.0};

    const int dir1_size = 3;
    const int dir2_size = 3;
    // const double max_length_dir1 = 10.0;
    // const double max_length_dir2 = 10.0;

    Eigen::VectorXd dir1_mesh(dir1_size);
    dir1_mesh << 0.0, 0.5, 1.0;

    Eigen::VectorXd dir2_mesh(dir2_size);
    dir2_mesh << 0.0, 0.5, 1.0;

    // TriMesh trimesh = create_2d_triangular_mesh(
    //     dir1_mesh, dir2_mesh, p1, p2, p3, max_length_dir1, max_length_dir2);

    // const Point2D p1 = {0.0, 0.0};
    // const Point2D p2 = {1.0, 0.0};
    // const Point2D p3 = {2.0, 2.0};
    // const Point2D p4 = {0.0, 2.0};

    // const int dir1_size = 3;
    // const int dir2_size = 5;
    // const double max_length_dir1 = 0.26;
    // const double max_length_dir2 = 0.4;

    // Eigen::VectorXd dir1_mesh(dir1_size);
    // dir1_mesh << 0.5, 0.75, 1.0;

    // Eigen::VectorXd dir2_mesh(dir2_size);
    // dir2_mesh << 0.0, 0.25, 0.5, 0.75, 1.0;
    // // dir2_mesh << 0.0, 0.125, 0.25;

    // std::cout << "Points 0: " << std::endl;
    // TriMesh trimesh2 = create_2d_disc_mesh(dir1_mesh, dir2_mesh, p1, p2,
    //                                       max_length_dir1, max_length_dir2);

    // std::cout << "Points 1: " << std::endl;
    // auto points = trimesh2.get_vertices();
    // // std::cout << "Points: " << std::endl;
    // std::cout << points.rows() << std::endl;
    // for (int i = 0; i < points.rows(); ++i) {
    //     print_point3d(points.row(i));
    // }
    // auto edges = trimesh.get_edges();
    // std::cout << "Edges: " << edges.size() << std::endl;
    // for (int i = 0; i < edges.size(); ++i) {
    //     std::cout << "[";
    //     for (const auto& edge : edges[i]) {
    //         std::cout << edge << ", ";
    //     }
    //     std::cout << "]," << std::endl;
    // }

    const Point3D p1(0.0, 0.0, 1.0);
    const Point3D p2(0.0, 1.0, 2.0);
    const Point3D p3(1.0, 0.0, 1.0);

    const auto sphere = std::make_shared<Sphere>(p1, p2, p3, 1.0, -1.0, 1.0, 0.0, pi * 2.0);

    ThermalMesh th_mesh;
    th_mesh.set_dir1_mesh({0.0, 0.25, 0.5, 0.75, 1.0});
    th_mesh.set_dir2_mesh({0.0, 0.25, 0.5, 0.75, 1.0});

    auto trimesh = sphere->create_mesh(th_mesh, 0.1);

    auto points = trimesh.get_vertices();
    std::cout << "Points: " << std::endl;
    for (int i = 0; i < points.rows(); ++i) {
        print_point3d(points.row(i));
    }

    auto edges = trimesh.get_edges();
    std::cout << "Edges: " << edges.size() << std::endl;
    for (int i = 0; i < edges.size(); ++i) {
        std::cout << "[";
        for (const auto& edge : edges[i]) {
            std::cout << edge << ", ";
        }
        std::cout << "]," << std::endl;
    }

    return 0;
}
