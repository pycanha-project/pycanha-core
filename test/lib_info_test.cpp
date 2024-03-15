// #include <filesystem>
// #include <fstream>
// #include <iostream>

// #include "pycanha-core/gmm/geometrymodel.hpp"
// #include "pycanha-core/gmm/primitives.hpp"
#include "pycanha-core/pycanha-core.hpp"  // NOLINT (misc-include-cleaner)

using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriMesh;
using pycanha::gmm::trimesher::create_2d_disc_mesh;

// Supress clang-tidy warning that main should'n throw exceptions
//  NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    const Point2D center = {0.0, 0.0};
    const Point2D outer_point = {1.41421, 0.0};

    const int dir1_size = 4;
    const int dir2_size = 3;
    const double max_length_dir1 = 1.0;
    const double max_length_dir2 = 3.40094;

    Eigen::VectorXd dir1_mesh(dir1_size);
    dir1_mesh << 0.0, 0.33, 0.67, 1.0;

    Eigen::VectorXd dir2_mesh(dir2_size);
    dir2_mesh << 0.25, 0.5, 0.75;

    TriMesh trimesh =
        create_2d_disc_mesh(dir1_mesh, dir2_mesh, center, outer_point,
                            max_length_dir1, max_length_dir2);

    auto v = trimesh.get_vertices();
    std::cout << "Number of vertices: " << v.rows() << std::endl;
    for (int i = 0; i < v.rows(); i++) {
        std::cout << "Vertex " << i << ": [" << v(i, 0) << ", "
                  << v(i, 1) << ", " << v(i, 2)
                  << "]" << std::endl;
    }

    return 0;
}
