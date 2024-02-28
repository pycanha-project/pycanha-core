#include <Eigen/Dense>

// #include "pycanha-core/gmm/gmm.hpp"
#include <iostream>
#include <memory>

#include "pycanha-core/pycanha-core.hpp"

using namespace pycanha;  // NOLINT

int main() {
    using gmm::Mesh;

    auto a = sizeof(Point3D);
    auto vector = std::vector<Point3D>(3);

    for (auto& point : vector) {
        point = Point3D(1, 2, 3);
    }

    Point3D p(1, 2, 3);

    double* ptr = vector[0].data();

    double* trick_ptr = reinterpret_cast<double*>(&p);

    std::cout << ptr[0] << std::endl;
    std::cout << ptr[1] << std::endl;
    std::cout << ptr[2] << std::endl;
    std::cout << ptr[3] << std::endl;
    std::cout << ptr[4] << std::endl;
    std::cout << ptr[5] << std::endl;
    std::cout << ptr[6] << std::endl;
    std::cout << ptr[7] << std::endl;
    std::cout << ptr[8] << std::endl;
    std::cout << ptr[9] << std::endl;
};
