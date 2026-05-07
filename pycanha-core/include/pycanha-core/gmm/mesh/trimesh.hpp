#pragma once

#include <Eigen/Dense>
#include <cstdint>

namespace pycanha::gmm {

using FaceIdVector = Eigen::Matrix<std::uint64_t, Eigen::Dynamic, 1>;

struct TriMesh {
    Eigen::MatrixX3d vertices;
    Eigen::MatrixX3i triangles;
    FaceIdVector face_ids;
};

}  // namespace pycanha::gmm
