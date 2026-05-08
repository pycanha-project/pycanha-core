#pragma once

#include <Eigen/Dense>
#include <cstdint>

#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm {

struct UnifiedTriMesh : TriMesh {
    Eigen::Matrix<std::uint64_t, Eigen::Dynamic, 1> geometry_ids;
};

}  // namespace pycanha::gmm
