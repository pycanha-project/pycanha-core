#pragma once

#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::gmm {

class UvMesher {
  public:
    [[nodiscard]] TriMesh mesh(const Primitive& primitive,
                               const ThermalMesh& thermal_mesh,
                               const MeshOptions& options) const;
};

}  // namespace pycanha::gmm
