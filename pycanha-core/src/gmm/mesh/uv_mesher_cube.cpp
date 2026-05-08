#include <stdexcept>

#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "uv_mesher_internal.hpp"

namespace pycanha::gmm::mesh::detail {

TriMesh mesh_primitive([[maybe_unused]] const Cube& cube,
                       [[maybe_unused]] const ThermalMesh& thermal_mesh,
                       [[maybe_unused]] const MeshOptions& options) {
    throw std::logic_error("Cube is cutter-only");
}

}  // namespace pycanha::gmm::mesh::detail
