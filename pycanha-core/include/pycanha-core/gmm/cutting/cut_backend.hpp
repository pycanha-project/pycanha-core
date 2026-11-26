#pragma once

#include <span>

#include "pycanha-core/gmm/cutting/cutter.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace pycanha::gmm::cutting {

class CutBackend {
  public:
    virtual ~CutBackend() = default;

    // What is left of `target` inside every KeepInside cutter and outside
    // every KeepOutside one. The order of the cutters does not matter.
    [[nodiscard]] virtual TriMeshD cut(
        const GeometryItem& target, std::span<const Cutter> cutters,
        const CoordinateTransformation& world_transform,
        const MeshOptions& options) const = 0;
};

}  // namespace pycanha::gmm::cutting
