#pragma once

#include <span>

#include "pycanha-core/gmm/cutting/cut_backend.hpp"
#include "pycanha-core/gmm/cutting/cutter.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace pycanha::gmm::cutting {

class ManifoldCutBackend : public CutBackend {
  public:
    [[nodiscard]] TriMeshD cut(const GeometryItem& target,
                               std::span<const Cutter> cutters,
                               const CoordinateTransformation& world_transform,
                               const MeshOptions& options) const override;
};

}  // namespace pycanha::gmm::cutting
