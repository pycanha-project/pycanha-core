#pragma once

#include "pycanha-core/gmm/cutting/cut_backend.hpp"

namespace pycanha::gmm::cutting {

class ManifoldCutBackend : public CutBackend {
  public:
    [[nodiscard]] TriMeshD cut(
        const GeometryItem& target, std::span<const Primitive> cutters,
        const CoordinateTransformation& world_transform,
        const MeshOptions& options) const override;
};

}  // namespace pycanha::gmm::cutting
