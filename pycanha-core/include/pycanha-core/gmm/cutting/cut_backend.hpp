#pragma once

#include <span>

#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/item.hpp"

namespace pycanha::gmm::cutting {

class CutBackend {
  public:
    virtual ~CutBackend() = default;

    [[nodiscard]] virtual TriMesh cut(
        const Item& target, std::span<const Primitive> cutters,
        const CoordinateTransformation& world_transform,
        const MeshOptions& options) const = 0;
};

}  // namespace pycanha::gmm::cutting
