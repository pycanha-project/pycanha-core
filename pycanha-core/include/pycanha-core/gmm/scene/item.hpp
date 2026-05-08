#pragma once

#include <optional>

#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::gmm {

class Item {
  public:
    Item(Primitive primitive, ThermalMesh thermal_mesh,
         CoordinateTransformation transform = {});

    [[nodiscard]] const Primitive& primitive() const noexcept;
    [[nodiscard]] const ThermalMesh& thermal_mesh() const noexcept;
    [[nodiscard]] const CoordinateTransformation& transform() const noexcept;
    [[nodiscard]] const std::optional<MeshOptions>& mesh_options_override()
        const noexcept;

    void set_primitive(Primitive primitive);
    void set_thermal_mesh(ThermalMesh thermal_mesh);
    void set_transform(CoordinateTransformation transform) noexcept;
    void set_mesh_options_override(
        std::optional<MeshOptions> mesh_options_override) noexcept;

  private:
    Primitive _primitive;
    ThermalMesh _thermal_mesh;
    CoordinateTransformation _transform;
    std::optional<MeshOptions> _mesh_options_override;
};

}  // namespace pycanha::gmm
