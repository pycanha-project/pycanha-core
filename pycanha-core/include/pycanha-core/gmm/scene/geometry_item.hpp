#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>

#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"

namespace pycanha::gmm {

// A single meshable primitive plus its ThermalMesh. Leaf of the scene tree;
// caches its own TriMeshD (expressed in this item's parent frame).
class GeometryItem final : public Geometry {
  public:
    GeometryItem(std::string name, Primitive primitive,
                 ThermalMesh thermal_mesh,
                 CoordinateTransformation transform = {});

    [[nodiscard]] const Primitive& primitive() const noexcept;
    [[nodiscard]] const ThermalMesh& thermal_mesh() const noexcept;
    [[nodiscard]] const std::optional<MeshOptions>& mesh_options_override()
        const noexcept;

    void set_primitive(Primitive primitive);
    void set_thermal_mesh(ThermalMesh thermal_mesh);
    void set_mesh_options_override(
        std::optional<MeshOptions> mesh_options_override);

    [[nodiscard]] std::span<const std::shared_ptr<Geometry>> children()
        const noexcept override;
    [[nodiscard]] const TriMeshD& mesh() const override;
    void create_mesh() override;

  protected:
    void invalidate_cache() override;

  private:
    [[nodiscard]] MeshOptions effective_options() const;
    [[nodiscard]] TriMeshD build_mesh() const;

    Primitive _primitive;
    ThermalMesh _thermal_mesh;
    std::optional<MeshOptions> _mesh_options_override;
    mutable std::optional<TriMeshD> _cached_mesh;
};

}  // namespace pycanha::gmm
