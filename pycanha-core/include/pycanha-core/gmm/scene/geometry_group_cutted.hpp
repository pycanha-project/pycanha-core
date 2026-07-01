#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace pycanha::gmm {

// Boolean-subtract group: every target is cut by the union of all cutters.
// Caches the resolved TriMeshD. A cutter must be a GeometryItem whose primitive
// is a closed solid (Sphere, Cylinder, Cone, Cube).
class GeometryGroupCutted final : public Geometry {
  public:
    GeometryGroupCutted(std::string name,
                        std::vector<std::shared_ptr<Geometry>> targets,
                        std::vector<std::shared_ptr<GeometryItem>> cutters,
                        CoordinateTransformation transform = {});

    // Adds a cutter. Throws std::invalid_argument on nullptr, an
    // already-registered cutter, a duplicate, or a non-closed-solid primitive.
    void cut_with(std::shared_ptr<GeometryItem> cutter);

    [[nodiscard]] std::span<const std::shared_ptr<Geometry>> targets()
        const noexcept;
    [[nodiscard]] std::span<const std::shared_ptr<GeometryItem>> cutters()
        const noexcept;

    [[nodiscard]] std::span<const std::shared_ptr<Geometry>> children()
        const noexcept override;
    [[nodiscard]] const TriMeshD& mesh() const override;
    void create_mesh() override;

  protected:
    void invalidate_cache() override;

  private:
    [[nodiscard]] MeshOptions effective_options() const;
    [[nodiscard]] TriMeshD build_mesh() const;

    std::vector<std::shared_ptr<Geometry>> _targets;
    std::vector<std::shared_ptr<GeometryItem>> _cutters;
    std::vector<std::shared_ptr<Geometry>>
        _all_children;  // targets then cutters
    mutable std::optional<TriMeshD> _cached_mesh;
};

// True if `primitive` is a closed solid usable as a cutter.
[[nodiscard]] bool is_closed_solid(const Primitive& primitive) noexcept;

}  // namespace pycanha::gmm
