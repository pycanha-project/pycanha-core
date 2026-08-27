#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace pycanha::gmm {

// Boolean-subtract group: every item in the target subtree is cut by the union
// of all cutters. This class declares WHICH cutters apply to which subtree; it
// does no meshing of its own. Targets may be any Geometry, including another
// cut group -- a chain of cuts resolves as one operation on the underlying
// primitive rather than as a cut of a cut, which is impossible (see
// gmm/scene/resolve.hpp). A cutter must be a GeometryItem whose primitive is a
// closed solid (Sphere, Cylinder, Cone, Cube, TriangularPrism).
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

  private:
    std::vector<std::shared_ptr<Geometry>> _targets;
    std::vector<std::shared_ptr<GeometryItem>> _cutters;
    std::vector<std::shared_ptr<Geometry>>
        _all_children;  // targets then cutters

    // Storage behind the mesh() reference, not a cache: an intermediate node's
    // resolved mesh is not reused by the model's own resolution, so keeping it
    // would spend memory answering a question nobody asks. The two ends are
    // cached instead -- the item's uncut mesh and the model root.
    mutable TriMeshD _walk_result;
};

// True if `primitive` is a closed solid usable as a cutter.
[[nodiscard]] bool is_closed_solid(const Primitive& primitive) noexcept;

}  // namespace pycanha::gmm
