#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "pycanha-core/gmm/cutting/cutter.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace pycanha::gmm {

// Boolean-cut group: every item in the target subtree is cut by all the
// cutters. Each cutter has a CutSense: KeepOutside (a hole, the default) or
// KeepInside (only what it encloses is left). What is left of a target is
// inside every KeepInside cutter and outside every KeepOutside one, whatever
// the order they were added in.
//
// This class declares WHICH cutters apply to which subtree; it does no meshing
// of its own. Targets may be any Geometry, including another cut group -- a
// chain of cuts resolves as one operation on the underlying primitive rather
// than as a cut of a cut, which is impossible (see gmm/scene/resolve.hpp). A
// cutter must be a GeometryItem whose primitive is a closed solid (Sphere,
// Cylinder, Cone, Cube, TriangularPrism).
class GeometryGroupCutted final : public Geometry {
  public:
    // `cut_senses` gives the sense of each of `cutters`, in the same order;
    // empty means KeepOutside for all of them. Throws
    // std::invalid_argument if its size is neither 0 nor the number of
    // cutters, and for the reasons cut_with does.
    GeometryGroupCutted(std::string name,
                        std::vector<std::shared_ptr<Geometry>> targets,
                        std::vector<std::shared_ptr<GeometryItem>> cutters,
                        CoordinateTransformation transform = {},
                        const std::vector<CutSense>& cut_senses = {});

    // Adds a cutter that keeps the given side of it. Throws
    // std::invalid_argument on nullptr, an already-registered cutter, a
    // duplicate, or a non-closed-solid primitive.
    void cut_with(std::shared_ptr<GeometryItem> cutter,
                  CutSense sense = CutSense::KeepOutside);

    [[nodiscard]] std::span<const std::shared_ptr<Geometry>> targets()
        const noexcept;
    [[nodiscard]] std::span<const std::shared_ptr<GeometryItem>> cutters()
        const noexcept;
    // The sense of each cutter, in the order of cutters().
    [[nodiscard]] std::span<const CutSense> cut_senses() const noexcept;

    [[nodiscard]] std::span<const std::shared_ptr<Geometry>> children()
        const noexcept override;
    [[nodiscard]] const TriMeshD& mesh() const override;
    void create_mesh() override;

  private:
    std::vector<std::shared_ptr<Geometry>> _targets;
    std::vector<std::shared_ptr<GeometryItem>> _cutters;
    std::vector<CutSense> _cut_senses;  // one per cutter
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
