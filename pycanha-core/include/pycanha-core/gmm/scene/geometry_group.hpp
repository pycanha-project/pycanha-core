#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"

namespace pycanha::gmm {

// A transform applied to a collection of child geometries. Holds shared_ptr
// children directly. Does NOT cache: mesh() recomputes by walking children.
class GeometryGroup : public Geometry {
  public:
    explicit GeometryGroup(
        std::string name,
        std::vector<std::shared_ptr<Geometry>> children = {},
        CoordinateTransformation transform = {});

    // Appends a child. Throws std::invalid_argument on nullptr, a node already
    // registered with a model (id() != 0), or a duplicate handle.
    void add(std::shared_ptr<Geometry> child);

    // Removes `child` if present; returns true if it was a child.
    bool remove_child(const std::shared_ptr<Geometry>& child);

    [[nodiscard]] std::span<const std::shared_ptr<Geometry>> children()
        const noexcept override;
    [[nodiscard]] const TriMeshD& mesh() const override;
    void create_mesh() override;

  protected:
    std::vector<std::shared_ptr<Geometry>> _children;

  private:
    friend class GeometryModel;

    mutable TriMeshD _walk_result;
};

}  // namespace pycanha::gmm
