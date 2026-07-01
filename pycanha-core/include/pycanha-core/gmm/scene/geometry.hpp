#pragma once

#include <memory>
#include <span>
#include <string>
#include <utility>

#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::gmm {

class GeometryModel;
class GeometryGroup;

// Abstract base of the object-centric scene tree. Concrete types are
// GeometryItem, GeometryGroup, GeometryGroupCutted. Objects are always held by
// std::shared_ptr; copying/moving a Geometry is disabled (identity matters).
class Geometry {
  public:
    Geometry(const Geometry&) = delete;
    Geometry& operator=(const Geometry&) = delete;
    Geometry(Geometry&&) = delete;
    Geometry& operator=(Geometry&&) = delete;
    virtual ~Geometry() = default;

    [[nodiscard]] const std::string& name() const noexcept { return _name; }
    [[nodiscard]] GeometryId id() const noexcept { return _id; }
    [[nodiscard]] const CoordinateTransformation& transform() const noexcept {
        return _transform;
    }
    [[nodiscard]] GeometryModel* owning_model() const noexcept {
        return _owning_model;
    }

    void set_transform(CoordinateTransformation transform) {
        _transform = std::move(transform);
        on_geometry_mutated();
    }

    // Immediate children (empty for GeometryItem).
    [[nodiscard]] virtual std::span<const std::shared_ptr<Geometry>> children()
        const noexcept = 0;

    // Subtree mesh expressed in the object's parent frame (this object's own
    // transform already applied). Lazy; cached by GeometryItem and
    // GeometryGroupCutted, walked on the fly by plain GeometryGroup.
    [[nodiscard]] virtual const TriMeshD& mesh() const = 0;

    // Forces a rebuild of this object's mesh (and its subtree).
    virtual void create_mesh() = 0;

  protected:
    Geometry(std::string name, CoordinateTransformation transform)
        : _name(std::move(name)),
          _transform(std::move(transform)),
          _id(next_geometry_id()) {}

    // Hook invoked after any content mutation. Non-virtual: it clears the
    // object's own cache (via the virtual invalidate_cache()) AND, if the
    // object is registered, invalidates the owning model's caches (Option B
    // mutation propagation). Defined in geometry.cpp (needs GeometryModel).
    void on_geometry_mutated();

    // Derived types override to reset their own cached mesh (no-op for a plain
    // GeometryGroup, which does not cache).
    virtual void invalidate_cache() {}

    std::string _name;
    CoordinateTransformation _transform;
    // Ephemeral, process-wide-unique runtime identity assigned at construction
    // (regenerated on load; never persisted to foreign formats, never embedded
    // in a name). Registration state is tracked by _owning_model, not by _id.
    GeometryId _id;
    GeometryModel* _owning_model = nullptr;

    friend class GeometryModel;
    friend class GeometryGroup;
};

}  // namespace pycanha::gmm
