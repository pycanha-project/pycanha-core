#include "pycanha-core/gmm/scene/geometry.hpp"

#include "pycanha-core/gmm/geometrymodel.hpp"

namespace pycanha::gmm {

void Geometry::on_geometry_mutated() {
    invalidate_cache();
    // Option B: a content edit on a registered object invalidates the owning
    // model's cached mesh / reverse lookup (and bumps its structure version)
    // WITHOUT rebuilding. The model rebuilds lazily on the next mesh() read.
    if (_owning_model != nullptr) {
        _owning_model->notify_content_changed();
    }
}

}  // namespace pycanha::gmm
