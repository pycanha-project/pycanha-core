#include "pycanha-core/gmm/scene/item.hpp"

#include <optional>
#include <utility>

#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::gmm {

Item::Item(Primitive primitive, ThermalMesh thermal_mesh,
           CoordinateTransformation transform)
    : _primitive(std::move(primitive)),
      _thermal_mesh(std::move(thermal_mesh)),
      _transform(std::move(transform)) {}

const Primitive& Item::primitive() const noexcept { return _primitive; }

const ThermalMesh& Item::thermal_mesh() const noexcept { return _thermal_mesh; }

const CoordinateTransformation& Item::transform() const noexcept {
    return _transform;
}

const std::optional<MeshOptions>& Item::mesh_options_override() const noexcept {
    return _mesh_options_override;
}

void Item::set_primitive(Primitive primitive) {
    _primitive = std::move(primitive);
}

void Item::set_thermal_mesh(ThermalMesh thermal_mesh) {
    _thermal_mesh = std::move(thermal_mesh);
}

void Item::set_transform(CoordinateTransformation transform) noexcept {
    _transform = std::move(transform);
}

void Item::set_mesh_options_override(
    std::optional<MeshOptions> mesh_options_override) noexcept {
    _mesh_options_override = mesh_options_override;
}

}  // namespace pycanha::gmm
