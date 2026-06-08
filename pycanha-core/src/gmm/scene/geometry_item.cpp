#include "pycanha-core/gmm/scene/geometry_item.hpp"

#include <utility>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/uv_mesher.hpp"
#include "pycanha-core/gmm/scene/scene_mesh_detail.hpp"

namespace pycanha::gmm {

GeometryItem::GeometryItem(std::string name, Primitive primitive,
                           ThermalMesh thermal_mesh,
                           CoordinateTransformation transform)
    : Geometry(std::move(name), std::move(transform)),
      _primitive(std::move(primitive)),
      _thermal_mesh(std::move(thermal_mesh)) {}

const Primitive& GeometryItem::primitive() const noexcept { return _primitive; }

const ThermalMesh& GeometryItem::thermal_mesh() const noexcept {
    return _thermal_mesh;
}

const std::optional<MeshOptions>& GeometryItem::mesh_options_override()
    const noexcept {
    return _mesh_options_override;
}

void GeometryItem::set_primitive(Primitive primitive) {
    _primitive = std::move(primitive);
    on_geometry_mutated();
}

void GeometryItem::set_thermal_mesh(ThermalMesh thermal_mesh) {
    _thermal_mesh = std::move(thermal_mesh);
    on_geometry_mutated();
}

void GeometryItem::set_mesh_options_override(
    std::optional<MeshOptions> mesh_options_override) {
    _mesh_options_override = std::move(mesh_options_override);
    on_geometry_mutated();
}

std::span<const std::shared_ptr<Geometry>> GeometryItem::children()
    const noexcept {
    return {};
}

MeshOptions GeometryItem::effective_options() const {
    if (_mesh_options_override.has_value()) {
        return *_mesh_options_override;
    }
    if (_owning_model != nullptr) {
        return _owning_model->default_mesh_options();
    }
    return MeshOptions{};
}

void GeometryItem::rebuild_mesh() const {
    const UvMesher mesher;
    TriMeshD built =
        mesher.mesh(_primitive, _thermal_mesh, effective_options());
    // Stamp provenance: a single primitive range spanning this item's faces.
    built.primitives.assign(
        1, TriMeshD::PrimitiveRange{
               _id, 0U, built.nf() > 0U ? built.nf() - 2U : 0U});
    detail::apply_transform_in_place(built, _transform);
    _cached_mesh = std::move(built);
}

const TriMeshD& GeometryItem::mesh() const {
    if (!_cached_mesh.has_value()) {
        rebuild_mesh();
    }
    return *_cached_mesh;
}

void GeometryItem::create_mesh() { rebuild_mesh(); }

void GeometryItem::on_geometry_mutated() { _cached_mesh.reset(); }

}  // namespace pycanha::gmm
