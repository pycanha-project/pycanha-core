#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/cutting/manifold_cut_backend.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/node_numbering.hpp"
#include "pycanha-core/gmm/ops/transform.hpp"
#include "pycanha-core/gmm/scene/scene_mesh_detail.hpp"

namespace pycanha::gmm {

bool is_closed_solid(const Primitive& primitive) noexcept {
    return std::holds_alternative<Sphere>(primitive) ||
           std::holds_alternative<Cylinder>(primitive) ||
           std::holds_alternative<Cone>(primitive) ||
           std::holds_alternative<Cube>(primitive);
}

namespace {

void validate_cutter(const std::shared_ptr<GeometryItem>& cutter) {
    if (cutter == nullptr) {
        throw std::invalid_argument("GeometryGroupCutted: null cutter");
    }
    if (cutter->id() != GeometryId{0}) {
        throw std::invalid_argument(
            "GeometryGroupCutted: cutter is already registered with a model");
    }
    if (!is_closed_solid(cutter->primitive())) {
        throw std::invalid_argument(
            "GeometryGroupCutted: cutter primitive must be a closed solid "
            "(Sphere, Cylinder, Cone, Cube)");
    }
}

}  // namespace

GeometryGroupCutted::GeometryGroupCutted(
    std::string name, std::vector<std::shared_ptr<Geometry>> targets,
    std::vector<std::shared_ptr<GeometryItem>> cutters,
    CoordinateTransformation transform)
    : Geometry(std::move(name), std::move(transform)) {
    for (auto& target : targets) {
        if (target == nullptr) {
            throw std::invalid_argument("GeometryGroupCutted: null target");
        }
        if (target->id() != GeometryId{0}) {
            throw std::invalid_argument(
                "GeometryGroupCutted: target is already registered");
        }
        _targets.push_back(target);
        _all_children.push_back(std::move(target));
    }
    for (auto& cutter : cutters) {
        cut_with(std::move(cutter));
    }
}

void GeometryGroupCutted::cut_with(std::shared_ptr<GeometryItem> cutter) {
    validate_cutter(cutter);
    if (std::find(_cutters.begin(), _cutters.end(), cutter) != _cutters.end()) {
        throw std::invalid_argument("GeometryGroupCutted: duplicate cutter");
    }
    _all_children.push_back(cutter);
    _cutters.push_back(std::move(cutter));
    on_geometry_mutated();
}

std::span<const std::shared_ptr<Geometry>> GeometryGroupCutted::targets()
    const noexcept {
    return _targets;
}

std::span<const std::shared_ptr<GeometryItem>> GeometryGroupCutted::cutters()
    const noexcept {
    return _cutters;
}

std::span<const std::shared_ptr<Geometry>> GeometryGroupCutted::children()
    const noexcept {
    return _all_children;
}

MeshOptions GeometryGroupCutted::effective_options() const {
    if (_owning_model != nullptr) {
        return _owning_model->default_mesh_options();
    }
    return MeshOptions{};
}

void GeometryGroupCutted::rebuild_mesh() const {
    const MeshOptions options = effective_options();
    const cutting::ManifoldCutBackend backend;

    std::vector<Primitive> cutter_primitives;
    cutter_primitives.reserve(_cutters.size());
    std::transform(_cutters.begin(), _cutters.end(),
                   std::back_inserter(cutter_primitives),
                   [](const std::shared_ptr<GeometryItem>& cutter) {
                       return ops::transform(cutter->primitive(),
                                             cutter->transform());
                   });

    TriMeshD combined;
    pycanha::MeshIndex offset = 0;
    for (const auto& target : _targets) {
        const auto item = std::dynamic_pointer_cast<GeometryItem>(target);
        if (item == nullptr) {
            throw std::invalid_argument(
                "GeometryGroupCutted: cut targets must be GeometryItems");
        }

        TriMeshD piece;
        if (_cutters.empty()) {
            piece = item->mesh();  // already in this group's local frame
        } else {
            piece = backend.cut(*item, cutter_primitives, item->transform(),
                                options);
            mesh::fill_node_numbers(piece, item->thermal_mesh());
            piece.primitives.assign(
                1, TriMeshD::PrimitiveRange{
                       item->id(), 0U,
                       piece.nf() > 0U ? piece.nf() - 2U : 0U});
        }
        offset = detail::concatenate_offset(combined, piece, offset);
    }

    detail::apply_transform_in_place(combined, _transform);
    _cached_mesh = std::move(combined);
}

const TriMeshD& GeometryGroupCutted::mesh() const {
    if (!_cached_mesh.has_value()) {
        rebuild_mesh();
    }
    return *_cached_mesh;
}

void GeometryGroupCutted::create_mesh() { rebuild_mesh(); }

void GeometryGroupCutted::on_geometry_mutated() { _cached_mesh.reset(); }

}  // namespace pycanha::gmm
