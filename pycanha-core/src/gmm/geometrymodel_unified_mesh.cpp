#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/cutting/manifold_cut_backend.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/sort.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/mesh/unified_trimesh.hpp"
#include "pycanha-core/gmm/mesh/uv_mesher.hpp"
#include "pycanha-core/gmm/ops/transform.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/cut_group.hpp"
#include "pycanha-core/gmm/scene/group.hpp"
#include "pycanha-core/gmm/scene/item.hpp"

namespace pycanha::gmm {
namespace {

void append_into_unified(UnifiedTriMesh& unified_mesh,
                         const TriMesh& local_mesh,
                         const CoordinateTransformation& transform,
                         GeometryId geometry_id) {
    if ((local_mesh.vertices.rows() == 0) ||
        (local_mesh.triangles.rows() == 0)) {
        return;
    }

    PYCANHA_ASSERT(local_mesh.face_ids.rows() == local_mesh.triangles.rows(),
                   "Each triangle must carry one face id");

    const Index vertex_offset = unified_mesh.vertices.rows();
    const Index triangle_offset = unified_mesh.triangles.rows();

    unified_mesh.vertices.conservativeResize(
        vertex_offset + local_mesh.vertices.rows(), 3);
    for (Index vertex_idx = 0; vertex_idx < local_mesh.vertices.rows();
         ++vertex_idx) {
        unified_mesh.vertices.row(vertex_offset + vertex_idx) =
            transform.apply(local_mesh.vertices.row(vertex_idx).transpose())
                .transpose();
    }

    unified_mesh.triangles.conservativeResize(
        triangle_offset + local_mesh.triangles.rows(), 3);
    unified_mesh.face_ids.conservativeResize(triangle_offset +
                                             local_mesh.triangles.rows());
    unified_mesh.geometry_ids.conservativeResize(triangle_offset +
                                                 local_mesh.triangles.rows());

    for (Index tri_idx = 0; tri_idx < local_mesh.triangles.rows(); ++tri_idx) {
        unified_mesh.triangles.row(triangle_offset + tri_idx) =
            local_mesh.triangles.row(tri_idx).array() + vertex_offset;
        unified_mesh.face_ids(triangle_offset + tri_idx) =
            local_mesh.face_ids(tri_idx);
        unified_mesh.geometry_ids(triangle_offset + tri_idx) =
            to_raw(geometry_id);
    }
}

}  // namespace

const UnifiedTriMesh& GeometryModel::unified_mesh() const {
    if (!_unified_mesh_dirty) {
        return _cached_unified_mesh;
    }

    _cached_unified_mesh = UnifiedTriMesh{};
    const UvMesher mesher;
    const cutting::ManifoldCutBackend cut_backend;

    struct Frame {
        Kind kind;
        std::uint32_t index;
        CoordinateTransformation accumulated_transform;
        std::vector<Primitive> active_cutters;
    };

    std::vector<Frame> stack{
        {Kind::Group, _root_group_index, CoordinateTransformation{}, {}}};
    while (!stack.empty()) {
        const Frame frame = stack.back();
        stack.pop_back();

        if (!is_active_group(frame.kind, frame.index)) {
            continue;
        }

        const Group& group = group_ref(frame.kind, frame.index);
        const CoordinateTransformation group_transform =
            group.transform().compose(frame.accumulated_transform);
        std::vector<Primitive> active_cutters = frame.active_cutters;

        if (frame.kind == Kind::CutGroup) {
            const CutGroup& cut_group = _cut_groups.at(frame.index);
            active_cutters.reserve(active_cutters.size() +
                                   cut_group.cutters().size());
            std::transform(cut_group.cutters().begin(),
                           cut_group.cutters().end(),
                           std::back_inserter(active_cutters),
                           [&group_transform](const Primitive& cutter) {
                               return ops::transform(cutter, group_transform);
                           });
        }

        for (const std::uint32_t item_index : group.child_item_indices()) {
            if ((item_index >= _item_active.size()) ||
                (_item_active[item_index] == 0U)) {
                continue;
            }

            const Item& item = _items[item_index];
            const MeshOptions mesh_options =
                item.mesh_options_override().value_or(_default_mesh_options);
            const CoordinateTransformation item_transform =
                item.transform().compose(group_transform);
            const bool has_active_cutters = !active_cutters.empty();
            const TriMesh local_mesh =
                has_active_cutters
                    ? cut_backend.cut(item, active_cutters, item_transform,
                                      mesh_options)
                    : mesher.mesh(item.primitive(), item.thermal_mesh(),
                                  mesh_options);
            append_into_unified(_cached_unified_mesh, local_mesh,
                                has_active_cutters ? CoordinateTransformation{}
                                                   : item_transform,
                                make_geometry_id(Kind::Item, item_index));
        }

        std::transform(group.child_cut_group_indices().rbegin(),
                       group.child_cut_group_indices().rend(),
                       std::back_inserter(stack),
                       [&group_transform,
                        &active_cutters](const std::uint32_t cut_group_index) {
                           return Frame{Kind::CutGroup, cut_group_index,
                                        group_transform, active_cutters};
                       });
        std::transform(group.child_group_indices().rbegin(),
                       group.child_group_indices().rend(),
                       std::back_inserter(stack),
                       [&group_transform,
                        &active_cutters](const std::uint32_t group_index) {
                           return Frame{Kind::Group, group_index,
                                        group_transform, active_cutters};
                       });
    }

    if (_cached_unified_mesh.triangles.rows() > 0) {
        mesh::ops::apply_permutation(
            _cached_unified_mesh,
            mesh::ops::permutation_by_face_id(_cached_unified_mesh));
    }

    _unified_mesh_dirty = false;
    return _cached_unified_mesh;
}

}  // namespace pycanha::gmm
