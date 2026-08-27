#include "pycanha-core/gmm/scene/resolve.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/cutting/manifold_cut_backend.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/node_numbering.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/ops/transform.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/gmm/scene/scene_mesh_detail.hpp"
#include "pycanha-core/utils/parallel_for.hpp"

namespace pycanha::gmm::detail {
namespace {

// What the walk carries down one branch. `node` is a reference because a
// scene node is never null -- GeometryGroup::add and the cut group's
// constructor both reject nullptr -- and saying so in the type keeps the
// invariant local instead of spread across two other files.
struct WalkFrame {
    std::reference_wrapper<const Geometry> node;
    CoordinateTransformation to_root;     // node's own frame -> output frame
    std::vector<ResolvedCutter> cutters;  // in the output frame
    std::size_t part;
    CoordinateTransformation part_to_root;
};

[[nodiscard]] MeshOptions options_for(const GeometryItem& item) {
    if (item.mesh_options_override().has_value()) {
        return *item.mesh_options_override();
    }
    if (item.owning_model() != nullptr) {
        return item.owning_model()->default_mesh_options();
    }
    return MeshOptions{};
}

void push_children(std::vector<WalkFrame>& stack, const WalkFrame& parent,
                   std::span<const std::shared_ptr<Geometry>> children,
                   const std::vector<ResolvedCutter>& cutters) {
    // Popped from the back, so children go on in reverse to come out in tree
    // order. Face ids follow this order.
    static_cast<void>(std::ranges::transform(
        std::views::reverse(children), std::back_inserter(stack),
        [&parent, &cutters](const std::shared_ptr<Geometry>& child) {
            return WalkFrame{
                .node = *child,
                .to_root = child->transform().compose(parent.to_root),
                .cutters = cutters,
                .part = parent.part,
                .part_to_root = parent.part_to_root};
        }));
}

}  // namespace

std::vector<ResolvedTarget> collect_targets(
    const Geometry& root, const CoordinateTransformation& seed,
    const std::unordered_map<const Geometry*, std::size_t>& frame_breaks) {
    std::vector<ResolvedTarget> targets;

    std::vector<WalkFrame> stack;
    stack.push_back(WalkFrame{.node = root,
                              .to_root = seed,
                              .cutters = {},
                              .part = 0,
                              .part_to_root = {}});

    while (!stack.empty()) {
        WalkFrame frame = std::move(stack.back());
        stack.pop_back();

        // A frame break starts a new rigid sub-frame AT this node, so its own
        // transform belongs to the part placement rather than to the geometry
        // under it -- the same "expressed in my own frame" rule a resolution
        // root follows.
        if (const auto break_it = frame_breaks.find(&frame.node.get());
            break_it != frame_breaks.end()) {
            frame.part = break_it->second + 1U;
            frame.part_to_root = frame.to_root;
        }

        if (const auto* item =
                dynamic_cast<const GeometryItem*>(&frame.node.get());
            item != nullptr) {
            targets.push_back(
                ResolvedTarget{.item = *item,
                               .to_root = frame.to_root,
                               .cutters = std::move(frame.cutters),
                               .part = frame.part,
                               .part_to_root = frame.part_to_root});
            continue;
        }

        if (const auto* cut_group =
                dynamic_cast<const GeometryGroupCutted*>(&frame.node.get());
            cut_group != nullptr) {
            // This group's cutters apply to everything in its target subtree,
            // on top of whatever the subtree already inherited.
            std::vector<ResolvedCutter> cutters = std::move(frame.cutters);
            cutters.reserve(cutters.size() + cut_group->cutters().size());
            static_cast<void>(std::ranges::transform(
                cut_group->cutters(), std::back_inserter(cutters),
                [&frame](const std::shared_ptr<GeometryItem>& cutter) {
                    return ResolvedCutter{
                        .primitive = ops::transform(
                            cutter->primitive(),
                            cutter->transform().compose(frame.to_root)),
                        .part = frame.part};
                }));
            // The cutters themselves emit no faces: they are tools.
            push_children(stack, frame, cut_group->targets(), cutters);
            continue;
        }

        push_children(stack, frame, frame.node.get().children(), frame.cutters);
    }

    return targets;
}

TriMeshD mesh_target(const ResolvedTarget& target) {
    const GeometryItem& item = target.item;
    // Cutting happens in the output frame; a part's geometry is then expressed
    // in the part's own frame, which is one rigid transform away.
    const CoordinateTransformation into_frame =
        target.part == 0 ? CoordinateTransformation{}
                         : target.part_to_root.inverse();
    const CoordinateTransformation to_frame =
        target.to_root.compose(into_frame);

    TriMeshD piece;
    if (target.cutters.empty()) {
        // Reuse the item's cached uncut mesh. It is in the item's parent
        // frame, so its own transform is already applied; undo that and put it
        // where this resolution wants it.
        piece = item.mesh();
        apply_transform_in_place(piece,
                                 item.transform().inverse().compose(to_frame));
    } else {
        std::vector<Primitive> cutters;
        cutters.reserve(target.cutters.size());
        static_cast<void>(std::ranges::transform(
            target.cutters, std::back_inserter(cutters),
            [&target, &into_frame](const ResolvedCutter& cutter) {
                return target.part == 0
                           ? cutter.primitive
                           : ops::transform(cutter.primitive, into_frame);
            }));
        const cutting::ManifoldCutBackend backend;
        piece = backend.cut(item, cutters, to_frame, options_for(item));
        mesh::fill_node_numbers(piece, item.thermal_mesh());
    }

    piece.primitives.assign(
        1, TriMeshD::PrimitiveRange{
               .geometry_id = item.id(),
               .first_face_id = 0U,
               .last_face_id = piece.nf() > 0U ? piece.nf() - 2U : 0U});
    return piece;
}

TriMeshD concatenate_at(std::span<const TriMeshD> pieces,
                        std::span<const pycanha::MeshIndex> face_offsets) {
    Eigen::Index total_vertices = 0;
    Eigen::Index total_triangles = 0;
    std::size_t total_ranges = 0;
    pycanha::MeshIndex total_faces = 0;
    for (std::size_t index = 0; index < pieces.size(); ++index) {
        const TriMeshD& piece = pieces[index];
        total_vertices += piece.vertices.rows();
        total_triangles += piece.triangles.rows();
        total_ranges += piece.primitives.size();
        total_faces = std::max(total_faces, face_offsets[index] + piece.nf());
    }

    TriMeshD combined;
    combined.num_faces = total_faces;
    combined.vertices.resize(total_vertices, 3);
    combined.triangles.resize(total_triangles, 3);
    combined.face_ids.resize(total_triangles);
    combined.node_numbers.setConstant(static_cast<Eigen::Index>(total_faces),
                                      NO_NODE);
    combined.primitives.reserve(total_ranges);

    Eigen::Index vertex_offset = 0;
    Eigen::Index triangle_offset = 0;
    for (std::size_t index = 0; index < pieces.size(); ++index) {
        const TriMeshD& piece = pieces[index];
        const pycanha::MeshIndex face_offset = face_offsets[index];

        if (piece.vertices.rows() > 0) {
            combined.vertices.middleRows(vertex_offset, piece.vertices.rows()) =
                piece.vertices;
        }
        if (piece.triangles.rows() > 0) {
            combined.triangles.middleRows(triangle_offset,
                                          piece.triangles.rows()) =
                (piece.triangles.array() +
                 static_cast<pycanha::MeshIndex>(vertex_offset))
                    .matrix();
            combined.face_ids.segment(triangle_offset, piece.face_ids.rows()) =
                (piece.face_ids.array() + face_offset).matrix();
        }
        if (piece.node_numbers.rows() > 0) {
            combined.node_numbers.segment(
                face_offset, piece.node_numbers.rows()) = piece.node_numbers;
        }
        for (const auto& range : piece.primitives) {
            combined.primitives.push_back(TriMeshD::PrimitiveRange{
                .geometry_id = range.geometry_id,
                .first_face_id = range.first_face_id + face_offset,
                .last_face_id = range.last_face_id + face_offset});
        }

        vertex_offset += piece.vertices.rows();
        triangle_offset += piece.triangles.rows();
    }

    return combined;
}

TriMeshD concatenate_in_order(std::span<const TriMeshD> pieces) {
    std::vector<pycanha::MeshIndex> face_offsets;
    face_offsets.reserve(pieces.size());
    pycanha::MeshIndex next_face = 0;
    for (const TriMeshD& piece : pieces) {
        face_offsets.push_back(next_face);
        next_face += piece.nf();
    }
    return concatenate_at(pieces, face_offsets);
}

TriMeshD resolve_subtree(const Geometry& root) {
    const auto targets = collect_targets(root, root.transform());
    std::vector<TriMeshD> pieces(targets.size());

    // An uncut item only moves its cached mesh into place, and filling that
    // cache mutates the item, so those stay on this thread.
    std::vector<std::size_t> cut_targets;
    for (std::size_t index = 0; index < targets.size(); ++index) {
        if (targets[index].cutters.empty()) {
            pieces[index] = mesh_target(targets[index]);
        } else {
            cut_targets.push_back(index);
        }
    }

    // The boolean cuts are where essentially all the time goes, they do not
    // touch each other, and each writes only its own face -- so the worker
    // count cannot change the result, only how long it takes.
    const unsigned workers =
        cut_targets.size() > 1U
            ? std::min<unsigned>(std::thread::hardware_concurrency(),
                                 static_cast<unsigned>(cut_targets.size()))
            : 1U;
    utils::parallel_for_index(cut_targets.size(), workers,
                              [&](std::size_t index) {
                                  pieces[cut_targets[index]] =
                                      mesh_target(targets[cut_targets[index]]);
                              });

    return concatenate_in_order(pieces);
}

}  // namespace pycanha::gmm::detail
