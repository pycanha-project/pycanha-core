#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <unordered_map>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace pycanha::gmm::detail {

// One meshable item found under a resolution root, with everything needed to
// mesh it in one backend call.
//
// Cuts are resolved TOP-DOWN from a resolution root, per item, all at once:
// `node.mesh()` means "this subtree, resolved with `node` itself as the
// resolution root, expressed in `node`'s parent frame", and only cutters
// inside the subtree apply. That is what makes a chain of cuts work --
// `(A - c1) - c2` is A cut by {c1, c2} in one operation -- and it is the only
// thing that can work, because the cut backend re-classifies every surviving
// triangle back onto the target's PRIMITIVE. A cut result is a triangle soup
// with no primitive, so it can never itself be a cut target.
//
// A consequence worth stating: a parent's mesh is no longer the concatenation
// of its children's meshes, and the same child legitimately resolves
// differently under two different roots.
// One cutter that applies to a target, in the resolution's output frame.
struct ResolvedCutter {
    Primitive primitive;
    // The part in force where the cut group sits. A cutter whose part differs
    // from the target's straddles a rigid-part boundary: the subtraction is
    // baked into the part at resolution time, so moving that part afterwards
    // leaves the hole behind. mesh_parts reports it rather than pretending
    // otherwise.
    std::size_t part = 0;
};

struct ResolvedTarget {
    // A reference, not a pointer: a target without an item is not a thing.
    std::reference_wrapper<const GeometryItem> item;
    // The item's own frame -> the resolution root's output frame.
    CoordinateTransformation to_root;
    // Every cutter of every enclosing cut group whose target subtree contains
    // this item, expressed in the same output frame.
    std::vector<ResolvedCutter> cutters;
    // 0 for the resolution root itself; otherwise 1 + the index of the
    // innermost enclosing frame break (see collect_targets).
    std::size_t part = 0;
    // That frame break's own frame -> the output frame. Identity for part 0.
    CoordinateTransformation part_to_root;
};

// Walks `root`'s subtree once, depth-first in tree order, and returns every
// item that produces faces. Cutter items are tools, not geometry, so they emit
// nothing.
//
// `seed` is the transform from `root`'s own frame to the output frame; pass
// `root.transform()` to get the subtree in `root`'s parent frame.
//
// `frame_breaks` names nodes that start a rigid sub-frame (the raytracer's
// articulated parts). Targets under one record its index in `part` and its
// placement in `part_to_root`; cutting still happens in the output frame, so a
// cutter that straddles a break is baked into the part it cuts -- see
// ResolvedCutter::part.
[[nodiscard]] std::vector<ResolvedTarget> collect_targets(
    const Geometry& root, const CoordinateTransformation& seed,
    const std::unordered_map<const Geometry*, std::size_t>& frame_breaks = {});

// The mesh of one target in the output frame: the item's cached uncut mesh
// moved into place when it has no cutters, one backend call when it has. Node
// numbers are filled and the face count and primitive range are stamped.
[[nodiscard]] TriMeshD mesh_target(const ResolvedTarget& target);

// Concatenates `pieces`, placing piece i's faces at face_offsets[i]. The
// result reports one past the highest face id it reaches, so a mesh holding a
// SUBSET of a model's faces -- a ScenePart, whose face ids stay global --
// still bounds the ids it can reference.
//
// Both overloads size every array in a single pass first. Appending one piece
// at a time means an Eigen conservativeResize per piece, and Eigen has no
// capacity concept, so each one reallocates and copies everything accumulated
// so far: assembling n pieces that way moves O(n^2) bytes.
[[nodiscard]] TriMeshD concatenate_at(
    std::span<const TriMeshD> pieces,
    std::span<const pycanha::MeshIndex> face_offsets);

// Concatenates `pieces` back to back, each starting where the previous ended.
[[nodiscard]] TriMeshD concatenate_in_order(std::span<const TriMeshD> pieces);

// collect + mesh + concatenate: the whole subtree of `root`, resolved with
// `root` as the resolution root, expressed in `root`'s parent frame.
[[nodiscard]] TriMeshD resolve_subtree(const Geometry& root);

}  // namespace pycanha::gmm::detail
