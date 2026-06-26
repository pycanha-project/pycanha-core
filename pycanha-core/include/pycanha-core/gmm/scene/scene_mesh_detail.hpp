#pragma once

#include <Eigen/Dense>
#include <utility>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"

namespace pycanha::gmm::detail {

// Applies `transform` to every vertex of `mesh` in place (full-precision).
inline void apply_transform_in_place(
    TriMeshD& mesh, const CoordinateTransformation& transform) {
    if (transform.is_identity()) {
        return;
    }
    for (Eigen::Index vertex_idx = 0; vertex_idx < mesh.vertices.rows();
         ++vertex_idx) {
        mesh.vertices.row(vertex_idx) =
            transform.apply(mesh.vertices.row(vertex_idx).transpose())
                .transpose();
    }
}

// Appends `src` into `dest`, shifting src's face_ids / node_numbers /
// primitive ranges by `face_id_offset` and its triangle indices by the current
// vertex count. Advances `face_id_offset` in place by src.nf() so it can be
// threaded through successive calls. node_numbers is kept dense, indexed by
// global face_id.
inline void concatenate_offset(TriMeshD& dest, const TriMeshD& src,
                               pycanha::MeshIndex& face_id_offset) {
    const pycanha::MeshIndex src_nf = src.nf();
    const Eigen::Index vertex_offset = dest.vertices.rows();
    const Eigen::Index tri_offset = dest.triangles.rows();

    if (src.vertices.rows() > 0) {
        dest.vertices.conservativeResize(vertex_offset + src.vertices.rows(),
                                         3);
        dest.vertices.bottomRows(src.vertices.rows()) = src.vertices;
    }

    if (src.triangles.rows() > 0) {
        dest.triangles.conservativeResize(tri_offset + src.triangles.rows(), 3);
        dest.triangles.bottomRows(src.triangles.rows()) =
            (src.triangles.array() +
             static_cast<pycanha::MeshIndex>(vertex_offset))
                .matrix();

        dest.face_ids.conservativeResize(tri_offset + src.face_ids.rows());
        dest.face_ids.bottomRows(src.face_ids.rows()) =
            (src.face_ids.array() + face_id_offset).matrix();
    }

    if (src_nf > 0) {
        const pycanha::MeshIndex needed = face_id_offset + src_nf;
        const Eigen::Index old_size = dest.node_numbers.rows();
        if (std::cmp_greater(needed, old_size)) {
            dest.node_numbers.conservativeResize(needed);
            dest.node_numbers
                .segment(old_size, static_cast<Eigen::Index>(needed) - old_size)
                .setZero();
        }
        if (src.node_numbers.rows() > 0) {
            dest.node_numbers.segment(face_id_offset, src.node_numbers.rows()) =
                src.node_numbers;
        }
    }

    dest.primitives.reserve(dest.primitives.size() + src.primitives.size());
    for (const auto& range : src.primitives) {
        dest.primitives.push_back(TriMeshD::PrimitiveRange{
            .geometry_id = range.geometry_id,
            .first_face_id = range.first_face_id + face_id_offset,
            .last_face_id = range.last_face_id + face_id_offset});
    }

    face_id_offset += src_nf;
}

}  // namespace pycanha::gmm::detail
