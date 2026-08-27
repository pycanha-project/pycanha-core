#pragma once

#include <Eigen/Dense>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh {

// Populates mesh.node_numbers (indexed directly by face_id) from the
// ThermalMesh per-side node assignment. face_ids encode face_pair k = face_id /
// 2, side = face_id % 2 (even = side 1, odd = side 2). For every face pair
// touched by a face_id, both side faces [2k] and [2k+1] are filled. Faces for
// face pairs not present in the mesh (holes left by cuts) stay NO_NODE ("no
// node assigned").
inline void fill_node_numbers(TriMeshD& mesh, const ThermalMesh& thermal_mesh) {
    const pycanha::MeshIndex faces = mesh.nf();
    mesh.node_numbers.setConstant(static_cast<Eigen::Index>(faces), NO_NODE);
    if (faces == 0U) {
        return;
    }

    const auto dir1_face_pairs = static_cast<pycanha::MeshIndex>(
        thermal_mesh.get_dir1_mesh().size() - 1U);
    if (dir1_face_pairs == 0U) {
        return;
    }

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.face_ids.rows(); ++tri_idx) {
        const pycanha::MeshIndex face_id = mesh.face_ids(tri_idx);
        const pycanha::MeshIndex face_pair = face_id / 2U;
        // Face pairs run with direction 1 fastest: face_pair = i + j * n1.
        const pycanha::MeshIndex face_pair_i = face_pair % dir1_face_pairs;
        const pycanha::MeshIndex face_pair_j = face_pair / dir1_face_pairs;
        mesh.node_numbers(static_cast<Eigen::Index>(2U * face_pair)) =
            thermal_mesh.node_of(face_pair_i, face_pair_j, 1U);
        mesh.node_numbers(static_cast<Eigen::Index>((2U * face_pair) + 1U)) =
            thermal_mesh.node_of(face_pair_i, face_pair_j, 2U);
    }
}

}  // namespace pycanha::gmm::mesh
