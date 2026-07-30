#pragma once

#include <Eigen/Dense>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh {

// Populates mesh.node_numbers (indexed directly by face_id) from the
// ThermalMesh per-side node assignment. face_ids encode cell k = face_id / 2,
// side = face_id % 2 (even = side 1, odd = side 2). For every cell touched by
// a face_id, both side slots [2k] and [2k+1] are filled. Slots for cells not
// present in the mesh (holes left by cuts) stay NO_NODE ("no node assigned").
inline void fill_node_numbers(TriMeshD& mesh, const ThermalMesh& thermal_mesh) {
    const pycanha::MeshIndex slots = mesh.nf();
    mesh.node_numbers.setConstant(static_cast<Eigen::Index>(slots), NO_NODE);
    if (slots == 0U) {
        return;
    }

    const auto dir1_cells = static_cast<pycanha::MeshIndex>(
        thermal_mesh.get_dir1_mesh().size() - 1U);
    if (dir1_cells == 0U) {
        return;
    }

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.face_ids.rows(); ++tri_idx) {
        const pycanha::MeshIndex face_id = mesh.face_ids(tri_idx);
        const pycanha::MeshIndex cell = face_id / 2U;
        // Cells run with direction 1 fastest: cell = i + j * n1.
        const pycanha::MeshIndex cell_i = cell % dir1_cells;
        const pycanha::MeshIndex cell_j = cell / dir1_cells;
        mesh.node_numbers(static_cast<Eigen::Index>(2U * cell)) =
            thermal_mesh.node_of(cell_i, cell_j, 1U);
        mesh.node_numbers(static_cast<Eigen::Index>((2U * cell) + 1U)) =
            thermal_mesh.node_of(cell_i, cell_j, 2U);
    }
}

}  // namespace pycanha::gmm::mesh
