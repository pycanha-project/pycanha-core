#include "pycanha-core/gmm/mesh/ops/subset.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh::ops {

TriMesh extract_by_face_id(const TriMesh& mesh,
                           std::span<const FaceId> face_ids) {
    std::unordered_set<std::uint64_t> wanted_face_ids;
    wanted_face_ids.reserve(face_ids.size());
    for (const FaceId face_id : face_ids) {
        wanted_face_ids.insert(to_raw(face_id));
    }

    std::vector<int> remap(static_cast<std::size_t>(mesh.vertices.rows()), -1);
    std::vector<Eigen::Index> used_vertices;
    std::vector<Eigen::Vector3i> kept_triangles;
    std::vector<std::uint64_t> kept_face_ids;

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        if (!wanted_face_ids.contains(mesh.face_ids(tri_idx))) {
            continue;
        }

        Eigen::Vector3i triangle;
        for (int corner = 0; corner < 3; ++corner) {
            const Eigen::Index source_vertex = mesh.triangles(tri_idx, corner);
            int& target_vertex = remap[static_cast<std::size_t>(source_vertex)];
            if (target_vertex < 0) {
                target_vertex = static_cast<int>(used_vertices.size());
                used_vertices.push_back(source_vertex);
            }
            triangle[corner] = target_vertex;
        }
        kept_triangles.push_back(triangle);
        kept_face_ids.push_back(mesh.face_ids(tri_idx));
    }

    TriMesh subset;
    subset.vertices.resize(static_cast<Eigen::Index>(used_vertices.size()), 3);
    subset.triangles.resize(static_cast<Eigen::Index>(kept_triangles.size()),
                            3);
    subset.face_ids.resize(static_cast<Eigen::Index>(kept_face_ids.size()));

    for (Eigen::Index vertex_idx = 0;
         vertex_idx < static_cast<Eigen::Index>(used_vertices.size());
         ++vertex_idx) {
        subset.vertices.row(vertex_idx) = mesh.vertices.row(
            used_vertices[static_cast<std::size_t>(vertex_idx)]);
    }
    for (Eigen::Index tri_idx = 0;
         tri_idx < static_cast<Eigen::Index>(kept_triangles.size());
         ++tri_idx) {
        subset.triangles.row(tri_idx) =
            kept_triangles[static_cast<std::size_t>(tri_idx)];
        subset.face_ids(tri_idx) =
            kept_face_ids[static_cast<std::size_t>(tri_idx)];
    }

    return subset;
}

}  // namespace pycanha::gmm::mesh::ops
