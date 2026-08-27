#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh::ops {

namespace detail {

using ValidateEdge = std::array<std::uint32_t, 2>;

struct ValidateEdgeHash {
    [[nodiscard]] std::size_t operator()(
        const ValidateEdge& edge) const noexcept {
        return (static_cast<std::size_t>(edge[0]) * 1315423911U) +
               static_cast<std::size_t>(edge[1]);
    }
};

[[nodiscard]] inline ValidateEdge normalized_edge(std::uint32_t lhs,
                                                  std::uint32_t rhs) noexcept {
    if (lhs < rhs) {
        return {lhs, rhs};
    }
    return {rhs, lhs};
}

template <class Scalar>
[[nodiscard]] std::unordered_map<ValidateEdge, int, ValidateEdgeHash>
edge_use_counts(const TriMesh<Scalar>& mesh) {
    std::unordered_map<ValidateEdge, int, ValidateEdgeHash> counts;
    counts.reserve(static_cast<std::size_t>(mesh.triangles.rows() * 3));

    for (Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const auto tri = mesh.triangles.row(tri_idx);
        ++counts[normalized_edge(tri(0), tri(1))];
        ++counts[normalized_edge(tri(1), tri(2))];
        ++counts[normalized_edge(tri(2), tri(0))];
    }

    return counts;
}

}  // namespace detail

template <class Scalar>
[[nodiscard]] bool is_watertight(const TriMesh<Scalar>& mesh) {
    const auto counts = detail::edge_use_counts(mesh);
    return std::all_of(counts.begin(), counts.end(),
                       [](const auto& entry) { return entry.second == 2; });
}

// A mesh is consistent when every triangle has a face id, every id is even (a
// triangle carries the side-1 id of its pair and defines both sides), and
// every id falls inside the face count the mesh declares. A cut may leave a
// face with no triangles, but never a triangle outside the declared faces.
template <class Scalar>
[[nodiscard]] bool has_consistent_face_ids(const TriMesh<Scalar>& mesh) {
    if (mesh.face_ids.rows() != mesh.triangles.rows()) {
        return false;
    }
    for (Eigen::Index tri_idx = 0; tri_idx < mesh.face_ids.rows(); ++tri_idx) {
        const auto face_id = mesh.face_ids(tri_idx);
        if ((face_id % 2U) != 0U || face_id >= mesh.nf()) {
            return false;
        }
    }
    return true;
}

}  // namespace pycanha::gmm::mesh::ops
