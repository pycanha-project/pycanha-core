#include "pycanha-core/gmm/mesh/ops/validate.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <unordered_map>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh::ops {
namespace {

using Edge = std::array<int, 2>;

struct EdgeHash {
    [[nodiscard]] std::size_t operator()(const Edge& edge) const noexcept {
        return static_cast<std::size_t>(edge[0]) * 1315423911U +
               static_cast<std::size_t>(edge[1]);
    }
};

[[nodiscard]] Edge normalized_edge(int lhs, int rhs) noexcept {
    if (lhs < rhs) {
        return {lhs, rhs};
    }
    return {rhs, lhs};
}

[[nodiscard]] std::unordered_map<Edge, int, EdgeHash> edge_use_counts(
    const TriMesh& mesh) {
    std::unordered_map<Edge, int, EdgeHash> counts;
    counts.reserve(static_cast<std::size_t>(mesh.triangles.rows() * 3));

    for (Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const Eigen::Vector3i tri = mesh.triangles.row(tri_idx);
        ++counts[normalized_edge(tri[0], tri[1])];
        ++counts[normalized_edge(tri[1], tri[2])];
        ++counts[normalized_edge(tri[2], tri[0])];
    }

    return counts;
}

}  // namespace

bool is_watertight(const TriMesh& mesh) {
    const auto counts = edge_use_counts(mesh);
    return std::all_of(counts.begin(), counts.end(),
                       [](const auto& entry) { return entry.second == 2; });
}

bool is_manifold(const TriMesh& mesh) {
    const auto counts = edge_use_counts(mesh);
    return std::all_of(counts.begin(), counts.end(),
                       [](const auto& entry) { return entry.second <= 2; });
}

bool is_watertight_on_curved_edges(const TriMesh& mesh) {
    return is_manifold(mesh);
}

bool has_consistent_face_ids(const TriMesh& mesh) {
    return mesh.face_ids.rows() == mesh.triangles.rows();
}

}  // namespace pycanha::gmm::mesh::ops
