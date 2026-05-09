#include "pycanha-core/gmm/mesh/ops/boundary_edges.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh::ops {
namespace {

using Edge = std::array<Eigen::Index, 2>;

struct EdgeHash {
    [[nodiscard]] std::size_t operator()(const Edge& edge) const noexcept {
        return static_cast<std::size_t>(edge[0] * 1315423911U + edge[1]);
    }
};

[[nodiscard]] Edge normalized_edge(Eigen::Index lhs, Eigen::Index rhs) {
    return lhs < rhs ? Edge{lhs, rhs} : Edge{rhs, lhs};
}

}  // namespace

std::vector<std::vector<Eigen::Index>> boundary_edge_loops(
    const TriMesh& mesh) {
    std::unordered_map<Edge, int, EdgeHash> edge_use_counts;
    std::vector<Edge> directed_edges;
    edge_use_counts.reserve(
        static_cast<std::size_t>(mesh.triangles.rows() * 3));
    directed_edges.reserve(static_cast<std::size_t>(mesh.triangles.rows() * 3));

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const Eigen::Vector3i triangle = mesh.triangles.row(tri_idx);
        const Edge edges[] = {{triangle[0], triangle[1]},
                              {triangle[1], triangle[2]},
                              {triangle[2], triangle[0]}};
        for (const Edge& edge : edges) {
            ++edge_use_counts[normalized_edge(edge[0], edge[1])];
            directed_edges.push_back(edge);
        }
    }

    std::vector<Edge> boundaries;
    for (const Edge& edge : directed_edges) {
        if (edge_use_counts[normalized_edge(edge[0], edge[1])] == 1) {
            boundaries.push_back(edge);
        }
    }

    std::vector<unsigned char> used(boundaries.size(), 0U);
    std::vector<std::vector<Eigen::Index>> loops;

    for (std::size_t start_idx = 0; start_idx < boundaries.size();
         ++start_idx) {
        if (used[start_idx] != 0U) {
            continue;
        }

        std::vector<Eigen::Index> loop{boundaries[start_idx][0],
                                       boundaries[start_idx][1]};
        used[start_idx] = 1U;

        while (true) {
            const Eigen::Index loop_end = loop.back();
            if (loop_end == loop.front()) {
                loop.pop_back();
                break;
            }

            bool found_next = false;
            for (std::size_t edge_idx = 0; edge_idx < boundaries.size();
                 ++edge_idx) {
                if ((used[edge_idx] != 0U) ||
                    (boundaries[edge_idx][0] != loop_end)) {
                    continue;
                }
                loop.push_back(boundaries[edge_idx][1]);
                used[edge_idx] = 1U;
                found_next = true;
                break;
            }

            if (!found_next) {
                break;
            }
        }

        if (loop.size() >= 2U) {
            loops.push_back(std::move(loop));
        }
    }

    return loops;
}

}  // namespace pycanha::gmm::mesh::ops