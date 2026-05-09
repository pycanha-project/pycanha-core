#include "pycanha-core/gmm/mesh/ops/clean.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm::mesh::ops {
namespace {

using VertexKey = std::array<long long, 3>;

struct VertexKeyHash {
    [[nodiscard]] std::size_t operator()(const VertexKey& key) const noexcept {
        return static_cast<std::size_t>(key[0] * 1315423911LL +
                                        key[1] * 2654435761LL + key[2]);
    }
};

[[nodiscard]] VertexKey make_vertex_key(const Eigen::RowVector3d& vertex,
                                        double tolerance) {
    return {static_cast<long long>(std::llround(vertex.x() / tolerance)),
            static_cast<long long>(std::llround(vertex.y() / tolerance)),
            static_cast<long long>(std::llround(vertex.z() / tolerance))};
}

void compact_referenced_vertices(TriMesh& mesh) {
    std::vector<unsigned char> used(
        static_cast<std::size_t>(mesh.vertices.rows()), 0U);
    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const Eigen::Vector3i triangle = mesh.triangles.row(tri_idx);
        used[static_cast<std::size_t>(triangle[0])] = 1U;
        used[static_cast<std::size_t>(triangle[1])] = 1U;
        used[static_cast<std::size_t>(triangle[2])] = 1U;
    }

    std::vector<int> remap(static_cast<std::size_t>(mesh.vertices.rows()), -1);
    Eigen::MatrixX3d compacted_vertices(0, 3);
    compacted_vertices.resize(
        static_cast<Eigen::Index>(std::count(used.begin(), used.end(), 1U)), 3);

    Eigen::Index next_vertex = 0;
    for (Eigen::Index vertex_idx = 0; vertex_idx < mesh.vertices.rows();
         ++vertex_idx) {
        if (used[static_cast<std::size_t>(vertex_idx)] == 0U) {
            continue;
        }

        remap[static_cast<std::size_t>(vertex_idx)] =
            static_cast<int>(next_vertex);
        compacted_vertices.row(next_vertex) = mesh.vertices.row(vertex_idx);
        ++next_vertex;
    }

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        for (int corner = 0; corner < 3; ++corner) {
            mesh.triangles(tri_idx, corner) = remap[static_cast<std::size_t>(
                mesh.triangles(tri_idx, corner))];
        }
    }

    mesh.vertices = std::move(compacted_vertices);
}

}  // namespace

void dedup_vertices(TriMesh& mesh, double tolerance) {
    const double effective_tolerance = std::max(tolerance, LENGTH_TOL);
    std::unordered_map<VertexKey, Eigen::Index, VertexKeyHash> vertex_map;
    vertex_map.reserve(static_cast<std::size_t>(mesh.vertices.rows()));

    Eigen::MatrixX3d deduped_vertices(0, 3);
    deduped_vertices.resize(mesh.vertices.rows(), 3);
    std::vector<Eigen::Index> remap(
        static_cast<std::size_t>(mesh.vertices.rows()), 0);
    Eigen::Index next_vertex = 0;

    for (Eigen::Index vertex_idx = 0; vertex_idx < mesh.vertices.rows();
         ++vertex_idx) {
        const VertexKey key =
            make_vertex_key(mesh.vertices.row(vertex_idx), effective_tolerance);
        const auto [iterator, inserted] =
            vertex_map.emplace(key, static_cast<Eigen::Index>(next_vertex));
        if (inserted) {
            deduped_vertices.row(next_vertex) = mesh.vertices.row(vertex_idx);
            remap[static_cast<std::size_t>(vertex_idx)] = next_vertex;
            ++next_vertex;
        } else {
            remap[static_cast<std::size_t>(vertex_idx)] = iterator->second;
        }
    }

    deduped_vertices.conservativeResize(next_vertex, 3);
    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        for (int corner = 0; corner < 3; ++corner) {
            mesh.triangles(tri_idx, corner) =
                static_cast<int>(remap[static_cast<std::size_t>(
                    mesh.triangles(tri_idx, corner))]);
        }
    }
    mesh.vertices = std::move(deduped_vertices);
}

void remove_degenerate_triangles(TriMesh& mesh, double area_tolerance) {
    const double effective_tolerance =
        std::max(area_tolerance, LENGTH_TOL * LENGTH_TOL);
    std::vector<Eigen::Vector3i> kept_triangles;
    std::vector<std::uint64_t> kept_face_ids;
    kept_triangles.reserve(static_cast<std::size_t>(mesh.triangles.rows()));
    kept_face_ids.reserve(static_cast<std::size_t>(mesh.face_ids.rows()));

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const Eigen::Vector3i triangle = mesh.triangles.row(tri_idx);
        if ((triangle[0] == triangle[1]) || (triangle[1] == triangle[2]) ||
            (triangle[2] == triangle[0])) {
            continue;
        }

        const Vector3D p0 = mesh.vertices.row(triangle[0]).transpose();
        const Vector3D p1 = mesh.vertices.row(triangle[1]).transpose();
        const Vector3D p2 = mesh.vertices.row(triangle[2]).transpose();
        const double area = 0.5 * ((p1 - p0).cross(p2 - p0)).norm();
        if (area <= effective_tolerance) {
            continue;
        }

        kept_triangles.push_back(triangle);
        kept_face_ids.push_back(mesh.face_ids(tri_idx));
    }

    mesh.triangles.resize(static_cast<Eigen::Index>(kept_triangles.size()), 3);
    mesh.face_ids.resize(static_cast<Eigen::Index>(kept_face_ids.size()));
    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        mesh.triangles.row(tri_idx) =
            kept_triangles[static_cast<std::size_t>(tri_idx)];
        mesh.face_ids(tri_idx) =
            kept_face_ids[static_cast<std::size_t>(tri_idx)];
    }

    compact_referenced_vertices(mesh);
}

}  // namespace pycanha::gmm::mesh::ops