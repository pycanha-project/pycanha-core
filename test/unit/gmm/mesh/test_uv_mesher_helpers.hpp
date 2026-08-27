#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <unordered_set>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::test {

[[nodiscard]] inline double sum_triangle_areas(const TriMeshD& mesh) {
    return mesh::ops::compute_areas(mesh).sum();
}

[[nodiscard]] inline std::unordered_set<std::uint64_t> valid_face_ids(
    const ThermalMesh& thermal_mesh) {
    std::unordered_set<std::uint64_t> face_ids;
    const std::size_t num_dir2_face_pairs =
        thermal_mesh.get_dir2_mesh().size() - 1U;
    for (std::size_t dir1_idx = 0;
         dir1_idx + 1U < thermal_mesh.get_dir1_mesh().size(); ++dir1_idx) {
        for (std::size_t dir2_idx = 0;
             dir2_idx + 1U < thermal_mesh.get_dir2_mesh().size(); ++dir2_idx) {
            // Even local face id = side 1 (front).
            const std::size_t linear_index =
                (dir1_idx * num_dir2_face_pairs) + dir2_idx;
            face_ids.insert(2U * static_cast<std::uint64_t>(linear_index));
        }
    }
    return face_ids;
}

[[nodiscard]] inline bool face_ids_cover_all_face_pairs(
    const TriMeshD& mesh, const ThermalMesh& thermal_mesh) {
    const auto expected_ids = valid_face_ids(thermal_mesh);
    std::unordered_set<std::uint64_t> actual_ids;
    for (Eigen::Index index = 0; index < mesh.face_ids.rows(); ++index) {
        const auto face_id = mesh.face_ids[index];
        if (!expected_ids.contains(face_id)) {
            return false;
        }
        actual_ids.insert(face_id);
    }
    return actual_ids == expected_ids;
}

[[nodiscard]] inline std::size_t count_vertices_near(const TriMeshD& mesh,
                                                     const Point3D& point,
                                                     double tolerance) {
    std::size_t count = 0U;
    for (Eigen::Index index = 0; index < mesh.vertices.rows(); ++index) {
        if ((mesh.vertices.row(index).transpose() - point).norm() <=
            tolerance) {
            ++count;
        }
    }
    return count;
}

// True when every listed point appears exactly once among the mesh vertices.
// Corners are the cheapest evidence that a primitive was meshed as the shape
// it is rather than as an approximation of it.
[[nodiscard]] inline bool corners_present(const TriMeshD& mesh,
                                          std::span<const Point3D> corners,
                                          double tolerance) {
    return std::ranges::all_of(corners, [&](const Point3D& corner) {
        return count_vertices_near(mesh, corner, tolerance) == 1U;
    });
}

[[nodiscard]] inline bool has_no_degenerate_triangles(const TriMeshD& mesh,
                                                      double min_area) {
    const auto areas = mesh::ops::compute_areas(mesh);
    for (Eigen::Index index = 0; index < areas.rows(); ++index) {
        if (!(areas[index] > min_area)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline double absolute_area_error(const TriMeshD& mesh,
                                                double expected_area) {
    return std::abs(sum_triangle_areas(mesh) - expected_area);
}

}  // namespace pycanha::gmm::test
