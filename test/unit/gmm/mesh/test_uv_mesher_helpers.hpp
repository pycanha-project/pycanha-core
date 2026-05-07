#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <unordered_set>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::test {

[[nodiscard]] inline double sum_triangle_areas(const TriMesh& mesh) {
    return mesh::ops::compute_areas(mesh).sum();
}

[[nodiscard]] inline std::unordered_set<std::uint64_t> valid_face_ids(
    const ThermalMesh& thermal_mesh) {
    std::unordered_set<std::uint64_t> face_ids;
    for (std::size_t dir1_idx = 0;
         dir1_idx + 1U < thermal_mesh.dir1_cuts().size(); ++dir1_idx) {
        for (std::size_t dir2_idx = 0;
             dir2_idx + 1U < thermal_mesh.dir2_cuts().size(); ++dir2_idx) {
            face_ids.insert(static_cast<std::uint64_t>(
                thermal_mesh.face_id(dir1_idx, dir2_idx, Side::Front)));
        }
    }
    return face_ids;
}

[[nodiscard]] inline bool face_ids_cover_all_cells(
    const TriMesh& mesh, const ThermalMesh& thermal_mesh) {
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

[[nodiscard]] inline std::size_t count_vertices_near(const TriMesh& mesh,
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

[[nodiscard]] inline bool has_no_degenerate_triangles(const TriMesh& mesh,
                                                      double min_area) {
    const auto areas = mesh::ops::compute_areas(mesh);
    for (Eigen::Index index = 0; index < areas.rows(); ++index) {
        if (!(areas[index] > min_area)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline double absolute_area_error(const TriMesh& mesh,
                                                double expected_area) {
    return std::abs(sum_triangle_areas(mesh) - expected_area);
}

}  // namespace pycanha::gmm::test
