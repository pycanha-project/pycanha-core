#include "pycanha-core/gmm/mesh/uv_mesher.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "uv_mesher_internal.hpp"

namespace pycanha::gmm {

TriMesh UvMesher::mesh(const Primitive& primitive,
                       const ThermalMesh& thermal_mesh,
                       const MeshOptions& options) const {
    return std::visit(
        [&thermal_mesh, &options](const auto& concrete_primitive) {
            return mesh::detail::mesh_primitive(concrete_primitive,
                                                thermal_mesh, options);
        },
        primitive);
}

}  // namespace pycanha::gmm

namespace pycanha::gmm::mesh::detail {
namespace {

struct QuantizedPoint {
    std::int64_t x;
    std::int64_t y;
    std::int64_t z;

    [[nodiscard]] bool operator==(const QuantizedPoint& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct QuantizedPointHash {
    [[nodiscard]] std::size_t operator()(
        const QuantizedPoint& point) const noexcept {
        std::size_t seed = static_cast<std::size_t>(point.x);
        seed ^= static_cast<std::size_t>(point.y) + 0x9e3779b9U + (seed << 6U) +
                (seed >> 2U);
        seed ^= static_cast<std::size_t>(point.z) + 0x9e3779b9U + (seed << 6U) +
                (seed >> 2U);
        return seed;
    }
};

[[nodiscard]] QuantizedPoint quantize_point(const Point3D& point) {
    return {static_cast<std::int64_t>(std::llround(point.x() / LENGTH_TOL)),
            static_cast<std::int64_t>(std::llround(point.y() / LENGTH_TOL)),
            static_cast<std::int64_t>(std::llround(point.z() / LENGTH_TOL))};
}

[[nodiscard]] double triangle_area(const Point3D& p0, const Point3D& p1,
                                   const Point3D& p2) {
    return 0.5 * ((p1 - p0).cross(p2 - p0)).norm();
}

[[nodiscard]] int fallback_arc_segments(double angle_span) {
    if (angle_span <= ANGLE_TOL) {
        return 1;
    }

    constexpr double quarter_turn = std::numbers::pi / 4.0;
    return std::max(1, static_cast<int>(std::ceil(angle_span / quarter_turn)));
}

[[nodiscard]] Eigen::Index append_vertex(
    std::vector<Point3D>& vertices,
    std::unordered_map<QuantizedPoint, Eigen::Index, QuantizedPointHash>&
        vertex_lookup,
    const Point3D& point) {
    const QuantizedPoint key = quantize_point(point);
    const auto it = vertex_lookup.find(key);
    if (it != vertex_lookup.end()) {
        return it->second;
    }

    const Eigen::Index index = static_cast<Eigen::Index>(vertices.size());
    vertices.push_back(point);
    vertex_lookup.emplace(key, index);
    return index;
}

void append_triangle(std::vector<std::array<Eigen::Index, 3>>& triangles,
                     std::vector<std::uint64_t>& face_ids,
                     const std::vector<Point3D>& vertices, Eigen::Index v0,
                     Eigen::Index v1, Eigen::Index v2,
                     std::uint64_t face_id_value) {
    if (v0 == v1 || v1 == v2 || v0 == v2) {
        return;
    }

    if (triangle_area(vertices[static_cast<std::size_t>(v0)],
                      vertices[static_cast<std::size_t>(v1)],
                      vertices[static_cast<std::size_t>(v2)]) <=
        LENGTH_TOL * LENGTH_TOL) {
        return;
    }

    triangles.push_back({v0, v1, v2});
    face_ids.push_back(face_id_value);
}

}  // namespace

double lerp(double start, double end, double t) noexcept {
    return start + (end - start) * t;
}

bool full_revolution(double start_angle, double end_angle) noexcept {
    constexpr double full_turn = 2.0 * std::numbers::pi;
    return std::abs((end_angle - start_angle) - full_turn) <= ANGLE_TOL;
}

int solve_arc_segments(double radius, double angle_span,
                       double deviation_tolerance) {
    const double positive_angle = std::abs(angle_span);
    if (positive_angle <= ANGLE_TOL || radius <= LENGTH_TOL) {
        return 1;
    }

    if (deviation_tolerance <= 0.0) {
        return fallback_arc_segments(positive_angle);
    }

    if (deviation_tolerance >= 2.0 * radius) {
        return 1;
    }

    const double cosine_argument =
        std::clamp(1.0 - deviation_tolerance / radius, -1.0, 1.0);
    const double denominator = 2.0 * std::acos(cosine_argument);
    if (denominator <= ANGLE_TOL) {
        return 1;
    }

    return std::max(1,
                    static_cast<int>(std::ceil(positive_angle / denominator)));
}

int solve_paraboloid_row_segments(double max_radius, double height,
                                  double row_start, double row_end,
                                  double deviation_tolerance) {
    if (max_radius <= LENGTH_TOL || height <= LENGTH_TOL) {
        return 1;
    }

    const double radius_start =
        max_radius * std::sqrt(std::max(row_start, 0.0));
    const double radius_end = max_radius * std::sqrt(std::max(row_end, 0.0));
    const double radius_span = std::abs(radius_end - radius_start);
    if (radius_span <= LENGTH_TOL) {
        return 1;
    }

    if (deviation_tolerance <= 0.0) {
        return std::max(
            1, static_cast<int>(std::ceil(4.0 * radius_span / max_radius)));
    }

    const double parabola_coefficient = height / (max_radius * max_radius);
    const double required =
        radius_span *
        std::sqrt(parabola_coefficient / (4.0 * deviation_tolerance));
    return std::max(1, static_cast<int>(std::ceil(required)));
}

DirSampler make_linear_dir_sampler(std::span<const double> cuts) {
    std::vector<double> local_cuts(cuts.begin(), cuts.end());
    return [local_cuts = std::move(local_cuts)](std::size_t cell_index,
                                                int step, int step_count) {
        const double t = step_count > 0 ? static_cast<double>(step) /
                                              static_cast<double>(step_count)
                                        : 0.0;
        return lerp(local_cuts[cell_index], local_cuts[cell_index + 1U], t);
    };
}

TriMesh build_mesh_from_plan(const ThermalMesh& thermal_mesh,
                             const SamplingPlan& plan) {
    const auto dir1_cuts = thermal_mesh.dir1_cuts();
    const auto dir2_cuts = thermal_mesh.dir2_cuts();
    const std::size_t num_dir1_cells = dir1_cuts.size() - 1U;
    const std::size_t num_dir2_cells = dir2_cuts.size() - 1U;

    if (plan.dir1_segments.size() != num_dir1_cells ||
        plan.dir2_segments.size() != num_dir2_cells) {
        throw std::invalid_argument("Sampling plan does not match ThermalMesh");
    }

    std::vector<Point3D> vertices;
    std::unordered_map<QuantizedPoint, Eigen::Index, QuantizedPointHash>
        vertex_lookup;
    std::vector<std::array<Eigen::Index, 3>> triangles;
    std::vector<std::uint64_t> face_ids;

    for (std::size_t dir1_idx = 0; dir1_idx < num_dir1_cells; ++dir1_idx) {
        const int dir1_segments = std::max(plan.dir1_segments[dir1_idx], 1);
        for (std::size_t dir2_idx = 0; dir2_idx < num_dir2_cells; ++dir2_idx) {
            const int dir2_segments = std::max(plan.dir2_segments[dir2_idx], 1);
            const auto face_id_value = static_cast<std::uint64_t>(
                thermal_mesh.face_id(dir1_idx, dir2_idx, Side::Front));

            std::vector<Eigen::Index> cell_vertices(static_cast<std::size_t>(
                (dir1_segments + 1) * (dir2_segments + 1)));

            const auto cell_vertex_index = [dir2_segments](int local_dir1,
                                                           int local_dir2) {
                return static_cast<std::size_t>(
                    local_dir1 * (dir2_segments + 1) + local_dir2);
            };

            for (int local_dir1 = 0; local_dir1 <= dir1_segments;
                 ++local_dir1) {
                const double dir1 =
                    plan.dir1_sample(dir1_idx, local_dir1, dir1_segments);
                for (int local_dir2 = 0; local_dir2 <= dir2_segments;
                     ++local_dir2) {
                    const double dir2 =
                        plan.dir2_sample(dir2_idx, local_dir2, dir2_segments);
                    cell_vertices[cell_vertex_index(local_dir1, local_dir2)] =
                        append_vertex(vertices, vertex_lookup,
                                      plan.point_at(dir1, dir2));
                }
            }

            for (int local_dir1 = 0; local_dir1 < dir1_segments; ++local_dir1) {
                for (int local_dir2 = 0; local_dir2 < dir2_segments;
                     ++local_dir2) {
                    const Eigen::Index v00 = cell_vertices[cell_vertex_index(
                        local_dir1, local_dir2)];
                    const Eigen::Index v10 = cell_vertices[cell_vertex_index(
                        local_dir1 + 1, local_dir2)];
                    const Eigen::Index v11 = cell_vertices[cell_vertex_index(
                        local_dir1 + 1, local_dir2 + 1)];
                    const Eigen::Index v01 = cell_vertices[cell_vertex_index(
                        local_dir1, local_dir2 + 1)];

                    append_triangle(triangles, face_ids, vertices, v00, v10,
                                    v11, face_id_value);
                    append_triangle(triangles, face_ids, vertices, v00, v11,
                                    v01, face_id_value);
                }
            }
        }
    }

    TriMesh mesh;
    mesh.vertices.resize(static_cast<Eigen::Index>(vertices.size()), 3);
    for (Eigen::Index vertex_idx = 0;
         vertex_idx < static_cast<Eigen::Index>(vertices.size());
         ++vertex_idx) {
        mesh.vertices.row(vertex_idx) =
            vertices[static_cast<std::size_t>(vertex_idx)];
    }

    mesh.triangles.resize(static_cast<Eigen::Index>(triangles.size()), 3);
    mesh.face_ids.resize(static_cast<Eigen::Index>(face_ids.size()));
    for (Eigen::Index triangle_idx = 0;
         triangle_idx < static_cast<Eigen::Index>(triangles.size());
         ++triangle_idx) {
        const auto& triangle =
            triangles[static_cast<std::size_t>(triangle_idx)];
        mesh.triangles.row(triangle_idx) << triangle[0], triangle[1],
            triangle[2];
        mesh.face_ids[triangle_idx] =
            face_ids[static_cast<std::size_t>(triangle_idx)];
    }

    return mesh;
}

Point3D triangle_strip_point(const Triangle& triangle, double dir1,
                             double dir2) {
    const Vector3D edge_1 = triangle.p2() - triangle.p1();
    const Vector3D edge_2 = triangle.p3() - triangle.p1();
    return triangle.p1() + dir1 * ((1.0 - dir2) * edge_1 + dir2 * edge_2);
}

}  // namespace pycanha::gmm::mesh::detail
