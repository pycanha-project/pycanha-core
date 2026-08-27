#include "pycanha-core/gmm/cutting/cut_mesh_ops.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"
#include "pycanha-core/gmm/primitives/triangular_prism.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha::gmm::cutting {
namespace {

// ---- vertex dedup / compaction (was mesh::ops clean.cpp) ------------------

using VertexKey = std::array<std::int64_t, 3>;

struct VertexKeyHash {
    [[nodiscard]] std::size_t operator()(const VertexKey& key) const noexcept {
        return static_cast<std::size_t>((key[0] * 1315423911LL) +
                                        (key[1] * 2654435761LL) + key[2]);
    }
};

[[nodiscard]] VertexKey make_vertex_key(const Eigen::RowVector3d& vertex,
                                        double tolerance) {
    return {static_cast<std::int64_t>(std::llround(vertex.x() / tolerance)),
            static_cast<std::int64_t>(std::llround(vertex.y() / tolerance)),
            static_cast<std::int64_t>(std::llround(vertex.z() / tolerance))};
}

void compact_referenced_vertices(TriMeshD& mesh) {
    std::vector<unsigned char> used(
        static_cast<std::size_t>(mesh.vertices.rows()), 0U);
    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        used[static_cast<std::size_t>(mesh.triangles(tri_idx, 0))] = 1U;
        used[static_cast<std::size_t>(mesh.triangles(tri_idx, 1))] = 1U;
        used[static_cast<std::size_t>(mesh.triangles(tri_idx, 2))] = 1U;
    }

    std::vector<pycanha::MeshIndex> remap(
        static_cast<std::size_t>(mesh.vertices.rows()), 0U);
    Eigen::MatrixX3d compacted_vertices(
        static_cast<Eigen::Index>(std::count(used.begin(), used.end(), 1U)), 3);

    Eigen::Index next_vertex = 0;
    for (Eigen::Index vertex_idx = 0; vertex_idx < mesh.vertices.rows();
         ++vertex_idx) {
        if (used[static_cast<std::size_t>(vertex_idx)] == 0U) {
            continue;
        }
        remap[static_cast<std::size_t>(vertex_idx)] =
            static_cast<pycanha::MeshIndex>(next_vertex);
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

// ---- centroid -> uv face-pair classification (was face_id_from_uv.cpp)
// ---------

[[nodiscard]] std::size_t find_face_pair(std::span<const double> cuts,
                                         double normalized_value) {
    const double clamped = std::clamp(normalized_value, 0.0, 1.0);
    const auto upper = std::ranges::upper_bound(cuts, clamped);
    if (upper == cuts.begin()) {
        return 0U;
    }
    if (upper == cuts.end()) {
        return cuts.size() - 2U;
    }
    return static_cast<std::size_t>(std::distance(cuts.begin(), upper) - 1);
}

[[nodiscard]] double normalize_linear(double value, double max_value) {
    return max_value > LENGTH_TOL ? value / max_value : 0.0;
}

[[nodiscard]] double clamp_periodic_angle(double angle, double start,
                                          double end) noexcept {
    constexpr double full_turn = 2.0 * std::numbers::pi;
    double best = start;
    double best_delta = std::numeric_limits<double>::infinity();

    for (const double shifted : {angle - full_turn, angle, angle + full_turn}) {
        const double candidate = std::clamp(shifted, start, end);
        const double delta = std::abs(shifted - candidate);
        if (delta < best_delta) {
            best = candidate;
            best_delta = delta;
        }
    }

    return best;
}

// Planar primitives already parametrise on [0, 1]^2, so there is nothing to
// normalise. The curved ones below still measure uv in arc length and height.
[[nodiscard]] std::pair<double, double> normalized_uv(
    const Triangle& /*triangle*/, const Point2D& uv) {
    return {std::clamp(uv.x(), 0.0, 1.0), std::clamp(uv.y(), 0.0, 1.0)};
}

[[nodiscard]] std::pair<double, double> normalized_uv(
    const Rectangle& /*rectangle*/, const Point2D& uv) {
    return {std::clamp(uv.x(), 0.0, 1.0), std::clamp(uv.y(), 0.0, 1.0)};
}

[[nodiscard]] std::pair<double, double> normalized_uv(
    const Quadrilateral& /*quadrilateral*/, const Point2D& uv) {
    return {std::clamp(uv.x(), 0.0, 1.0), std::clamp(uv.y(), 0.0, 1.0)};
}

[[nodiscard]] std::pair<double, double> normalized_uv(const Disc& disc,
                                                      const Point2D& uv) {
    const double radius = uv.y();
    const double theta =
        radius > LENGTH_TOL ? uv.x() / radius : disc.start_angle();
    return {(clamp_periodic_angle(theta, disc.start_angle(), disc.end_angle()) -
             disc.start_angle()) /
                (disc.end_angle() - disc.start_angle()),
            (radius - disc.inner_radius()) /
                (disc.outer_radius() - disc.inner_radius())};
}

[[nodiscard]] std::pair<double, double> normalized_uv(const Cylinder& cylinder,
                                                      const Point2D& uv) {
    const double theta = uv.x() / cylinder.radius();
    return {(clamp_periodic_angle(theta, cylinder.start_angle(),
                                  cylinder.end_angle()) -
             cylinder.start_angle()) /
                (cylinder.end_angle() - cylinder.start_angle()),
            normalize_linear(uv.y(), (cylinder.p2() - cylinder.p1()).norm())};
}

[[nodiscard]] std::pair<double, double> normalized_uv(const Cone& cone,
                                                      const Point2D& uv) {
    const double total_height = (cone.p2() - cone.p1()).norm();
    const double height_fraction = normalize_linear(uv.y(), total_height);
    const double radius =
        cone.radius1() + ((cone.radius2() - cone.radius1()) * height_fraction);
    const double theta =
        radius > LENGTH_TOL ? uv.x() / radius : cone.start_angle();
    return {(clamp_periodic_angle(theta, cone.start_angle(), cone.end_angle()) -
             cone.start_angle()) /
                (cone.end_angle() - cone.start_angle()),
            height_fraction};
}

[[nodiscard]] std::pair<double, double> normalized_uv(const Sphere& sphere,
                                                      const Point2D& uv) {
    const double latitude = uv.y() / sphere.radius();
    const double longitude =
        std::abs(std::cos(latitude)) > LENGTH_TOL
            ? uv.x() / (sphere.radius() * std::cos(latitude))
            : sphere.start_angle();
    const double min_latitude =
        std::asin(sphere.base_truncation() / sphere.radius());
    const double max_latitude =
        std::asin(sphere.apex_truncation() / sphere.radius());
    return {(clamp_periodic_angle(longitude, sphere.start_angle(),
                                  sphere.end_angle()) -
             sphere.start_angle()) /
                (sphere.end_angle() - sphere.start_angle()),
            (latitude - min_latitude) / (max_latitude - min_latitude)};
}

[[nodiscard]] std::pair<double, double> normalized_uv(
    const Paraboloid& paraboloid, const Point2D& uv) {
    const double total_height = (paraboloid.p2() - paraboloid.p1()).norm();
    const double height_fraction = normalize_linear(uv.y(), total_height);
    const double radius = paraboloid.radius() * std::sqrt(height_fraction);
    const double theta =
        radius > LENGTH_TOL ? uv.x() / radius : paraboloid.start_angle();
    return {(clamp_periodic_angle(theta, paraboloid.start_angle(),
                                  paraboloid.end_angle()) -
             paraboloid.start_angle()) /
                (paraboloid.end_angle() - paraboloid.start_angle()),
            height_fraction};
}

[[nodiscard]] std::pair<double, double> normalized_uv(
    [[maybe_unused]] const Cube& cube, [[maybe_unused]] const Point2D& uv) {
    throw std::logic_error(
        "Cube face classification requires a face topology definition");
}

[[nodiscard]] std::pair<double, double> normalized_uv(
    [[maybe_unused]] const TriangularPrism& prism,
    [[maybe_unused]] const Point2D& uv) {
    throw std::logic_error("TriangularPrism is cutter-only: it has no faces");
}

}  // namespace

void dedup_vertices(TriMeshD& mesh, double tolerance) {
    const double effective_tolerance = std::max(tolerance, LENGTH_TOL);
    std::unordered_map<VertexKey, Eigen::Index, VertexKeyHash> vertex_map;
    vertex_map.reserve(static_cast<std::size_t>(mesh.vertices.rows()));

    Eigen::MatrixX3d deduped_vertices(mesh.vertices.rows(), 3);
    std::vector<Eigen::Index> remap(
        static_cast<std::size_t>(mesh.vertices.rows()), 0);
    Eigen::Index next_vertex = 0;

    for (Eigen::Index vertex_idx = 0; vertex_idx < mesh.vertices.rows();
         ++vertex_idx) {
        const VertexKey key =
            make_vertex_key(mesh.vertices.row(vertex_idx), effective_tolerance);
        const auto [iterator, inserted] = vertex_map.emplace(key, next_vertex);
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
                static_cast<pycanha::MeshIndex>(remap[static_cast<std::size_t>(
                    mesh.triangles(tri_idx, corner))]);
        }
    }
    mesh.vertices = std::move(deduped_vertices);
}

void remove_degenerate_triangles(TriMeshD& mesh, double area_tolerance) {
    const double effective_tolerance =
        std::max(area_tolerance, LENGTH_TOL * LENGTH_TOL);
    std::vector<std::array<pycanha::MeshIndex, 3>> kept_triangles;
    std::vector<pycanha::MeshIndex> kept_face_ids;
    kept_triangles.reserve(static_cast<std::size_t>(mesh.triangles.rows()));
    kept_face_ids.reserve(static_cast<std::size_t>(mesh.face_ids.rows()));

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const pycanha::MeshIndex c0 = mesh.triangles(tri_idx, 0);
        const pycanha::MeshIndex c1 = mesh.triangles(tri_idx, 1);
        const pycanha::MeshIndex c2 = mesh.triangles(tri_idx, 2);
        if ((c0 == c1) || (c1 == c2) || (c2 == c0)) {
            continue;
        }

        const Vector3D p0 = mesh.vertices.row(static_cast<Eigen::Index>(c0));
        const Vector3D p1 = mesh.vertices.row(static_cast<Eigen::Index>(c1));
        const Vector3D p2 = mesh.vertices.row(static_cast<Eigen::Index>(c2));
        const double area = 0.5 * ((p1 - p0).cross(p2 - p0)).norm();
        if (area <= effective_tolerance) {
            continue;
        }

        kept_triangles.push_back({c0, c1, c2});
        kept_face_ids.push_back(mesh.face_ids(tri_idx));
    }

    mesh.triangles.resize(static_cast<Eigen::Index>(kept_triangles.size()), 3);
    mesh.face_ids.resize(static_cast<Eigen::Index>(kept_face_ids.size()));
    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const auto& triangle =
            kept_triangles[static_cast<std::size_t>(tri_idx)];
        mesh.triangles(tri_idx, 0) = triangle[0];
        mesh.triangles(tri_idx, 1) = triangle[1];
        mesh.triangles(tri_idx, 2) = triangle[2];
        mesh.face_ids(tri_idx) =
            kept_face_ids[static_cast<std::size_t>(tri_idx)];
    }

    compact_referenced_vertices(mesh);
}

pycanha::MeshIndex classify_triangle_by_centroid(
    const TriMeshD& mesh, Eigen::Index triangle_index,
    const Primitive& primitive, const ThermalMesh& thermal_mesh) {
    const Point3D p0 =
        mesh.vertices
            .row(static_cast<Eigen::Index>(mesh.triangles(triangle_index, 0)))
            .transpose();
    const Point3D p1 =
        mesh.vertices
            .row(static_cast<Eigen::Index>(mesh.triangles(triangle_index, 1)))
            .transpose();
    const Point3D p2 =
        mesh.vertices
            .row(static_cast<Eigen::Index>(mesh.triangles(triangle_index, 2)))
            .transpose();
    const Point3D centroid = (p0 + p1 + p2) / 3.0;

    const Point2D uv = std::visit(
        [&centroid](const auto& concrete_primitive) {
            return concrete_primitive.to_uv(centroid);
        },
        primitive);
    Vector3D triangle_normal = (p1 - p0).cross(p2 - p0);
    if (triangle_normal.norm() <= LENGTH_TOL) {
        triangle_normal = std::visit(
            [&uv](const auto& concrete_primitive) {
                return concrete_primitive.normal_at_uv(uv);
            },
            primitive);
    } else {
        triangle_normal.normalize();
    }

    const Vector3D primitive_normal = std::visit(
        [&uv](const auto& concrete_primitive) {
            return concrete_primitive.normal_at_uv(uv);
        },
        primitive);
    // Choosing the subdivision is this function's job; choosing the side is
    // not. A triangle always defines both faces of its pair, so it carries the
    // even id and its winding is side 1 by definition. A cut result whose
    // winding disagrees with the primitive is an orientation bug in the mesher
    // or the boolean, not a side-2 triangle: returning an odd id here would
    // make the item's face count odd and shift the parity of every later item
    // in the model.
    if (triangle_normal.dot(primitive_normal) < 0.0) {
        SPDLOG_LOGGER_WARN(
            pycanha::get_logger(),
            "cut result triangle {} is wound against the primitive normal; "
            "assigning it to side 1 anyway",
            triangle_index);
        PYCANHA_ASSERT(false,
                       "cut result triangle wound against the primitive");
    }

    const auto [dir1, dir2] = std::visit(
        [&uv](const auto& concrete_primitive) {
            return normalized_uv(concrete_primitive, uv);
        },
        primitive);

    const std::size_t face_pair_i =
        find_face_pair(thermal_mesh.get_dir1_mesh(), dir1);
    const std::size_t face_pair_j =
        find_face_pair(thermal_mesh.get_dir2_mesh(), dir2);
    // Direction 1 varies fastest, as everywhere else face pairs are numbered.
    const std::size_t linear_index =
        (face_pair_j * (thermal_mesh.get_dir1_mesh().size() - 1U)) +
        face_pair_i;
    return static_cast<pycanha::MeshIndex>(2U * linear_index);
}

}  // namespace pycanha::gmm::cutting
