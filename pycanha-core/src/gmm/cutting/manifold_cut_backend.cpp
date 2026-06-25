#include "pycanha-core/gmm/cutting/manifold_cut_backend.hpp"

#include <manifold/common.h>
#include <manifold/manifold.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/cutting/cut_mesh_ops.hpp"
#include "pycanha-core/gmm/cutting/cutter_proxy.hpp"
#include "pycanha-core/gmm/cutting/proxy_shell.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/mesh/uv_mesher.hpp"
#include "pycanha-core/gmm/ops/transform.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"

namespace pycanha::gmm::cutting {
namespace {

[[nodiscard]] std::vector<std::uint32_t> triangle_original_ids(
    const manifold::MeshGL64& mesh) {
    std::vector<std::uint32_t> original_ids(mesh.NumTri(), 0U);
    if (mesh.runOriginalID.empty()) {
        return original_ids;
    }

    std::vector<std::uint64_t> run_index(mesh.runIndex.begin(),
                                         mesh.runIndex.end());
    if (run_index.size() == mesh.runOriginalID.size()) {
        run_index.push_back(mesh.triVerts.size());
    }

    for (std::size_t run = 0; run < mesh.runOriginalID.size(); ++run) {
        const auto tri_begin = static_cast<std::size_t>(run_index[run] / 3U);
        const auto tri_end = static_cast<std::size_t>(run_index[run + 1U] / 3U);
        std::fill(original_ids.begin() + static_cast<std::ptrdiff_t>(tri_begin),
                  original_ids.begin() + static_cast<std::ptrdiff_t>(tri_end),
                  mesh.runOriginalID[run]);
    }

    return original_ids;
}

[[nodiscard]] TriMeshD extract_by_original_id(const manifold::MeshGL64& mesh,
                                              std::uint32_t original_id) {
    const auto original_ids = triangle_original_ids(mesh);

    std::vector<std::uint64_t> vertex_remap(
        mesh.NumVert(), std::numeric_limits<std::uint64_t>::max());
    std::vector<std::uint64_t> used_vertices;
    std::vector<std::array<pycanha::MeshIndex, 3>> triangles;

    for (std::size_t tri = 0; tri < original_ids.size(); ++tri) {
        if (original_ids[tri] != original_id) {
            continue;
        }

        std::array<pycanha::MeshIndex, 3> triangle{};
        for (int corner = 0; corner < 3; ++corner) {
            const auto source_vertex =
                mesh.triVerts[(tri * 3U) + static_cast<std::size_t>(corner)];
            auto& target_vertex = vertex_remap[source_vertex];
            if (target_vertex == std::numeric_limits<std::uint64_t>::max()) {
                target_vertex = used_vertices.size();
                used_vertices.push_back(source_vertex);
            }
            triangle[static_cast<std::size_t>(corner)] =
                static_cast<pycanha::MeshIndex>(target_vertex);
        }
        triangles.push_back(triangle);
    }

    TriMeshD tri_mesh;
    tri_mesh.vertices.resize(static_cast<Eigen::Index>(used_vertices.size()),
                             3);
    tri_mesh.triangles.resize(static_cast<Eigen::Index>(triangles.size()), 3);
    tri_mesh.face_ids.resize(static_cast<Eigen::Index>(triangles.size()));

    for (Eigen::Index vertex_idx = 0;
         std::cmp_less(vertex_idx, used_vertices.size()); ++vertex_idx) {
        const auto source_vertex =
            used_vertices[static_cast<std::size_t>(vertex_idx)];
        tri_mesh.vertices.row(vertex_idx) = Eigen::RowVector3d(
            mesh.vertProperties[(source_vertex * mesh.numProp) + 0U],
            mesh.vertProperties[(source_vertex * mesh.numProp) + 1U],
            mesh.vertProperties[(source_vertex * mesh.numProp) + 2U]);
    }

    for (Eigen::Index tri_idx = 0; std::cmp_less(tri_idx, triangles.size());
         ++tri_idx) {
        const auto& triangle = triangles[static_cast<std::size_t>(tri_idx)];
        tri_mesh.triangles(tri_idx, 0) = triangle[0];
        tri_mesh.triangles(tri_idx, 1) = triangle[1];
        tri_mesh.triangles(tri_idx, 2) = triangle[2];
        tri_mesh.face_ids(tri_idx) = 0U;
    }

    return tri_mesh;
}

[[nodiscard]] double proxy_thickness(const TriMeshD& mesh,
                                     const MeshOptions& options) {
    const auto bbox = mesh::ops::bounding_box(mesh);
    const double scale = bbox.diagonal().norm();
    return std::max(
        {options.deviation_tolerance * 0.25, scale * 1.0e-6, 1.0e-6});
}

}  // namespace

TriMeshD ManifoldCutBackend::cut(
    const GeometryItem& target, std::span<const Primitive> cutters,
    const CoordinateTransformation& world_transform,
    const MeshOptions& options) const {
    const Primitive world_target =
        ops::transform(target.primitive(), world_transform);
    const UvMesher mesher;
    TriMeshD target_mesh =
        mesher.mesh(world_target, target.thermal_mesh(), options);
    if (cutters.empty() || (target_mesh.triangles.rows() == 0)) {
        return target_mesh;
    }

    const std::uint32_t outer_original_id = manifold::Manifold::ReserveIDs(3U);
    const manifold::Manifold proxy = build_primitive_proxy(
        target_mesh,
        ProxyMeta{.source_item_id = target.id(),
                  .thickness = proxy_thickness(target_mesh, options),
                  .outer_original_id = outer_original_id});

    std::vector<manifold::Manifold> cutter_manifolds;
    cutter_manifolds.reserve(cutters.size());
    std::ranges::transform(
        cutters, std::back_inserter(cutter_manifolds),
        [](const Primitive& cutter) { return build_cutter(cutter); });

    const manifold::Manifold cut_union =
        cutter_manifolds.size() == 1U
            ? cutter_manifolds.front()
            : manifold::Manifold::BatchBoolean(cutter_manifolds,
                                               manifold::OpType::Add);
    const manifold::Manifold result =
        proxy.Boolean(cut_union, manifold::OpType::Subtract);
    if (result.Status() != manifold::Manifold::Error::NoError) {
        throw std::runtime_error("Manifold Boolean cut failed");
    }

    TriMeshD cut_mesh =
        extract_by_original_id(result.GetMeshGL64(), outer_original_id);
    dedup_vertices(cut_mesh, proxy_thickness(target_mesh, options) * 0.5);
    remove_degenerate_triangles(cut_mesh,
                                proxy_thickness(target_mesh, options) *
                                    proxy_thickness(target_mesh, options));

    for (Eigen::Index tri_idx = 0; tri_idx < cut_mesh.triangles.rows();
         ++tri_idx) {
        cut_mesh.face_ids(tri_idx) = classify_triangle_by_centroid(
            cut_mesh, tri_idx, world_target, target.thermal_mesh());
    }

    return cut_mesh;
}

}  // namespace pycanha::gmm::cutting
