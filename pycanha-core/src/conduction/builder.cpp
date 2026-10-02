#include "pycanha-core/conduction/builder.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "engine_detail.hpp"
#include "pycanha-core/conduction/network_part.hpp"
#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/face_pair_geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/gmm/scene/resolve.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"
#include "pycanha-core/utils/parallel_for.hpp"
#include "pycanha-core/utils/profiling.hpp"

namespace pycanha::conduction {

namespace {

// What survives of each face pair of a cut item: the fraction of its area,
// from the cut triangulation over the item's uncut one (same mesher, same
// tessellation, so the ratio cancels the tessellation error), and the
// area-weighted centroid of the surviving triangles, in the root frame.
struct CutSurvival {
    std::vector<double> fraction;
    std::vector<Vector3D> centroid;
};

[[nodiscard]] CutSurvival cut_survival(
    const gmm::detail::ResolvedTarget& target) {
    const gmm::GeometryItem& item = target.item;
    const gmm::ThermalMesh& thermal_mesh = item.thermal_mesh();
    const std::size_t face_pairs = (thermal_mesh.get_dir1_mesh().size() - 1U) *
                                   (thermal_mesh.get_dir2_mesh().size() - 1U);

    std::vector<double> uncut_area(face_pairs, 0.0);
    const auto triangle_geometry = [](const gmm::TriMeshD& mesh,
                                      Eigen::Index triangle) {
        const auto corners = mesh.triangles.row(triangle);
        const Point3D vertex_0 = mesh.vertices.row(to_idx(corners(0)));
        const Point3D vertex_1 = mesh.vertices.row(to_idx(corners(1)));
        const Point3D vertex_2 = mesh.vertices.row(to_idx(corners(2)));
        return std::pair{
            0.5 * (vertex_1 - vertex_0).cross(vertex_2 - vertex_0).norm(),
            Point3D{(vertex_0 + vertex_1 + vertex_2) / 3.0}};
    };
    const gmm::TriMeshD& uncut = item.mesh();
    for (Eigen::Index triangle = 0; triangle < uncut.triangles.rows();
         ++triangle) {
        const auto face_pair = to_sizet(to_idx(uncut.face_ids(triangle))) / 2U;
        uncut_area[face_pair] += triangle_geometry(uncut, triangle).first;
    }

    CutSurvival survival{
        .fraction = std::vector<double>(face_pairs, 0.0),
        .centroid = std::vector<Vector3D>(face_pairs, Vector3D::Zero())};
    const gmm::TriMeshD cut = gmm::detail::mesh_target(target);
    for (Eigen::Index triangle = 0; triangle < cut.triangles.rows();
         ++triangle) {
        const auto face_pair = to_sizet(to_idx(cut.face_ids(triangle))) / 2U;
        const auto [area, centre] = triangle_geometry(cut, triangle);
        survival.fraction[face_pair] += area;
        survival.centroid[face_pair] += area * centre;
    }
    for (std::size_t face_pair = 0; face_pair < face_pairs; ++face_pair) {
        const double surviving = survival.fraction[face_pair];
        if (surviving > 0.0) {
            survival.centroid[face_pair] /= surviving;
        }
        survival.fraction[face_pair] =
            uncut_area[face_pair] > 0.0
                ? std::clamp(surviving / uncut_area[face_pair], 0.0, 1.0)
                : 0.0;
    }
    return survival;
}

[[nodiscard]] unsigned workers_for(std::size_t tasks) {
    return tasks > 1U ? std::min<unsigned>(std::thread::hardware_concurrency(),
                                           static_cast<unsigned>(tasks))
                      : 1U;
}

}  // namespace

std::string_view to_string(DiagnosticCode code) noexcept {
    switch (code) {
        case DiagnosticCode::CutFacePairs:
            return "CutFacePairs";
        case DiagnosticCode::UncoupledNodes:
            return "UncoupledNodes";
        case DiagnosticCode::UnmeshedPrimitive:
            return "UnmeshedPrimitive";
        case DiagnosticCode::InactiveSideSkipped:
            return "InactiveSideSkipped";
        case DiagnosticCode::MissingBulk:
            return "MissingBulk";
        case DiagnosticCode::ZeroThickness:
            return "ZeroThickness";
        case DiagnosticCode::ZeroConductivity:
            return "ZeroConductivity";
        case DiagnosticCode::MixedBulkOnNode:
            return "MixedBulkOnNode";
        case DiagnosticCode::DiscreteLinkFallback:
            return "DiscreteLinkFallback";
        case DiagnosticCode::NoNodeNumbers:
            return "NoNodeNumbers";
        case DiagnosticCode::DegenerateFacePair:
            return "DegenerateFacePair";
        case DiagnosticCode::AxisSingularity:
            return "AxisSingularity";
    }
    return "Unknown";
}

TmmBuildReport build_tmm_from_gmm(ThermalModel& model,
                                  const TmmBuildOptions& options) {
    ThermalMathematicalModel& tmm = model.tmm();
    detail::require_empty_tmm(tmm);

    // Every item with its placement and the cutters that apply to it; no
    // triangulation.
    const gmm::GeometryGroup& root = *model.gmm().root_group();
    const auto targets = gmm::detail::collect_targets(root, root.transform());

    // Cut items: their fraction of each face pair from their cut
    // triangulation. The uncut triangulation is an item's cache, which filling
    // mutates, so those are made here; the cuts then run in parallel.
    std::vector<CutSurvival> survival(targets.size());
    {
        PYCANHA_PROFILE_SCOPE("build_tmm_from_gmm: cuts");
        std::vector<std::size_t> cut_targets;
        for (std::size_t target = 0; target < targets.size(); ++target) {
            const gmm::GeometryItem& item = targets[target].item;
            if (!targets[target].cutters.empty() &&
                gmm::FacePairGeometryEvaluator(item.primitive(),
                                               item.thermal_mesh())
                    .is_supported()) {
                static_cast<void>(item.mesh());
                cut_targets.push_back(target);
            }
        }
        utils::parallel_for_index(
            cut_targets.size(), workers_for(cut_targets.size()),
            [&](std::size_t entry) {
                survival[cut_targets[entry]] =
                    cut_survival(targets[cut_targets[entry]]);
            });
    }

    // One network part per item, each from its own definition.
    std::vector<NetworkPart> parts(targets.size());
    {
        PYCANHA_PROFILE_SCOPE("build_tmm_from_gmm: parts");
        utils::parallel_for_index(
            targets.size(), workers_for(targets.size()),
            [&](std::size_t target) {
                parts[target] = build_network_part(
                    targets[target].item, targets[target].to_root, options,
                    survival[target].fraction, survival[target].centroid);
            });
    }

    PYCANHA_PROFILE_SCOPE("build_tmm_from_gmm: commit");
    return commit_network_parts(tmm, parts, options);
}

}  // namespace pycanha::conduction
