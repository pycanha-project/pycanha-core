#include <cstddef>
#include <utility>
#include <vector>

#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"
#include "uv_mesher_internal.hpp"

namespace pycanha::gmm::mesh::detail {
namespace {

[[nodiscard]] SamplingPlan make_planar_sampling_plan(
    const ThermalMesh& thermal_mesh, SurfacePointFunction point_at) {
    const std::size_t num_dir1_face_pairs =
        thermal_mesh.get_dir1_mesh().size() - 1U;
    const std::size_t num_dir2_face_pairs =
        thermal_mesh.get_dir2_mesh().size() - 1U;
    return {
        .dir1_segments = std::vector<int>(num_dir1_face_pairs, 1),
        .dir2_segments = std::vector<int>(num_dir2_face_pairs, 1),
        .dir1_sample = make_linear_dir_sampler(thermal_mesh.get_dir1_mesh()),
        .dir2_sample = make_linear_dir_sampler(thermal_mesh.get_dir2_mesh()),
        .point_at = std::move(point_at)};
}

}  // namespace

// The sampling plan already works in normalised [0, 1] cuts, and so does
// every planar primitive's uv, so the point function is just to_cartesian.
TriMeshD mesh_primitive(const Triangle& triangle,
                        const ThermalMesh& thermal_mesh,
                        const MeshOptions& /*options*/) {
    const auto plan = make_planar_sampling_plan(
        thermal_mesh, [&triangle](double dir1, double dir2) {
            return triangle.to_cartesian({dir1, dir2});
        });
    return build_mesh_from_plan(thermal_mesh, plan);
}

TriMeshD mesh_primitive(const Rectangle& rectangle,
                        const ThermalMesh& thermal_mesh,
                        const MeshOptions& /*options*/) {
    const auto plan = make_planar_sampling_plan(
        thermal_mesh, [&rectangle](double dir1, double dir2) {
            return rectangle.to_cartesian({dir1, dir2});
        });
    return build_mesh_from_plan(thermal_mesh, plan);
}

TriMeshD mesh_primitive(const Quadrilateral& quadrilateral,
                        const ThermalMesh& thermal_mesh,
                        const MeshOptions& /*options*/) {
    const auto plan = make_planar_sampling_plan(
        thermal_mesh, [&quadrilateral](double dir1, double dir2) {
            return quadrilateral.to_cartesian({dir1, dir2});
        });
    return build_mesh_from_plan(thermal_mesh, plan);
}

}  // namespace pycanha::gmm::mesh::detail
