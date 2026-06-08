#include <cstddef>
#include <utility>
#include <vector>

#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "uv_mesher_internal.hpp"

namespace pycanha::gmm::mesh::detail {

TriMeshD mesh_primitive(const Disc& disc, const ThermalMesh& thermal_mesh,
                       const MeshOptions& options) {
    const auto dir1_cuts = thermal_mesh.get_dir1_mesh();
    std::vector<int> dir1_segments(dir1_cuts.size() - 1U, 1);
    const double angle_span = disc.end_angle() - disc.start_angle();
    for (std::size_t index = 0; index + 1U < dir1_cuts.size(); ++index) {
        dir1_segments[index] = solve_arc_segments(
            disc.outer_radius(),
            (dir1_cuts[index + 1U] - dir1_cuts[index]) * angle_span,
            options.deviation_tolerance);
    }

    const SamplingPlan plan{
        std::move(dir1_segments),
        std::vector<int>(thermal_mesh.get_dir2_mesh().size() - 1U, 1),
        make_linear_dir_sampler(thermal_mesh.get_dir1_mesh()),
        make_linear_dir_sampler(thermal_mesh.get_dir2_mesh()),
        [&disc](double dir1, double dir2) {
            const double angle = disc.start_angle() +
                                 dir1 * (disc.end_angle() - disc.start_angle());
            const double radius =
                disc.inner_radius() +
                dir2 * (disc.outer_radius() - disc.inner_radius());
            return disc.to_cartesian({angle * radius, radius});
        }};
    return build_mesh_from_plan(thermal_mesh, plan);
}

}  // namespace pycanha::gmm::mesh::detail
