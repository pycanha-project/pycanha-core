#include <vector>

#include "uv_mesher_internal.hpp"

namespace pycanha::gmm::mesh::detail {

TriMesh mesh_primitive(const Disc& disc, const ThermalMesh& thermal_mesh,
                       const MeshOptions& options) {
    const auto dir1_cuts = thermal_mesh.dir1_cuts();
    std::vector<int> dir1_segments(dir1_cuts.size() - 1U, 1);
    const double angle_span = disc.end_angle() - disc.start_angle();
    for (std::size_t index = 0; index + 1U < dir1_cuts.size(); ++index) {
        dir1_segments[index] = solve_arc_segments(
            disc.outer_radius(),
            (dir1_cuts[index + 1U] - dir1_cuts[index]) * angle_span,
            options.deviation_tolerance);
    }

    SamplingPlan plan{
        std::move(dir1_segments),
        std::vector<int>(thermal_mesh.dir2_cuts().size() - 1U, 1),
        make_linear_dir_sampler(thermal_mesh.dir1_cuts()),
        make_linear_dir_sampler(thermal_mesh.dir2_cuts()),
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
