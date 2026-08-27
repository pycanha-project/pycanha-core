#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "uv_mesher_internal.hpp"

namespace pycanha::gmm::mesh::detail {
namespace {

[[nodiscard]] std::vector<int> solve_circumferential_segments(
    std::span<const double> dir1_cuts, double max_radius, double start_angle,
    double end_angle, double deviation_tolerance) {
    std::vector<int> segments(dir1_cuts.size() - 1U, 1);
    const double angle_span = end_angle - start_angle;
    for (std::size_t index = 0; index + 1U < dir1_cuts.size(); ++index) {
        segments[index] = solve_arc_segments(
            max_radius, (dir1_cuts[index + 1U] - dir1_cuts[index]) * angle_span,
            deviation_tolerance);
    }
    return segments;
}

}  // namespace

TriMeshD mesh_primitive(const Cylinder& cylinder,
                        const ThermalMesh& thermal_mesh,
                        const MeshOptions& options) {
    const double height = (cylinder.p2() - cylinder.p1()).norm();
    const SamplingPlan plan{
        .dir1_segments = solve_circumferential_segments(
            thermal_mesh.get_dir1_mesh(), cylinder.radius(),
            cylinder.start_angle(), cylinder.end_angle(),
            options.deviation_tolerance),
        .dir2_segments =
            std::vector<int>(thermal_mesh.get_dir2_mesh().size() - 1U, 1),
        .dir1_sample = make_linear_dir_sampler(thermal_mesh.get_dir1_mesh()),
        .dir2_sample = make_linear_dir_sampler(thermal_mesh.get_dir2_mesh()),
        .point_at = [&cylinder, height](double dir1, double dir2) {
            const double angle =
                cylinder.start_angle() +
                (dir1 * (cylinder.end_angle() - cylinder.start_angle()));
            return cylinder.to_cartesian(
                {angle * cylinder.radius(), dir2 * height});
        }};
    return build_mesh_from_plan(thermal_mesh, plan);
}

TriMeshD mesh_primitive(const Cone& cone, const ThermalMesh& thermal_mesh,
                        const MeshOptions& options) {
    const double max_radius = std::max(cone.radius1(), cone.radius2());
    const double height = (cone.p2() - cone.p1()).norm();
    const SamplingPlan plan{
        .dir1_segments = solve_circumferential_segments(
            thermal_mesh.get_dir1_mesh(), max_radius, cone.start_angle(),
            cone.end_angle(), options.deviation_tolerance),
        .dir2_segments =
            std::vector<int>(thermal_mesh.get_dir2_mesh().size() - 1U, 1),
        .dir1_sample = make_linear_dir_sampler(thermal_mesh.get_dir1_mesh()),
        .dir2_sample = make_linear_dir_sampler(thermal_mesh.get_dir2_mesh()),
        .point_at = [&cone, height](double dir1, double dir2) {
            const double angle =
                cone.start_angle() +
                (dir1 * (cone.end_angle() - cone.start_angle()));
            const double radius =
                cone.radius1() + (dir2 * (cone.radius2() - cone.radius1()));
            return cone.to_cartesian({angle * radius, dir2 * height});
        }};
    return build_mesh_from_plan(thermal_mesh, plan);
}

TriMeshD mesh_primitive(const Paraboloid& paraboloid,
                        const ThermalMesh& thermal_mesh,
                        const MeshOptions& options) {
    const auto dir2_cuts = thermal_mesh.get_dir2_mesh();
    const double height = (paraboloid.p2() - paraboloid.p1()).norm();
    std::vector<int> dir2_segments(dir2_cuts.size() - 1U, 1);
    for (std::size_t index = 0; index + 1U < dir2_cuts.size(); ++index) {
        dir2_segments[index] = solve_paraboloid_row_segments(
            paraboloid.radius(), height, dir2_cuts[index],
            dir2_cuts[index + 1U], options.deviation_tolerance);
    }

    std::vector<double> local_dir2_cuts(dir2_cuts.begin(), dir2_cuts.end());
    const SamplingPlan plan{
        .dir1_segments = solve_circumferential_segments(
            thermal_mesh.get_dir1_mesh(), paraboloid.radius(),
            paraboloid.start_angle(), paraboloid.end_angle(),
            options.deviation_tolerance),
        .dir2_segments = std::move(dir2_segments),
        .dir1_sample = make_linear_dir_sampler(thermal_mesh.get_dir1_mesh()),
        .dir2_sample =
            [local_dir2_cuts = std::move(local_dir2_cuts)](
                std::size_t face_pair_index, int step, int step_count) {
                const double start =
                    std::sqrt(local_dir2_cuts[face_pair_index]);
                const double end =
                    std::sqrt(local_dir2_cuts[face_pair_index + 1U]);
                const double t = step_count > 0
                                     ? static_cast<double>(step) /
                                           static_cast<double>(step_count)
                                     : 0.0;
                const double radius_fraction = lerp(start, end, t);
                return radius_fraction * radius_fraction;
            },
        .point_at =
            [&paraboloid, height](double dir1, double dir2) {
                const double angle = paraboloid.start_angle() +
                                     (dir1 * (paraboloid.end_angle() -
                                              paraboloid.start_angle()));
                const double radius = paraboloid.radius() * std::sqrt(dir2);
                return paraboloid.to_cartesian({angle * radius, dir2 * height});
            }};
    return build_mesh_from_plan(thermal_mesh, plan);
}

}  // namespace pycanha::gmm::mesh::detail
