#include <cmath>
#include <vector>

#include "uv_mesher_internal.hpp"

namespace pycanha::gmm::mesh::detail {
namespace {

[[nodiscard]] double sphere_min_latitude(const Sphere& sphere) {
    return std::asin(sphere.base_truncation() / sphere.radius());
}

[[nodiscard]] double sphere_max_latitude(const Sphere& sphere) {
    return std::asin(sphere.apex_truncation() / sphere.radius());
}

[[nodiscard]] double sphere_max_parallel_radius(const Sphere& sphere) {
    const double min_latitude = sphere_min_latitude(sphere);
    const double max_latitude = sphere_max_latitude(sphere);
    if (min_latitude <= 0.0 && max_latitude >= 0.0) {
        return sphere.radius();
    }

    const double closest_to_equator =
        std::abs(min_latitude) < std::abs(max_latitude) ? min_latitude
                                                        : max_latitude;
    return sphere.radius() * std::cos(closest_to_equator);
}

}  // namespace

TriMesh mesh_primitive(const Sphere& sphere, const ThermalMesh& thermal_mesh,
                       const MeshOptions& options) {
    const auto dir1_cuts = thermal_mesh.dir1_cuts();
    const auto dir2_cuts = thermal_mesh.dir2_cuts();
    const double min_latitude = sphere_min_latitude(sphere);
    const double max_latitude = sphere_max_latitude(sphere);
    const double latitude_span = max_latitude - min_latitude;
    const double effective_tolerance = options.deviation_tolerance > 0.0
                                           ? options.deviation_tolerance * 0.5
                                           : options.deviation_tolerance;

    std::vector<int> dir1_segments(dir1_cuts.size() - 1U, 1);
    for (std::size_t index = 0; index + 1U < dir1_cuts.size(); ++index) {
        dir1_segments[index] =
            solve_arc_segments(sphere_max_parallel_radius(sphere),
                               (dir1_cuts[index + 1U] - dir1_cuts[index]) *
                                   (sphere.end_angle() - sphere.start_angle()),
                               effective_tolerance);
    }

    std::vector<int> dir2_segments(dir2_cuts.size() - 1U, 1);
    for (std::size_t index = 0; index + 1U < dir2_cuts.size(); ++index) {
        dir2_segments[index] = solve_arc_segments(
            sphere.radius(),
            (dir2_cuts[index + 1U] - dir2_cuts[index]) * latitude_span,
            effective_tolerance);
    }

    SamplingPlan plan{
        std::move(dir1_segments), std::move(dir2_segments),
        make_linear_dir_sampler(thermal_mesh.dir1_cuts()),
        make_linear_dir_sampler(thermal_mesh.dir2_cuts()),
        [&sphere, min_latitude, latitude_span](double dir1, double dir2) {
            const double longitude =
                sphere.start_angle() +
                dir1 * (sphere.end_angle() - sphere.start_angle());
            const double latitude = min_latitude + dir2 * latitude_span;
            return sphere.to_cartesian({
                sphere.radius() * longitude * std::cos(latitude),
                sphere.radius() * latitude,
            });
        }};
    return build_mesh_from_plan(thermal_mesh, plan);
}

}  // namespace pycanha::gmm::mesh::detail
