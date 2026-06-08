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
    const std::size_t num_dir1_cells = thermal_mesh.get_dir1_mesh().size() - 1U;
    const std::size_t num_dir2_cells = thermal_mesh.get_dir2_mesh().size() - 1U;
    return {std::vector<int>(num_dir1_cells, 1),
            std::vector<int>(num_dir2_cells, 1),
            make_linear_dir_sampler(thermal_mesh.get_dir1_mesh()),
            make_linear_dir_sampler(thermal_mesh.get_dir2_mesh()),
            std::move(point_at)};
}

}  // namespace

TriMesh mesh_primitive(const Triangle& triangle,
                       const ThermalMesh& thermal_mesh,
                       const MeshOptions& /*options*/) {
    const auto plan = make_planar_sampling_plan(
        thermal_mesh, [&triangle](double dir1, double dir2) {
            return triangle_strip_point(triangle, dir1, dir2);
        });
    return build_mesh_from_plan(thermal_mesh, plan);
}

TriMesh mesh_primitive(const Rectangle& rectangle,
                       const ThermalMesh& thermal_mesh,
                       const MeshOptions& /*options*/) {
    const double u_extent = (rectangle.p2() - rectangle.p1()).norm();
    const double v_extent = rectangle.to_uv(rectangle.p3()).y();
    const auto plan = make_planar_sampling_plan(
        thermal_mesh,
        [&rectangle, u_extent, v_extent](double dir1, double dir2) {
            return rectangle.to_cartesian({dir1 * u_extent, dir2 * v_extent});
        });
    return build_mesh_from_plan(thermal_mesh, plan);
}

TriMesh mesh_primitive(const Quadrilateral& quadrilateral,
                       const ThermalMesh& thermal_mesh,
                       const MeshOptions& /*options*/) {
    const double u_extent = (quadrilateral.p2() - quadrilateral.p1()).norm();
    const double v_extent = quadrilateral.to_uv(quadrilateral.p4()).y();
    const auto plan = make_planar_sampling_plan(
        thermal_mesh,
        [&quadrilateral, u_extent, v_extent](double dir1, double dir2) {
            return quadrilateral.to_cartesian(
                {dir1 * u_extent, dir2 * v_extent});
        });
    return build_mesh_from_plan(thermal_mesh, plan);
}

}  // namespace pycanha::gmm::mesh::detail
