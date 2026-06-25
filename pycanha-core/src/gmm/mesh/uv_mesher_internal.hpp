#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"

namespace pycanha::gmm::mesh::detail {

using DirSampler = std::function<double(std::size_t, int, int)>;
using SurfacePointFunction = std::function<Point3D(double, double)>;

struct SamplingPlan {
    std::vector<int> dir1_segments;
    std::vector<int> dir2_segments;
    DirSampler dir1_sample;
    DirSampler dir2_sample;
    SurfacePointFunction point_at;
};

[[nodiscard]] double lerp(double start, double end, double t) noexcept;
[[nodiscard]] bool full_revolution(double start_angle,
                                   double end_angle) noexcept;
[[nodiscard]] int solve_arc_segments(double radius, double angle_span,
                                     double deviation_tolerance);
[[nodiscard]] int solve_paraboloid_row_segments(double max_radius,
                                                double height, double row_start,
                                                double row_end,
                                                double deviation_tolerance);
[[nodiscard]] DirSampler make_linear_dir_sampler(std::span<const double> cuts);
[[nodiscard]] TriMeshD build_mesh_from_plan(const ThermalMesh& thermal_mesh,
                                            const SamplingPlan& plan);
[[nodiscard]] Point3D triangle_strip_point(const Triangle& triangle,
                                           double dir1, double dir2);

[[nodiscard]] TriMeshD mesh_primitive(const Triangle& triangle,
                                      const ThermalMesh& thermal_mesh,
                                      const MeshOptions& options);
[[nodiscard]] TriMeshD mesh_primitive(const Rectangle& rectangle,
                                      const ThermalMesh& thermal_mesh,
                                      const MeshOptions& options);
[[nodiscard]] TriMeshD mesh_primitive(const Quadrilateral& quadrilateral,
                                      const ThermalMesh& thermal_mesh,
                                      const MeshOptions& options);
[[nodiscard]] TriMeshD mesh_primitive(const Cylinder& cylinder,
                                      const ThermalMesh& thermal_mesh,
                                      const MeshOptions& options);
[[nodiscard]] TriMeshD mesh_primitive(const Cone& cone,
                                      const ThermalMesh& thermal_mesh,
                                      const MeshOptions& options);
[[nodiscard]] TriMeshD mesh_primitive(const Paraboloid& paraboloid,
                                      const ThermalMesh& thermal_mesh,
                                      const MeshOptions& options);
[[nodiscard]] TriMeshD mesh_primitive(const Disc& disc,
                                      const ThermalMesh& thermal_mesh,
                                      const MeshOptions& options);
[[nodiscard]] TriMeshD mesh_primitive(const Sphere& sphere,
                                      const ThermalMesh& thermal_mesh,
                                      const MeshOptions& options);
[[nodiscard]] TriMeshD mesh_primitive(const Cube& cube,
                                      const ThermalMesh& thermal_mesh,
                                      const MeshOptions& options);

}  // namespace pycanha::gmm::mesh::detail
