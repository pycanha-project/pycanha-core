#pragma once

#include <Eigen/Dense>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::gmm::cutting {

// Private mesh-cleanup helpers used only by the cut backend. These were
// formerly public mesh::ops (clean.*, classify.*); they are internalized here
// because boolean cutting is their only consumer. All operate on TriMeshD
// (full-precision cut pipeline).

// Merges vertices closer than `tolerance` (clamped to LENGTH_TOL) and
// reindexes triangles accordingly.
void dedup_vertices(TriMeshD& mesh, double tolerance);

// Drops triangles with a repeated corner or area <= `area_tolerance`, then
// compacts the now-unreferenced vertices.
void remove_degenerate_triangles(TriMeshD& mesh, double area_tolerance);

// Assigns the per-side face_id to triangle `triangle_index` of `mesh` by
// projecting its centroid onto `primitive` and locating the owning UV face pair
// in `thermal_mesh`. Even result = side 1, odd = side 2 (the former Side enum,
// inlined). Returns a MeshIndex face_id.
[[nodiscard]] pycanha::MeshIndex classify_triangle_by_centroid(
    const TriMeshD& mesh, Eigen::Index triangle_index,
    const Primitive& primitive, const ThermalMesh& thermal_mesh);

}  // namespace pycanha::gmm::cutting
