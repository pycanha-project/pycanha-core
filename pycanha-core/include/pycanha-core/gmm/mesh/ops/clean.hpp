#pragma once

#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::mesh::ops {

void dedup_vertices(TriMesh& mesh, double tolerance);
void remove_degenerate_triangles(TriMesh& mesh, double area_tolerance);

}  // namespace pycanha::gmm::mesh::ops
