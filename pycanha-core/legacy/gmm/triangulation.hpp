#pragma once

namespace pycanha::gmm {

class TriMesh;  // Forward declaration

namespace trimesher {

int cdt_trimesher_cutted_mesh(TriMesh& /*trimesh*/);

void cdt_trimesher(TriMesh& trimesh);

}  // namespace trimesher
}  // namespace pycanha::gmm
