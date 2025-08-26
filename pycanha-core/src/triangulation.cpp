

#include "pycanha-core/gmm/triangulation.hpp"

#include "pycanha-core/gmm/trimesh.hpp"

// #include "CDT/include/CDT.h"
// cppcheck-suppress *
#include <CDT/include/Triangulation.h>

#include <array>
#include <cstdint>
// #include <iostream>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Eigen/src/Core/Matrix.h"  // Eigen::RowVectorXd (to supress clang-tidy warning)
#include "Eigen/src/Core/util/Constants.h"  // Eigen::Dynamics (to supress clang-tidy warning)
#include "pycanha-core/parameters.hpp"  // pycanha::MeshIndex

namespace pycanha::gmm::trimesher {

int cdt_trimesher_cutted_mesh(TriMesh& /*trimesh*/) {
    // It seems possible to start the triangulation from an initial grid.
    // A regular one created directly by CDT or a user provided one.
    // Then we could add the points and holes?
    // https://github.com/artem-ogre/CDT/issues/19
    return -1;
}

void cdt_trimesher(TriMesh& trimesh) {
    // Create the CDT object
    // CDT::Triangulation<double> cdt(CDT::VertexInsertionOrder::Auto,
    //                               CDT::IntersectingConstraintEdges::Resolve,
    //                               LENGTH_TOL);

    // Vertex insertion order is used only for inserting the points in the
    // internal K-D tree The indexes are maintained, so it should be safe to
    // retrieve the triangles from ctd and use the original points. Source:
    // https://artem-ogre.github.io/CDT/structCDT_1_1VertexInsertionOrder.html
    CDT::Triangulation<double> cdt(CDT::VertexInsertionOrder::Auto);

    // Points
    // The points are 2D here, but they will be transformed later on.
    // So the 2D coordinates are stored in the x and y componentes of the 3D
    // points.

    const VerticesList& points = trimesh.get_vertices();
    // Add the points
    cdt.insertVertices(
        points.rowwise().begin(), points.rowwise().end(),
        [](const Eigen::RowVectorXd& row) { return row[0]; },
        [](const Eigen::RowVectorXd& row) { return row[1]; });

    // Boundary edges

    // Create a temporary vector to store the boundary edges
    MeshIndex num_bound_edges_pairs = 0;

    // Count the perimeter edges
    num_bound_edges_pairs = std::accumulate(
        trimesh.get_perimeter_edges().begin(),  // start iterator
        trimesh.get_perimeter_edges().end(),    // end iterator
        num_bound_edges_pairs,  // initial value for the accumulation
        [&trimesh](const auto& sum, const auto& edge_idx) {
            return sum + trimesh.get_edges()[edge_idx].size() - 1;
        });
    // Same as:
    // for (const auto& edge_idx : trimesh.get_perimeter_edges()) {
    //    num_bound_edges_pairs += trimesh.get_edges()[edge_idx].size() - 1;
    //}

    // Create and fill the boundary edges vector
    std::vector<std::array<MeshIndex, 2>> boundary_edges(num_bound_edges_pairs);
    pycanha::MeshIndex idx = 0;
    for (const auto& edge_idx : trimesh.get_perimeter_edges()) {
        for (pycanha::MeshIndex i = 0;
             i < trimesh.get_edges()[edge_idx].size() - 1; ++i) {
            boundary_edges[idx][0] = trimesh.get_edges()[edge_idx][i];
            boundary_edges[idx][1] = trimesh.get_edges()[edge_idx][i + 1];
            ++idx;
        }
    }

    //  Boundary edges delimite the whole surface, including holes.
    cdt.insertEdges(
        boundary_edges.begin(), boundary_edges.end(),
        [](const std::array<MeshIndex, 2>& edge) { return edge[0]; },
        [](const std::array<MeshIndex, 2>& edge) { return edge[1]; });

    // Face edges are internal edges, they do not delimit the surface.
    // They are used so each face has its own set of triangles.
    // So, each triangle only belongs to one face.
    // This type of edges do not create holes in the meshed surface.
    // As for CDT design, they need to be inserted twice, but should not
    // be duplicated or coincide with the boundary edges. So we need to
    // extract the interior as those edges that are not part of the perimeter
    // (bound).

    // Create an unordered_set from the perimete
    std::unordered_set<uint32_t> set_bound_edges(
        trimesh.get_perimeter_edges().begin(),
        trimesh.get_perimeter_edges().end());

    // Iterate through the edges, adding elements that are not found
    // in the set_bound_edges
    std::unordered_set<uint32_t> set_interior_edges;

    for (int edge_idx = 0; edge_idx < trimesh.get_edges().size(); ++edge_idx) {
        if (set_bound_edges.find(edge_idx) == set_bound_edges.end()) {
            set_interior_edges.insert(edge_idx);
        }
    }

    // Create a vector from the unordered_set

    // Count the interior edges
    pycanha::MeshIndex num_interior_edges_pairs = 0;
    num_interior_edges_pairs = std::accumulate(
        set_interior_edges.begin(),  // start iterator
        set_interior_edges.end(),    // end iterator
        num_interior_edges_pairs,    // initial value for the accumulation
        [&trimesh](const auto& sum, const auto& edge_idx) {
            return sum + trimesh.get_edges()[edge_idx].size() - 1;
        });

    // Same as:
    // for (const auto& edge_idx : set_interior_edges) {
    //    num_interior_edges_pairs += trimesh.get_edges()[edge_idx].size() - 1;
    //}

    // Create and fill the interior edges vector
    std::vector<std::array<MeshIndex, 2>> interior_edges(
        num_interior_edges_pairs);
    idx = 0;
    for (const auto& edge_idx : set_interior_edges) {
        for (pycanha::MeshIndex i = 0;
             i < trimesh.get_edges()[edge_idx].size() - 1; ++i) {
            interior_edges[idx][0] = trimesh.get_edges()[edge_idx][i];
            interior_edges[idx][1] = trimesh.get_edges()[edge_idx][i + 1];
            ++idx;
        }
    }

    // As for CDT design, they need to be inserted twice.
    cdt.insertEdges(
        interior_edges.begin(), interior_edges.end(),
        [](const std::array<MeshIndex, 2>& edge) { return edge[0]; },
        [](const std::array<MeshIndex, 2>& edge) { return edge[1]; });
    cdt.insertEdges(
        interior_edges.begin(), interior_edges.end(),
        [](const std::array<MeshIndex, 2>& edge) { return edge[0]; },
        [](const std::array<MeshIndex, 2>& edge) { return edge[1]; });

    // Triangulate the surface
    cdt.eraseOuterTrianglesAndHoles();

    TrianglesList triangles(cdt.triangles.size(), 3);

    for (pycanha::MeshIndex i = 0;
         i < static_cast<pycanha::MeshIndex>(cdt.triangles.size()); i++) {
        auto& cdt_triangles = cdt.triangles[i].vertices;
        triangles.row(i) << cdt_triangles[0], cdt_triangles[1],
            cdt_triangles[2];
    }

    trimesh.set_triangles(std::move(triangles));

    trimesh.set_face_ids(Eigen::Matrix<MeshIndex, Eigen::Dynamic, 1>::Zero(
        static_cast<pycanha::Index>(cdt.triangles.size())));
}

}  // namespace pycanha::gmm::trimesher
