#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"

namespace pycanha::gmm {

// Triangular surface mesh templated on the vertex scalar type.
//   - TriMeshD (double) is the precise pipeline type used everywhere
//     internally (per-item caches, per-cut-group caches, the model root mesh
//     before conversion).
//   - TriMeshF (float) is what the model root exposes via
//   GeometryModel::mesh().
//
// face_ids index the per-side face identity: even = side 1, odd = side 2.
// Only even ids are ever stored: a triangle always defines BOTH faces of its
// pair (the odd face is the same triangle with the normal reversed), so the
// triangle carries the even id and its winding normal is side 1.
// node_numbers is DENSE, indexed directly by face_id (gaps allowed after
// cuts): node_numbers[face_id] is the side-1 node, node_numbers[face_id + 1]
// the side-2 node. primitives records which contiguous face_id range came from
// which source GeometryId, sorted by first_face_id.
template <class Scalar>
class TriMesh {
  public:
    using VertexMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, 3>;
    using TriangleMatrix = Eigen::Matrix<pycanha::MeshIndex, Eigen::Dynamic, 3>;
    using FaceIdVector = Eigen::Matrix<pycanha::MeshIndex, Eigen::Dynamic, 1>;
    using NodeNumberVector = Eigen::Matrix<pycanha::NodeNum, Eigen::Dynamic, 1>;

    struct PrimitiveRange {
        GeometryId geometry_id{};
        pycanha::MeshIndex first_face_id = 0;  // inclusive
        pycanha::MeshIndex last_face_id = 0;   // inclusive
    };

    VertexMatrix vertices;                   // Np x 3
    TriangleMatrix triangles;                // Nt x 3
    FaceIdVector face_ids;                   // Nt
    NodeNumberVector node_numbers;           // Nf (dense, indexed by face_id)
    std::vector<PrimitiveRange> primitives;  // sorted by first_face_id

    // How many faces this mesh OWNS, which is not the same as how many still
    // have triangles: a cut removes triangles but never removes the faces they
    // belonged to. Set by whoever builds the mesh (the mesher from the
    // ThermalMesh, the cut backend from its target, concatenation from the sum
    // of its pieces) and left alone by anything that only drops triangles.
    // Deriving it from max(face_ids) instead would shrink an item whose LAST
    // pair was cut away, renumbering every later item in the model.
    pycanha::MeshIndex num_faces = 0;

    [[nodiscard]] pycanha::MeshIndex np() const noexcept {
        return static_cast<pycanha::MeshIndex>(vertices.rows());
    }
    [[nodiscard]] pycanha::MeshIndex nt() const noexcept {
        return static_cast<pycanha::MeshIndex>(triangles.rows());
    }
    // Number of faces this mesh owns (both sides counted), including faces a
    // cut left with no triangles of their own.
    [[nodiscard]] pycanha::MeshIndex nf() const noexcept { return num_faces; }

    // Converts this mesh to a different vertex scalar type. Integer data and
    // primitive provenance are copied verbatim; only the vertices are cast.
    template <class Other>
    [[nodiscard]] TriMesh<Other> cast() const {
        TriMesh<Other> out;
        out.vertices = vertices.template cast<Other>();
        out.triangles = triangles;
        out.face_ids = face_ids;
        out.node_numbers = node_numbers;
        out.num_faces = num_faces;
        out.primitives.reserve(primitives.size());
        for (const auto& range : primitives) {
            out.primitives.push_back(typename TriMesh<Other>::PrimitiveRange{
                range.geometry_id, range.first_face_id, range.last_face_id});
        }
        return out;
    }
};

using TriMeshD = TriMesh<double>;
using TriMeshF = TriMesh<float>;

}  // namespace pycanha::gmm
