#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pycanha-core/gmm/id.hpp"
#include "pycanha-core/gmm/materials.hpp"
#include "pycanha-core/gmm/triangulation.hpp"
#include "pycanha-core/parameters.hpp"
#include "pycanha-core/utils/eigenutils.hpp"

using Eigen::seq;
using namespace pycanha;  // NOLINT
namespace pycanha::gmm {

/**
 * @brief Typedef for the vertices collection
 */
using VerticesList = Eigen::Matrix<double, Eigen::Dynamic, 3>;

/**
 * @brief Typedef for the vertices collection (double precision)
 */
using VerticesListDouble = Eigen::Matrix<double, Eigen::Dynamic, 3>;

/**
 * @brief Typedef for the vertices collection (single precision)
 */
using VerticesListFloat = Eigen::Matrix<float, Eigen::Dynamic, 3>;

/**
 * @brief Typedef for the triangles collection
 */
using TrianglesList = Eigen::Matrix<MeshIndex, Eigen::Dynamic, 3>;

/**
 * @brief Typedef for the face ids vector
 */
using FaceIdsList = Eigen::Matrix<MeshIndex, Eigen::Dynamic, 1>;

/**
 * @brief Typedef for edges. Each edge is a list of points (as indices) defining
 * the edge.
 */
using Edges = Eigen::Matrix<MeshIndex, Eigen::Dynamic, 1>;

/**
 * @brief Typedef for a list of edges. Each edge is a list of points (as
 * indices) defining the edge.
 */
using EdgesList = std::vector<Edges>;

/**
 * @brief Typedef for a list of ids of the edges.
 */
using EdgesIdsList = Eigen::Matrix<MeshIndex, Eigen::Dynamic, 1>;

/**
 * @brief Typedef for face edges.
 */
using FaceEdges = std::vector<EdgesIdsList>;

/**
 * @brief Class for storing triangular meshes
 *
 * Triangular meshes are a common way to represent 3D surfaces.
 * This class stores them as three main components: vertices, triangles, and
 * face ids.
 */

class TriMesh {
  public:
    /**
     * @brief Default constructor
     */
    TriMesh() = default;

    /**
     * @brief Constructor with parameters
     *
     * @param vertices Input vertices matrix
     * @param triangles Input triangles matrix
     * @param face_ids Input face ids vector
     * @param edges Input all the edges
     * @param perimeter_edges Input perimeter edges indices
     * @param faces_edges Input faces edges vector of indices
     */
    TriMesh(VerticesList vertices, TrianglesList triangles,
            FaceIdsList face_ids, EdgesList edges, EdgesIdsList perimeter_edges,
            FaceEdges faces_edges)
        : _vertices(std::move(vertices)),
          _triangles(std::move(triangles)),
          _face_ids(std::move(face_ids)),
          _edges(std::move(edges)),
          _perimeter_edges(std::move(perimeter_edges)),
          _faces_edges(std::move(faces_edges)) {}

    /**
     * @brief Get vertices matrix
     *
     * @return const VerticesList& Vertices matrix
     */
    [[nodiscard]] const VerticesList& get_vertices() const { return _vertices; }

    /**
     * @brief Get non-const vertices matrix
     * @return VerticesList& Vertices matrix
     */
    VerticesList& get_vertices() { return _vertices; }

    /**
     * @brief Set vertices matrix
     *
     * @param vertices Input vertices matrix
     */
    void set_vertices(VerticesList vertices) {
        _vertices = std::move(vertices);
    }

    /**
     * @brief Get triangles matrix
     *
     * @return const TrianglesList& Triangles matrix
     */
    [[nodiscard]] const TrianglesList& get_triangles() const {
        return _triangles;
    }

    /**
     * @brief Get non-const triangles matrix
     *
     * @return TrianglesList& Triangles matrix
     */
    TrianglesList& get_triangles() { return _triangles; }

    /**
     * @brief Set triangles matrix
     *
     * @param triangles Input triangles matrix
     */
    void set_triangles(TrianglesList triangles) {
        _triangles = std::move(triangles);
    }

    /**
     * @brief Get face ids vector
     *
     * @return const FaceIdsList& Face ids vector
     */
    [[nodiscard]] const FaceIdsList& get_face_ids() const { return _face_ids; }

    /**
     * @brief Get non-const face ids vector
     *
     * @return FaceIdsList& Face ids vector
     */
    FaceIdsList& get_face_ids() { return _face_ids; }

    /**
     * @brief Set face ids vector
     *
     * @param face_ids Input face ids vector
     */
    void set_face_ids(FaceIdsList face_ids) { _face_ids = std::move(face_ids); }

    /**
     * @brief Get edges
     *
     * @return const EdgesList& edges
     */
    [[nodiscard]] const EdgesList& get_edges() const { return _edges; }

    /**
     * @brief Get non-const edges
     *
     * @return EdgesList& edges
     */
    EdgesList& get_edges() { return _edges; }

    /**
     * @brief Set the edges
     *
     * @param edges Input edges
     */
    void set_edges(EdgesList edges) { _edges = std::move(edges); }

    /**
     * @brief Get perimeter edges indices
     *
     * @return const EdgesIdsList& Perimeter edges indices
     */
    [[nodiscard]] const EdgesIdsList& get_perimeter_edges() const {
        return _perimeter_edges;
    }

    /**
     * @brief Get non-const perimeter edges indices
     *
     * @return EdgesIdsList& Perimeter edges indices
     */
    EdgesIdsList& get_perimeter_edges() { return _perimeter_edges; }

    /**
     * @brief Set perimeter edges indices
     *
     * @param perimeter_edges Input perimeter edges indices
     */
    void set_perimeter_edges(EdgesIdsList perimeter_edges) {
        _perimeter_edges = std::move(perimeter_edges);
    }

    /**
     * @brief Get faces edges vector of indices
     *
     * @return const FaceEdges& Faces edges vector of
     * indices
     */
    [[nodiscard]] const FaceEdges& get_faces_edges() const {
        return _faces_edges;
    }

    /**
     * @brief Set faces edges vector of indices
     *
     * @param faces_edges Input faces edges vector of indices
     */
    void set_faces_edges(FaceEdges faces_edges) {
        _faces_edges = std::move(faces_edges);
    }

    /**
     * @brief Get cummulative area vector
     *
     * @return const std::vector<float>>& cummulative area vector
     */
    [[nodiscard]] const std::vector<float>& get_cumareas() const {
        return _cumareas;
    }

    /**
     * @brief Get the number of faces
     *
     * @return MeshIndex Number of faces
     */
    [[nodiscard]] MeshIndex get_number_of_faces() const { return _n_faces; }

    /**
     * @brief  Set the number of faces
     *
     * @param n_faces Number of faces
     */
    void set_number_of_faces(MeshIndex n_faces) { _n_faces = n_faces; }

    /**
     * @brief Get the surface 1 color
     *
     * @return const ColorRGB& Surface 1 color
     */
    [[nodiscard]] const ColorRGB& get_surface1_color() const {
        return _surface1_color;
    }

    /**
     * @brief Set the surface 1 color
     *
     * @param surface1_color Input surface 1 color
     */
    void set_surface1_color(const ColorRGB& surface1_color) {
        _surface1_color = surface1_color;
    }

    /**
     * @brief Get the surface 2 color
     *
     * @return const ColorRGB& Surface 2 color
     */
    [[nodiscard]] const ColorRGB& get_surface2_color() const {
        return _surface2_color;
    }

    /**
     * @brief Set the surface 2 color
     *
     * @param surface2_color Input surface 2 color
     */
    void set_surface2_color(const ColorRGB& surface2_color) {
        _surface2_color = surface2_color;
    }

    /**
     * @brief Sort the triangles according to the face ids
     */
    void sort_triangles() {
        // Create a vector of indices
        // TODO: Change indices to Eigen object (for avoid castings)
        std::vector<MeshIndex> indices(
            static_cast<MeshIndex>(_triangles.rows()));

        if (_triangles.rows() == 0) {
            return;
        }

        std::iota(indices.begin(), indices.end(), 0);

        // Sort the indices according to the face ids
        std::sort(indices.begin(), indices.end(),
                  [this](MeshIndex i1, MeshIndex i2) {
                      return _face_ids[i1] < _face_ids[i2];
                  });

        // Sort the triangles
        TrianglesList sorted_triangles(_triangles.rows(), _triangles.cols());
        for (VectorIndex i = 0; i < indices.size(); ++i) {
            sorted_triangles.row(static_cast<Index>(i)) =
                _triangles.row(indices[i]);
        }
        _triangles = std::move(sorted_triangles);

        // Reorder the face ids
        FaceIdsList sorted_face_ids(_face_ids.size());
        for (Index i = 0; i < static_cast<Index>(indices.size()); ++i) {
            sorted_face_ids[i] =
                _face_ids[indices[static_cast<VectorIndex>(i)]];
            // std::cout << sorted_face_ids[i] << "\n";
        }

        _face_ids = std::move(sorted_face_ids);
        _are_triangles_sorted = true;
    }

    void compute_areas() {
        // Cummulative areas only make sense if the triangles are sorted
        if (!_are_triangles_sorted) {
            sort_triangles();
        }

        // Size of triangles list
        auto N = static_cast<MeshIndex>(_triangles.rows());

        if (N == 0) {
            return;
        }

        // Resize _cumareas to hold N elements
        _cumareas.resize(N);

        MeshIndex current_face = _face_ids[0];  // Should be 0
        double cum_area = 0.0;
        for (MeshIndex i = 0; i < N; ++i) {
            // If we are on a new face, reset the cumulative area
            if (_face_ids[i] != current_face) {
                current_face = _face_ids[i];
                cum_area = 0.0;
            }

            // Get vertices of triangle i
            const Point3D p1 = _vertices.row(_triangles(i, 0));
            const Point3D p2 = _vertices.row(_triangles(i, 1));
            const Point3D p3 = _vertices.row(_triangles(i, 2));

            // Compute area of triangle i
            cum_area += 0.5 * ((p2 - p1).cross(p3 - p1)).norm();

            // Store cumulative area
            _cumareas[i] = static_cast<float>(cum_area);
        }
    }

  private:
    VerticesList _vertices;  //!< Vertices as an Nx3 double matrix

    TrianglesList _triangles;  //!< Triangles as an Nx3 MeshIndex matrix

    std::vector<float> _cumareas;  //!< Cumulative areas of the triangles

    FaceIdsList _face_ids;  //!< Face ids as an N MeshIndex vector

    EdgesList _edges;  //!< Edges

    EdgesIdsList _perimeter_edges;  //!< List of perimeter edges

    FaceEdges _faces_edges;  //!< For each face, the list of edges

    bool _are_triangles_sorted =
        false;  //!< Whether the triangles are sorted or not

    MeshIndex _n_faces = 0;  //!< Number of faces

    ColorRGB _surface1_color = ColorRGB({0, 0, 0});  //!< Surface 1 color
    ColorRGB _surface2_color = ColorRGB({0, 0, 0});  //!< Surface 1 color
};

/**
 * @brief A collection of TriMeshes in a compacted cache-friendly format
 *
 * This class is used to store the whole mesh of the geometrical model.
 * It contains the information to plot the model and to perform the ray-tracing
 * calculations.
 */
class TriMeshModel {
    VerticesListFloat _vertices;  //!< Vertices as an (Nv,3) float matrix
    TrianglesList _triangles;     //!< Triangles as an (Nt,3) MeshIndex matrix
    FaceIdsList _face_ids;        //!< Face index of the triangles as an
                                  //!< (Nt) MeshIndex vector
    std::vector<float>
        _face_cumarea;  //!< Cumulative area of the triangles that made each
                        //!< face as an (Nt) float vector

    std::vector<int8_t>
        _face_activity;  //!< Activity of the faces as an (Nf/2) uint8_t vector
                         //!< -1: both inactive, 0: both active
                         //!< 1: only first active, 2: only second active

    // TODO: Use a class/struct/map, to not repeat the same information
    std::vector<std::array<float, 6>> _opticals;  //!< Optical properties of the
                                                  //!< faces as an (Nf,3)
                                                  //!< float matrix

    MeshIndex _n_faces = 0;       //!< Number of faces
    MeshIndex _n_geometries = 0;  //!< Number of geometries

    std::vector<ColorRGB>
        _front_colors;  //!< RGB Color of the front surface of the geometries as
                        //!< an (Ng,3) uint8_t matrix

    std::vector<ColorRGB>
        _back_colors;  //!< RGB Color of the back surface of the geometries as
                       //!< an (Ng,3) uint8_t matrix

    std::vector<MeshIndex> _geometries_triangles = {
        0};  //!< For each geometry, start and end
             //!< indices of the faces. Size: Ng + 1

    std::vector<MeshIndex> _geometries_vertices = {
        0};  //!< For each geometry, start and end
             //!< indices of the vertices. Size: Ng + 1

    std::vector<MeshIndex> _geometries_edges = {
        0};  //!< For each geometry, start and end
             //!< indices of the edges. Size: Ng + 1

    std::vector<MeshIndex> _geometries_perimeter_edges = {
        0};  //!< For each geometry, start and end
             //!< indices of the perimeter edges. Size: Ng + 1

    std::vector<GeometryIdType>
        _geometries_id;  //!< Unique ID of each geometry. Size: Ng

    // Useful for the plot
    EdgesList _edges;  //!< Edges

    EdgesIdsList _perimeter_edges;  //!< List of perimeter edges

    FaceEdges _faces_edges;  //!< For each face, the list of edges

    // This mesh should have the following properties:

    // _face_ids:
    // The Face ID identifies a face of the thermal mesh. Because there is a
    // distinction between the front and back properties, back and front sides
    // have different IDs. The _face_ids vector stored the Face ID of the
    // triangles that made up the faces. But it is only necessary to store one
    // side id, if the other one can be deducted from it. By convention, the id
    // of the front side is stored, which is always an even number. The backface
    // id is always an odd number equal to the front side + 1 (by convention) So
    // an example of face_ids could be:
    //   [0,0, 2,2,2,2, ..., (Nf-1), (Nf-1), (Nf-1)]
    // For convenience to use the Face ID as indices in the VF or GR matrices,
    // the Face IDs stored in _face_ids should not have gaps. But in principle,
    // it is not necessary to be in order, so this should be valid:
    //   [6,6, 2,2,2,2, ..., (Nf-1),(Nf-1),(Nf-1), 4,4,4]
    //
    // _n_faces:
    // Number of faces in the model.
    // Equal to: is unique(_face_ids)*2
    // Also equal to: max(_face_ids) + 1 (no gaps)

  public:
    /**
     * @brief Get the vertices matrix
     *
     * @return const VerticesListFloat& Vertices matrix
     */
    [[nodiscard]] const VerticesListFloat& get_vertices() const {
        return _vertices;
    }

    /**
     * @brief Get the non-const vertices matrix
     *
     * @return  VerticesListFloat& Vertices matrix
     */
    [[nodiscard]] VerticesListFloat& get_vertices() { return _vertices; }

    /**
     * @brief Get the vertices matrix
     *
     * @return VerticesListFloat& Vertices matrix
     */
    void set_vertices(VerticesListFloat vertices) {
        _vertices = std::move(vertices);
    }

    /**
     * @brief Get the triangles matrix
     *
     * @return const TrianglesList& Triangles matrix
     */
    [[nodiscard]] const TrianglesList& get_triangles() const {
        return _triangles;
    }

    /**
     * @brief Get the non-const triangles matrix
     *
     * @return TrianglesList& Triangles matrix
     */
    TrianglesList& get_triangles() { return _triangles; }

    /**
     * @brief Set the triangles matrix
     *
     * @param triangles Input triangles matrix
     */
    void set_triangles(TrianglesList triangles) {
        _triangles = std::move(triangles);
    }

    /**
     * @brief Get the face ids vector
     *
     * @return const FaceIdsList& Face ids vector
     */
    [[nodiscard]] const FaceIdsList& get_face_ids() const { return _face_ids; }

    /**
     * @brief Get the non-const face ids vector
     *
     * @return FaceIdsList& Face ids vector
     */
    FaceIdsList& get_face_ids() { return _face_ids; }

    /**
     * @brief Set the face ids vector
     *
     * @param face_ids Input face ids vector
     */
    void set_face_ids(FaceIdsList face_ids) { _face_ids = std::move(face_ids); }

    /**
     * @brief Get the cumulative area vector
     *
     * @return const std::vector<float>& Cumulative area vector
     */
    [[nodiscard]] const std::vector<float>& get_cumareas() const {
        return _face_cumarea;
    }

    /**
     * @brief Get the face activity vector
     *
     * @return const std::vector<int8_t>& Face activity vector
     */
    [[nodiscard]] const std::vector<int8_t>& get_face_activity() const {
        return _face_activity;
    }

    /**
     * @brief Get the non-const face activity vector
     *
     * @return std::vector<int8_t>& Face activity vector
     */
    std::vector<int8_t>& get_face_activity() { return _face_activity; }

    /**
     * @brief Set the face activity vector
     *
     * @param face_activity Input face activity vector
     */
    void set_face_activity(std::vector<int8_t> face_activity) {
        _face_activity = std::move(face_activity);
    }

    /**
     * @brief Get the optical properties vector
     *
     * @return const std::vector<std::array<float, 6>>& Optical properties
     * vector
     */
    [[nodiscard]] const std::vector<std::array<float, 6>>& get_opticals()
        const {
        return _opticals;
    }

    /**
     * @brief Get the non-const optical properties vector
     *
     * @return std::vector<std::array<float, 6>>& Optical properties vector
     */
    std::vector<std::array<float, 6>>& get_opticals() { return _opticals; }

    /**
     * @brief Set the optical properties vector
     *
     * @param opticals Input optical properties vector
     */
    void set_opticals(std::vector<std::array<float, 6>> opticals) {
        _opticals = std::move(opticals);
    }

    /**
     * @brief Get the number of faces
     *
     * @return MeshIndex Number of faces
     */
    [[nodiscard]] MeshIndex get_number_of_faces() const { return _n_faces; }

    /**
     * @brief Set the number of faces
     *
     * @param n_faces Number of faces
     */
    void set_number_of_faces(MeshIndex n_faces) { _n_faces = n_faces; }

    /**
     * @brief Get the number of geometries
     *
     * @return MeshIndex Number of geometries
     */
    [[nodiscard]] MeshIndex get_number_of_geometries() const {
        return _n_geometries;
    }

    /**
     * @brief Set the number of geometries
     *
     * @param n_geometries Number of geometries
     */
    void set_number_of_geometries(MeshIndex n_geometries) {
        _n_geometries = n_geometries;
    }

    /**
     * @brief Get the front colors vector
     *
     * @return const std::vector<ColorRGB>& Front colors vector
     */
    [[nodiscard]] const std::vector<ColorRGB>& get_front_colors() const {
        return _front_colors;
    }

    /**
     * @brief Get the non-const front colors vector
     *
     * @return std::vector<ColorRGB>& Front colors vector
     */
    std::vector<ColorRGB>& get_front_colors() { return _front_colors; }

    /**
     * @brief Set the front colors vector
     *
     * @param front_colors Input front colors vector
     */
    void set_front_colors(std::vector<ColorRGB> front_colors) {
        _front_colors = std::move(front_colors);
    }

    /**
     * @brief Get the back colors vector
     *
     * @return const std::vector<ColorRGB>& Back colors vector
     */
    [[nodiscard]] const std::vector<ColorRGB>& get_back_colors() const {
        return _back_colors;
    }

    /**
     * @brief Get the non-const back colors vector
     *
     * @return std::vector<ColorRGB>& Back colors vector
     */
    std::vector<ColorRGB>& get_back_colors() { return _back_colors; }

    /**
     * @brief Set the back colors vector
     *
     * @param back_colors Input back colors vector
     */
    void set_back_colors(std::vector<ColorRGB> back_colors) {
        _back_colors = std::move(back_colors);
    }

    /**
     * @brief Get the geometries triangles vector
     *
     * @return const std::vector<MeshIndex>& Geometries triangles vector
     */
    [[nodiscard]] const std::vector<MeshIndex>& get_geometries_triangles()
        const {
        return _geometries_triangles;
    }

    /**
     * @brief Get the non-const geometries triangles vector
     *
     * @return std::vector<MeshIndex>& Geometries triangles vector
     */
    std::vector<MeshIndex>& get_geometries_triangles() {
        return _geometries_triangles;
    }

    /**
     * @brief Set the geometries triangles vector
     *
     * @param geometries_triangles Input geometries triangles vector
     */
    void set_geometries_triangles(std::vector<MeshIndex> geometries_triangles) {
        _geometries_triangles = std::move(geometries_triangles);
    }

    /**
     * @brief Get the geometries vertices vector
     *
     * @return const std::vector<MeshIndex>& Geometries vertices vector
     */
    [[nodiscard]] const std::vector<MeshIndex>& get_geometries_vertices()
        const {
        return _geometries_vertices;
    }

    /**
     * @brief Get the non-const geometries vertices vector
     *
     * @return std::vector<MeshIndex>& Geometries vertices vector
     */
    std::vector<MeshIndex>& get_geometries_vertices() {
        return _geometries_vertices;
    }

    /**
     * @brief Set the geometries vertices vector
     *
     * @param geometries_vertices Input geometries vertices vector
     */
    void set_geometries_vertices(std::vector<MeshIndex> geometries_vertices) {
        _geometries_vertices = std::move(geometries_vertices);
    }

    /**
     * @brief Get the geometries edges vector
     *
     * @return const std::vector<MeshIndex>& Geometries edges vector
     */
    [[nodiscard]] const std::vector<MeshIndex>& get_geometries_edges() const {
        return _geometries_edges;
    }

    /**
     * @brief Get the non-const geometries edges vector
     *
     * @return std::vector<MeshIndex>& Geometries edges vector
     */
    std::vector<MeshIndex>& get_geometries_edges() { return _geometries_edges; }

    /**
     * @brief Set the geometries edges vector
     *
     * @param geometries_edges Input geometries edges vector
     */
    void set_geometries_edges(std::vector<MeshIndex> geometries_edges) {
        _geometries_edges = std::move(geometries_edges);
    }

    /**
     * @brief Get the geometries perimeter edges vector
     *
     * @return const std::vector<MeshIndex>& Geometries perimeter edges vector
     */
    [[nodiscard]] const std::vector<MeshIndex>& get_geometries_perimeter_edges()
        const {
        return _geometries_perimeter_edges;
    }

    /**
     * @brief Get the non-const geometries perimeter edges vector
     *
     * @return std::vector<MeshIndex>& Geometries perimeter edges vector
     */
    std::vector<MeshIndex>& get_geometries_perimeter_edges() {
        return _geometries_perimeter_edges;
    }

    /**
     * @brief Set the geometries perimeter edges vector
     *
     * @param geometries_perimeter_edges Input geometries perimeter edges vector
     */
    void set_geometries_perimeter_edges(
        std::vector<MeshIndex> geometries_perimeter_edges) {
        _geometries_perimeter_edges = std::move(geometries_perimeter_edges);
    }

    /**
     * @brief Get the geometries id vector
     *
     * @return const std::vector<GeometryIdType>& Geometries id vector
     */
    [[nodiscard]] const std::vector<GeometryIdType>& get_geometries_id() const {
        return _geometries_id;
    }

    /**
     * @brief Get the non-const geometries id vector
     *
     * @return std::vector<GeometryIdType>& Geometries id vector
     */
    std::vector<GeometryIdType>& get_geometries_id() { return _geometries_id; }

    /**
     * @brief Set the geometries id vector
     *
     * @param geometries_id Input geometries id vector
     */
    void set_geometries_id(std::vector<GeometryIdType> geometries_id) {
        _geometries_id = std::move(geometries_id);
    }

    /**
     * @brief Get the edges
     *
     * @return const EdgesList& Edges
     */
    [[nodiscard]] const EdgesList& get_edges() const { return _edges; }

    /**
     * @brief Get the non-const edges
     *
     * @return EdgesList& Edges
     */
    EdgesList& get_edges() { return _edges; }

    /**
     * @brief Set the edges
     *
     * @param edges Input edges
     */
    void set_edges(EdgesList edges) { _edges = std::move(edges); }

    /**
     * @brief Get the perimeter edges
     *
     * @return const EdgesIdsList& Perimeter edges
     */
    [[nodiscard]] const EdgesIdsList& get_perimeter_edges() const {
        return _perimeter_edges;
    }

    /**
     * @brief Get the non-const perimeter edges
     *
     * @return EdgesIdsList& Perimeter edges
     */
    EdgesIdsList& get_perimeter_edges() { return _perimeter_edges; }

    /**
     * @brief Set the perimeter edges
     *
     * @param perimeter_edges Input perimeter edges
     */
    void set_perimeter_edges(EdgesIdsList perimeter_edges) {
        _perimeter_edges = std::move(perimeter_edges);
    }

    /**
     * @brief Get the faces edges
     *
     * @return const FaceEdges& Faces edges
     */
    [[nodiscard]] const FaceEdges& get_faces_edges() const {
        return _faces_edges;
    }

    /**
     * @brief Set the faces edges
     *
     * @param faces_edges Input faces edges
     */
    void set_faces_edges(FaceEdges faces_edges) {
        _faces_edges = std::move(faces_edges);
    }

    /**
     * @brief Aggregate the contents of a Trimesh to this one
     * @param trimesh The trimesh Trimesh
     */
    void add_mesh(const TriMesh& trimesh, GeometryIdType geometry_id) {
        // Get the size of the mesh
        auto new_trimesh_n_points =
            static_cast<MeshIndex>(trimesh.get_vertices().rows());
        auto new_trimesh_n_triangles =
            static_cast<MeshIndex>(trimesh.get_triangles().rows());
        auto new_trimesh_n_edges =
            static_cast<MeshIndex>(trimesh.get_edges().size());

        // Get the size of the current mesh
        auto current_n_points = static_cast<MeshIndex>(_vertices.rows());
        auto current_n_triangles = static_cast<MeshIndex>(_triangles.rows());
        auto current_n_edges = static_cast<MeshIndex>(_edges.size());

        // MeshIndex offset_points = static_cast<MeshIndex>(_vertices.rows());
        // MeshIndex offset_edges = static_cast<MeshIndex>(_edges.size());

        // 1) Aggregate vertices
        VerticesListFloat temp_vertices(current_n_points + new_trimesh_n_points,
                                        3);
        temp_vertices << _vertices, trimesh.get_vertices().cast<float>();
        _vertices = std::move(temp_vertices);

        // 2) Aggregate triangles with offset
        TrianglesList temp_triangles(
            current_n_triangles + new_trimesh_n_triangles, 3);
        temp_triangles << _triangles,
            (trimesh.get_triangles().array() + current_n_points);
        _triangles = std::move(temp_triangles);

        // 3) Aggregate Faces ids

        // TODO(PERFORMANCE): Ensure faces ids are sorted so this is not
        // necessary.
        auto last_face_id_it =
            std::max_element(_face_ids.begin(), _face_ids.end());
        MeshIndex last_face_id = 0;
        if (last_face_id_it != _face_ids.end()) {
            last_face_id = *last_face_id_it + 2;
        }
        // This check is not necessary by design.
        // TODO(PERFORMANCE): when sure it works, and use an assert instead.
        if (last_face_id % 2 != 0) {
            std::cout << "Current face_ids" << current_n_triangles << "\n";
            std::cout << "New face_ids" << new_trimesh_n_triangles << "\n";
            throw std::runtime_error(
                "Faces IDs numbering error. Contact developers.");
        }

        FaceIdsList temp_face_ids(current_n_triangles +
                                  new_trimesh_n_triangles);
        // Copy the current face ids
        std::copy(_face_ids.begin(), _face_ids.end(), temp_face_ids.begin());
        // Add the new face ids with the offset

        std::transform(
            trimesh.get_face_ids().begin(), trimesh.get_face_ids().end(),
            temp_face_ids.begin() + current_n_triangles,
            [last_face_id](MeshIndex val) { return val + last_face_id; });
        _face_ids = std::move(temp_face_ids);

        // 4) Aggregate areas
        std::vector<float> temp_areas(current_n_triangles +
                                      new_trimesh_n_triangles);

        std::copy(_face_cumarea.begin(), _face_cumarea.end(),
                  temp_areas.begin());
        std::copy(trimesh.get_cumareas().begin(), trimesh.get_cumareas().end(),
                  temp_areas.begin() + current_n_triangles);

        // 5) Face activity
        // TODO(FEATURE)

        // 6) Opticals
        // TODO(FEATURE)

        // 7) Number of faces and geometries
        _n_faces += trimesh.get_number_of_faces();
        _n_geometries += 1;

        // 8) Colors
        _front_colors.push_back(trimesh.get_surface1_color());
        _back_colors.push_back(trimesh.get_surface2_color());

        // 9) Geometries start and end indices
        _geometries_vertices.push_back(_geometries_vertices.back() +
                                       new_trimesh_n_points);
        _geometries_triangles.push_back(_geometries_triangles.back() +
                                        new_trimesh_n_triangles);

        _geometries_edges.push_back(_geometries_edges.back() +
                                    new_trimesh_n_edges);

        _geometries_perimeter_edges.push_back(
            _geometries_perimeter_edges.back() +
            static_cast<MeshIndex>(trimesh.get_perimeter_edges().size()));

        // 10) Add geometry id
        _geometries_id.push_back(geometry_id);

        // 11) Aggregate edges
        EdgesList adjusted_edges = trimesh.get_edges();
        for (auto& edge : adjusted_edges) {
            // Use std::transform to increment each element in the collection
            std::transform(edge.begin(), edge.end(), edge.begin(),
                           [current_n_points](auto index) {
                               return index + current_n_points;
                           });
        }
        _edges.insert(_edges.end(), adjusted_edges.begin(),
                      adjusted_edges.end());

        // 12) Aggregate perimeter_edges
        Edges adjusted_perimeter_edges = trimesh.get_perimeter_edges();

        // for index: adjusted_perimeter_edges) { index += current_n_edges; }
        std::transform(
            adjusted_perimeter_edges.begin(), adjusted_perimeter_edges.end(),
            adjusted_perimeter_edges.begin(),
            [current_n_edges](auto index) { return index + current_n_edges; });

        // Calculate the total size needed
        const Index previous_number_of_perimeter_edges =
            _perimeter_edges.rows();

        // Resize _perimeter_edges to fit both the original and the new elements
        _perimeter_edges.conservativeResize(previous_number_of_perimeter_edges +
                                                adjusted_perimeter_edges.rows(),
                                            Eigen::NoChange);

        // Use Eigen's block operation to copy the elements from
        // adjusted_perimeter_edges into the correct position in
        // _perimeter_edges
        _perimeter_edges.block(previous_number_of_perimeter_edges, 0,
                               adjusted_perimeter_edges.rows(), 1) =
            adjusted_perimeter_edges;

        // 12) Aggregate faces_edges
        FaceEdges adjusted_faces_edges = trimesh.get_faces_edges();
        for (auto& face : adjusted_faces_edges) {
            // for index: face) { index += current_n_edges; }
            std::transform(face.begin(), face.end(), face.begin(),
                           [current_n_edges](auto index) {
                               return index + current_n_edges;
                           });
        }

        _faces_edges.insert(_faces_edges.end(), adjusted_faces_edges.begin(),
                            adjusted_faces_edges.end());
    }

    TriMesh get_geometry_mesh(MeshIndex mesh_idx) {
        // Create TriMesh object corresponding to the mesh of the geometry with
        // id mesh_idx Check that the mesh_idx is valid
        if (mesh_idx >= _n_geometries) {
            throw std::invalid_argument("Invalid mesh index");
        }

        // Get the start and end indices of the vertices, triangles and edges
        const MeshIndex start_vertices = _geometries_vertices[mesh_idx];
        const MeshIndex end_vertices = _geometries_vertices[mesh_idx + 1];
        const MeshIndex start_triangles = _geometries_triangles[mesh_idx];
        const MeshIndex end_triangles = _geometries_triangles[mesh_idx + 1];
        const MeshIndex start_edges = _geometries_edges[mesh_idx];
        const MeshIndex end_edges = _geometries_edges[mesh_idx + 1];
        const MeshIndex start_perimeter_edges =
            _geometries_perimeter_edges[mesh_idx];
        const MeshIndex end_perimeter_edges =
            _geometries_perimeter_edges[mesh_idx + 1];

        // Copy the content to new objects
        // Vertices
        VerticesListDouble new_vertices =
            _vertices.block(start_vertices, 0, end_vertices - start_vertices, 3)
                .cast<double>();
        // Triangles
        TrianglesList new_triangles = _triangles.block(
            start_triangles, 0, end_triangles - start_triangles, 3);
        new_triangles.array() -= start_vertices;
        // Faces id
        FaceIdsList new_face_ids = _face_ids.block(
            start_triangles, 0, end_triangles - start_triangles, 1);

        // Edges
        EdgesList new_edges(_edges.begin() + start_edges,
                            _edges.begin() + end_edges);
        for (auto& edge : new_edges) {
            // for index: edge) { index -= start_vertices; }
            std::transform(edge.begin(), edge.end(), edge.begin(),
                           [start_vertices](auto index) {
                               return index - start_vertices;
                           });
        }

        // Perimeter edges
        auto new_number_of_perimeter_edges =
            end_perimeter_edges - start_perimeter_edges;
        Edges new_perimeter_edges(new_number_of_perimeter_edges);
        // Use Eigen's block operation to select the subrange and then apply a
        // unary operation
        new_perimeter_edges =
            _perimeter_edges
                .segment(start_perimeter_edges, new_number_of_perimeter_edges)
                .unaryExpr(
                    [start_edges](MeshIndex n) { return n - start_edges; });

        // Faces edges
        // TODO(PERFORMANCE): Not needed if faces are sorted
        const MeshIndex face_idx_start =
            *(std::min_element(new_face_ids.begin(), new_face_ids.end())) / 2;
        const MeshIndex face_idx_end =
            *(std::max_element(new_face_ids.begin(), new_face_ids.end())) / 2;

        FaceEdges new_faces_edges(_faces_edges.begin() + face_idx_start,
                                  _faces_edges.begin() + face_idx_end + 1);

        // Correct the idx of the edges
        for (auto& face_edges : new_faces_edges) {
            // for edge: face_edges) { edge -= start_edges; }
            std::transform(
                face_edges.begin(), face_edges.end(), face_edges.begin(),
                [start_edges](auto edge) { return edge - start_edges; });
        }

        // Color
        const ColorRGB surf_1_color = _front_colors[mesh_idx];
        const ColorRGB surf_2_color = _back_colors[mesh_idx];

        // Create the TriMesh object
        TriMesh geometry_mesh;
        geometry_mesh.set_vertices(std::move(new_vertices));
        geometry_mesh.set_triangles(std::move(new_triangles));
        geometry_mesh.set_face_ids(std::move(new_face_ids));
        geometry_mesh.set_edges(std::move(new_edges));
        geometry_mesh.set_perimeter_edges(std::move(new_perimeter_edges));
        geometry_mesh.set_faces_edges(std::move(new_faces_edges));
        geometry_mesh.set_surface1_color(surf_1_color);
        geometry_mesh.set_surface2_color(surf_2_color);
        return geometry_mesh;
    }
    void clear() {
        // Resize all Eigen Matrix objects to 0
        _vertices.resize(0, 3);
        _triangles.resize(0, 3);
        _face_ids.resize(0, 1);
        _perimeter_edges.resize(0, 1);

        // Clear all std::vector objects
        _face_cumarea.clear();
        _face_activity.clear();
        _opticals.clear();
        _front_colors.clear();
        _back_colors.clear();
        _geometries_triangles.clear();
        _geometries_triangles.push_back(0);
        _geometries_vertices.clear();
        _geometries_vertices.push_back(0);
        _geometries_edges.clear();
        _geometries_edges.push_back(0);
        _geometries_perimeter_edges.clear();
        _geometries_perimeter_edges.push_back(0);

        _geometries_id.clear();
        _edges.clear();
        _faces_edges.clear();

        // Reset all simple variables
        _n_faces = 0;
        _n_geometries = 0;
    }
};

// Define the pointers to the classes
using TriMeshPtr = std::shared_ptr<TriMesh>;
using TriMeshModelPtr = std::shared_ptr<TriMeshModel>;

namespace trimesher {

inline void print_point2d(const Point2D& p) {
    std::cout << "[" << p[0] << ", " << p[1] << "],"
              << "\n";
}

inline void print_point3d(const Point3D& p) {
    std::cout << "[" << p[0] << ", " << p[1] << ", " << p[2] << "],"
              << "\n";
}

inline void print_points(const TriMesh& trimesh) {
    for (int i = 0; i < trimesh.get_vertices().rows(); ++i) {
        std::cout << "Point:" << i << "\n";
        print_point3d(trimesh.get_vertices().row(i));
    }
}

/*
                  2D Rectangular mesh definition

                 e06       e07           e08
           P18---P19---P20------P21------P22--------P23
            |           |        |                   |
            |    F3     |   F4   |       F5          |
            |           |        |                   |
     ^     P12   P13   P14      P15      P16        P17
dir2 |      |           |        |                   |
     |  e10 |        e12|     e14|                   | e16
            |           |        |                   |
           P06---P07---P08------P09------P10--------P11
            |    e03    |  e04   |       e05         |
            |           |        |                   |
        e09 |    F0     |   F1   |        F2         | e15
            |           |        |                   |
           P00---P01---P02------P03------P04--------P05
                 e00       e01           e02

 dir1_mesh  +-----------+--------+--------------------+
           [0]         [1]      [2]                  [3]

                            ------>
                              dir1
Note:
  1. C arrays starts at 0, so P00 is actually P[0]
  2. Points at the corner of each face are always defined.
  3. Middle points at the face edges might or not be added depending on
     the value of max_distance_points_dir1 and max_distance_points_dir2.
     The distance between points on an edge will always be less or equal
     to max_distance_points_dirX (depending on the direction of the edge).
  4. Edges are vectors of indexes to the corresponding points. In the above
     diagram (notice the 0 indexing):
       P = {P00, P01, ..., P21}
       e01 = {0, 1, 2}, e02 = {2, 3}

Pxx -> Points of the mesh
Fxx -> Faces of the mesh
exx -> Edges of the mesh

Preconditions:
  1. dir1_mesh is sorted and dir1_mesh.size() >= 2
  2. dir2_mesh is sorted and dir2_mesh.size() >= 2
*/
// TODO: Refactor this function to reduce its complexity. clang-tidy warning.
// NOLINTBEGIN(readability-function-cognitive-complexity)
inline TriMesh create_2d_rectangular_mesh(const Eigen::VectorXd& dir1_mesh,
                                          const Eigen::VectorXd& dir2_mesh,
                                          double max_distance_points_dir1,
                                          double max_distance_points_dir2) {
    // Assert preconditions (only in debug mode)
    assert(dir1_mesh.size() >= 2);
    assert(dir2_mesh.size() >= 2);
    assert(2 * dir1_mesh.size() * dir2_mesh.size() <=
           std::numeric_limits<MeshIndex>::max());
    assert(utils::is_sorted(dir1_mesh));
    assert(utils::is_sorted(dir2_mesh));

    // TODO(PERFORMANCE): Use Eigen instead of std::vector to avoid casting

    // 1. Determine the number of points to reserve space
    auto dir1_size = static_cast<MeshIndex>(dir1_mesh.size());
    auto dir2_size = static_cast<MeshIndex>(dir2_mesh.size());

    MeshIndex num_points_dir1 = dir1_size;  // Minimum number of points
    MeshIndex num_points_dir2 = dir2_size;  // Minimum number of points

    std::vector<MeshIndex> additional_points_dir1(dir1_size - 1, 0);
    std::vector<MeshIndex> additional_points_dir2(dir2_size - 1, 0);

    // Now check if additional points are needed based on
    // max_distance_points If max_distance is negative or zero, then we
    // don't add additional points
    if (max_distance_points_dir1 > LENGTH_TOL) {
        for (MeshIndex i = 0; i < dir1_size - 1; ++i) {
            const double distance = dir1_mesh[i + 1] - dir1_mesh[i];
            const auto num_additional_points = static_cast<MeshIndex>(
                std::floor(distance / max_distance_points_dir1));
            num_points_dir1 += num_additional_points;
            additional_points_dir1[i] = num_additional_points;
        }
    }

    if (max_distance_points_dir2 > LENGTH_TOL) {
        for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
            const double distance = dir2_mesh[i + 1] - dir2_mesh[i];
            const auto num_additional_points = static_cast<MeshIndex>(
                std::floor(distance / max_distance_points_dir2));
            num_points_dir2 += num_additional_points;
            additional_points_dir2[i] = num_additional_points;
        }
    }

    // 2. Create the mesh points (3D points with z = 0)
    VerticesList points(num_points_dir1 * num_points_dir2, 3);
    std::vector<double> full_dir1_mesh(
        static_cast<VectorIndex>(num_points_dir1));
    std::vector<double> full_dir2_mesh(
        static_cast<VectorIndex>(num_points_dir2));

    // Fill the full mesh arrays with the original and additional mesh
    // points
    MeshIndex p_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        const double distance = (dir1_mesh[i_dir1 + 1] - dir1_mesh[i_dir1]) /
                                (additional_points_dir1[i_dir1] + 1);
        for (MeshIndex i_add1 = 0; i_add1 <= additional_points_dir1[i_dir1];
             ++i_add1) {
            full_dir1_mesh[p_idx] =
                dir1_mesh[i_dir1] + static_cast<double>(i_add1) * distance;
            ++p_idx;
        }
    }
    full_dir1_mesh[full_dir1_mesh.size() - 1] = dir1_mesh[dir1_size - 1];
    p_idx = 0;
    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
        const double distance = (dir2_mesh[i_dir2 + 1] - dir2_mesh[i_dir2]) /
                                (additional_points_dir2[i_dir2] + 1);
        for (MeshIndex i_add1 = 0; i_add1 <= additional_points_dir2[i_dir2];
             ++i_add1) {
            full_dir2_mesh[p_idx] =
                dir2_mesh[i_dir2] + static_cast<double>(i_add1) * distance;
            ++p_idx;
        }
    }
    full_dir2_mesh[full_dir2_mesh.size() - 1] = dir2_mesh[dir2_size - 1];

    // Fill the points array
    p_idx = 0;
    for (MeshIndex i_dir2 = 0; i_dir2 < num_points_dir2; ++i_dir2) {
        for (MeshIndex i_dir1 = 0; i_dir1 < num_points_dir1; ++i_dir1) {
            points(p_idx, 0) = full_dir1_mesh[static_cast<VectorIndex>(i_dir1)];
            points(p_idx, 1) = full_dir2_mesh[static_cast<VectorIndex>(i_dir2)];
            points(p_idx, 2) = 0.0;
            ++p_idx;
        }
    }

    // 3. Create the edges

    // Reserve space for the edges
    EdgesList edges((dir1_size - 1) * (dir2_size) +
                    (dir1_size) * (dir2_size - 1));

    // Fill the edges

    // First the edges in direction 1
    MeshIndex e_idx = 0;
    MeshIndex skip_dir2_rows = 0;
    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size; ++i_dir2) {
        p_idx = num_points_dir1 * (i_dir2 + skip_dir2_rows);
        for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
            const MeshIndex num_edge_points =
                additional_points_dir1[i_dir1] + 2;
            Edges edge(num_edge_points);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
            // num_edge_points where a compiled constant, it would be fine
            edge[0] = p_idx;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

            for (MeshIndex i_add1 = 0; i_add1 <= additional_points_dir1[i_dir1];
                 ++i_add1) {
                edge[i_add1 + 1] = p_idx + i_add1 + 1;
            }
            edges[e_idx] = edge;
            e_idx++;
            p_idx += num_edge_points - 1;
        }
        if (i_dir2 < dir2_size - 1) {
            skip_dir2_rows += additional_points_dir2[i_dir2];
        }
    }

    // Then the edges in direction 2
    MeshIndex skip_dir1_cols = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size; ++i_dir1) {
        p_idx = i_dir1 + skip_dir1_cols;
        for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
            const MeshIndex num_edge_points =
                additional_points_dir2[i_dir2] + 2;
            Edges edge(num_edge_points);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
            // num_edge_points where a compiled constant, it would be fine
            edge[0] = p_idx;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
            for (MeshIndex i_add2 = 0; i_add2 <= additional_points_dir2[i_dir2];
                 ++i_add2) {
                edge[i_add2 + 1] = p_idx + (i_add2 + 1) * num_points_dir1;
            }
            edges[e_idx] = edge;
            e_idx++;
            p_idx += (num_edge_points - 1) * num_points_dir1;
        }
        if (i_dir1 < dir1_size - 1) {
            skip_dir1_cols += additional_points_dir1[i_dir1];
        }
    }

    // 4. Create the permiter edges. Anti-clockwise order starting from P01
    // -> dir1 edge
    const MeshIndex num_perimeter_edges = 2 * (dir1_size + dir2_size - 2);
    EdgesIdsList perimeter_edges(num_perimeter_edges);
    MeshIndex per_edge_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        perimeter_edges[per_edge_idx] = i_dir1;
        ++per_edge_idx;
    }
    MeshIndex skip_horizontal_edges = (dir2_size) * (dir1_size - 1);
    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
        perimeter_edges[per_edge_idx] =
            i_dir2 + (dir2_size - 1) * (dir1_size - 1) + skip_horizontal_edges;
        ++per_edge_idx;
    }
    skip_horizontal_edges = (dir2_size - 1) * (dir1_size - 1);
    // Reverse order loop for anti-clockwise edges order
    // To loop in reverse order with an unsigned type, we need to use a
    // trick. See: https://stackoverflow.com/a/5458251
    for (MeshIndex i_dir1 = dir1_size - 2; i_dir1 != static_cast<MeshIndex>(-1);
         --i_dir1) {
        perimeter_edges[per_edge_idx] = i_dir1 + skip_horizontal_edges;
        ++per_edge_idx;
    }
    skip_horizontal_edges = (dir2_size) * (dir1_size - 1);
    // Reverse order loop for anti-clockwise edges order
    // To loop in reverse order with an unsigned type, we need to use a
    // trick. See: https://stackoverflow.com/a/5458251
    for (MeshIndex i_dir2 = dir2_size - 2; i_dir2 != static_cast<MeshIndex>(-1);
         --i_dir2) {
        perimeter_edges[per_edge_idx] = i_dir2 + skip_horizontal_edges;
        ++per_edge_idx;
    }

    // 5. Create the faces edges
    const MeshIndex num_faces = (dir1_size - 1) * (dir2_size - 1);
    FaceEdges faces_edges(num_faces, EdgesIdsList(4));
    MeshIndex face_idx = 0;
    skip_horizontal_edges = (dir2_size) * (dir1_size - 1);
    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
        for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
            faces_edges[face_idx][0] = i_dir1 + (dir1_size - 1) * i_dir2;
            faces_edges[face_idx][1] =
                i_dir2 + (dir2_size - 1) * (i_dir1 + 1) + skip_horizontal_edges;
            faces_edges[face_idx][2] = i_dir1 + (dir1_size - 1) * (i_dir2 + 1);
            faces_edges[face_idx][3] =
                i_dir2 + (dir2_size - 1) * (i_dir1) + skip_horizontal_edges;
            ++face_idx;
        }
    }

    // 6. Create the trimesh

    TriMesh trimesh;

    trimesh.set_vertices(std::move(points));
    trimesh.set_edges(std::move(edges));
    trimesh.set_perimeter_edges(std::move(perimeter_edges));
    trimesh.set_faces_edges(std::move(faces_edges));
    trimesh.set_number_of_faces(num_faces * 2);

    return trimesh;
}
// NOLINTEND(readability-function-cognitive-complexity)

// This function differs from create_2d_rectangular_mesh in that it
// the order of points is different. First the points that lies in the dir1
// lines, then the additional points in the dir2 lines. Finally the interior
// points.
// TODO: Refactor this function to reduce its complexity. clang-tidy warning.
// NOLINTBEGIN(readability-function-cognitive-complexity)
inline TriMesh create_2d_quadrilateral_mesh(
    const Eigen::VectorXd& dir1_mesh_normalized,
    const Eigen::VectorXd& dir2_mesh_normalized, const Point2D& p1,
    const Point2D& p2, const Point2D& p3, const Point2D& p4,
    double max_distance_points_hdir, double max_distance_points_vdir) {
    // Assert preconditions (only in debug mode)
    assert(dir1_mesh_normalized.size() >= 2);
    assert(dir2_mesh_normalized.size() >= 2);
    assert(2 * dir1_mesh_normalized.size() * dir2_mesh_normalized.size() <=
           std::numeric_limits<MeshIndex>::max());
    assert(utils::is_sorted(dir1_mesh_normalized));
    assert(utils::is_sorted(dir2_mesh_normalized));

    // Normalize the mesh
    // Eigen::VectorXd dir1_mesh_normalized = dir1_mesh;
    // Eigen::VectorXd dir2_mesh_normalized = dir2_mesh;
    // Eigen::VectorXd dir1_mesh_normalized =
    //     dir1_mesh / (dir1_mesh[dir1_mesh.size() - 1]);
    // Eigen::VectorXd dir2_mesh_normalized =
    //     dir2_mesh / (dir2_mesh[dir2_mesh.size() - 1]);
    auto v12 = p2 - p1;
    auto v23 = p3 - p2;
    auto v14 = p4 - p1;
    auto v43 = p3 - p4;

    const Eigen::VectorXd dir1_mesh = dir1_mesh_normalized * v12.norm();
    const Eigen::VectorXd dir2_mesh = dir2_mesh_normalized * v23.norm();

    // 1. Determine the number of points to reserve space
    const auto dir1_size = static_cast<MeshIndex>(dir1_mesh.size());
    const auto dir2_size = static_cast<MeshIndex>(dir2_mesh.size());

    MeshIndex num_points_edges =
        dir1_size * dir2_size;  // Minimum number of points

    // In the quadrilateral case, each line of the mesh has a different length,
    // so it might result in different numbers of points to be added. Note also
    // that the direction of the lines for dir1 and for dir2 changes from one
    // line to the next one.
    std::vector<std::vector<MeshIndex>> additional_points_dir1(
        dir2_size, std::vector<MeshIndex>(dir1_size - 1, 0));
    std::vector<std::vector<MeshIndex>> additional_points_dir2(
        dir1_size, std::vector<MeshIndex>(dir2_size - 1, 0));

    // We also create a vector of vectors to store the dir_mesh of each line
    // dir1_meshes and dir2_meshes will contain the non-normalized mesh of each
    // line. To recover the actual points from the meshes, you would need the
    // start point and a direction. For dir1 and dir2 lines, the direction
    // changes from one line to the next one.
    std::vector<Eigen::VectorXd> dir1_meshes(dir2_size);
    std::vector<Eigen::VectorXd> dir2_meshes(dir1_size);
    std::vector<Vector2D> dir1_directions(dir2_size);
    std::vector<Vector2D> dir2_directions(dir1_size);
    std::vector<Point2D> dir1_start_points(dir2_size);
    std::vector<Point2D> dir2_start_points(dir1_size);
    for (MeshIndex i = 0; i < dir2_size; ++i) {
        auto start_point = p1 + v14 * dir2_mesh_normalized[i];
        auto end_point = p2 + v23 * dir2_mesh_normalized[i];
        dir1_start_points[i] = start_point;
        dir1_directions[i] = (end_point - start_point).normalized();
        const double distance = (end_point - start_point).norm();
        dir1_meshes[i] = dir1_mesh_normalized * distance;
    }
    for (MeshIndex i = 0; i < dir1_size; ++i) {
        auto start_point = p1 + v12 * dir1_mesh_normalized[i];
        auto end_point = p4 + v43 * dir1_mesh_normalized[i];
        dir2_start_points[i] = start_point;
        dir2_directions[i] = (end_point - start_point).normalized();
        const double distance = (end_point - start_point).norm();
        dir2_meshes[i] = dir2_mesh_normalized * distance;
    }

    // Now check if additional points are needed based on
    // max_distance_points If max_distance is negative or zero, then we
    // don't add additional points
    if (max_distance_points_hdir > LENGTH_TOL) {
        // We loop over all the dir1 ("horizontal") lines
        for (MeshIndex j = 0; j < dir2_size; ++j) {
            for (MeshIndex i = 0; i < dir1_size - 1; ++i) {
                const Eigen::VectorXd& mesh =
                    dir1_meshes[j];  // Mesh of the current line
                double hdistance = mesh[i + 1] - mesh[i];
                // Project the distance in the horizontal direction
                hdistance = hdistance * dir1_directions[j].norm();
                const auto num_additional_points = static_cast<MeshIndex>(
                    std::floor(hdistance / max_distance_points_hdir));
                num_points_edges += num_additional_points;
                additional_points_dir1[j][i] = num_additional_points;
            }
        }
    }

    if (max_distance_points_vdir > LENGTH_TOL) {
        // We loop over all the dir2 ("vertical") lines
        for (MeshIndex j = 0; j < dir1_size; ++j) {
            for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
                const Eigen::VectorXd& mesh =
                    dir2_meshes[j];  // Mesh of the current line
                double vdistance = mesh[i + 1] - mesh[i];
                // Project the distance in the vertical direction
                vdistance = vdistance * dir2_directions[j].norm();
                const auto num_additional_points = static_cast<MeshIndex>(
                    std::floor(vdistance / max_distance_points_vdir));
                num_points_edges += num_additional_points;
                additional_points_dir2[j][i] = num_additional_points;
            }
        }
    }

    // 2. Create the mesh points (3D points with z = 0)

    // Fill the full mesh arrays with the original and additional mesh
    // points
    for (MeshIndex j = 0; j < dir2_size; ++j) {
        const Eigen::VectorXd& dir_mesh =
            dir1_meshes[j];  // Mesh of the current line

        constexpr MeshIndex init_value = 0;
        const MeshIndex additional_points =
            std::accumulate(additional_points_dir1[j].begin(),
                            additional_points_dir1[j].end(), init_value);
        Eigen::VectorXd full_dir_mesh(dir1_size + additional_points);

        int idx = 0;
        for (MeshIndex i = 0; i < dir1_size - 1; ++i) {
            const double distance = (dir_mesh[i + 1] - dir_mesh[i]) /
                                    (additional_points_dir1[j][i] + 1);
            for (MeshIndex i_add = 0; i_add <= additional_points_dir1[j][i];
                 ++i_add) {
                full_dir_mesh[idx] = dir_mesh[i] + distance * i_add;
                ++idx;
            }
        }
        full_dir_mesh[full_dir_mesh.size() - 1] = dir_mesh[dir1_size - 1];
        dir1_meshes[j] = std::move(full_dir_mesh);
    }

    for (MeshIndex j = 0; j < dir1_size; ++j) {
        const Eigen::VectorXd& dir_mesh =
            dir2_meshes[j];  // Mesh of the current line

        constexpr MeshIndex init_value = 0;
        const MeshIndex additional_points =
            std::accumulate(additional_points_dir2[j].begin(),
                            additional_points_dir2[j].end(), init_value);
        Eigen::VectorXd full_dir_mesh(dir2_size + additional_points);

        int idx = 0;
        for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
            const double distance = (dir_mesh[i + 1] - dir_mesh[i]) /
                                    (additional_points_dir2[j][i] + 1);
            for (MeshIndex i_add = 0; i_add <= additional_points_dir2[j][i];
                 ++i_add) {
                full_dir_mesh[idx] = dir_mesh[i] + distance * i_add;
                ++idx;
            }
        }
        full_dir_mesh[full_dir_mesh.size() - 1] = dir_mesh[dir2_size - 1];
        dir2_meshes[j] = std::move(full_dir_mesh);
    }

    // Loop over the faces to determine the number of interior points to be
    // added
    Eigen::Matrix<MeshIndex, Eigen::Dynamic, Eigen::Dynamic>
        additional_points_faces_dir1((dir2_size - 1), (dir1_size - 1));
    Eigen::Matrix<MeshIndex, Eigen::Dynamic, Eigen::Dynamic>
        additional_points_faces_dir2((dir2_size - 1), (dir1_size - 1));

    MeshIndex num_interior_points = 0;
    for (MeshIndex j = 0; j < dir2_size - 1; ++j) {
        for (MeshIndex i = 0; i < dir1_size - 1; ++i) {
            additional_points_faces_dir1(j, i) = std::max(
                additional_points_dir1[j][i], additional_points_dir1[j + 1][i]);
            additional_points_faces_dir2(j, i) = std::max(
                additional_points_dir2[i][j], additional_points_dir2[i + 1][j]);
            num_interior_points += additional_points_faces_dir1(j, i) *
                                   additional_points_faces_dir2(j, i);
        }
    }

    // Fill the points array

    VerticesList points(num_points_edges + num_interior_points, 3);
    MeshIndex p_idx = 0;

    // Loop over dir1 lines
    for (MeshIndex j = 0; j < dir2_size; ++j) {
        const Eigen::VectorXd& dir_mesh =
            dir1_meshes[j];  // Mesh of the current line
        const Vector2D& dir = dir1_directions[j];
        const Point2D& start_point = dir1_start_points[j];
        for (const auto& mesh_i : dir_mesh) {
            points(p_idx, 0) = start_point.x() + mesh_i * dir.x();
            points(p_idx, 1) = start_point.y() + mesh_i * dir.y();
            points(p_idx, 2) = 0.0;
            ++p_idx;
        }
    }

    // Loop over dir2 lines
    for (MeshIndex j = 0; j < dir1_size; ++j) {
        const Eigen::VectorXd& dir_mesh =
            dir2_meshes[j];  // Mesh of the current line
        const Vector2D& dir = dir2_directions[j];
        const Point2D& start_point = dir2_start_points[j];
        int line_point_idx = 0;
        for (MeshIndex i = 0; i < dir2_mesh_normalized.size() - 1; ++i) {
            // We skip the non-additional points as those are already added
            for (MeshIndex iadd = 0; iadd < additional_points_dir2[j][i];
                 ++iadd) {
                points(p_idx, 0) =
                    start_point.x() + dir_mesh[line_point_idx + 1] * dir.x();
                points(p_idx, 1) =
                    start_point.y() + dir_mesh[line_point_idx + 1] * dir.y();
                points(p_idx, 2) = 0.0;
                ++p_idx;
                ++line_point_idx;
            }
            ++line_point_idx;
        }
    }

    // 3. Create the edges

    // Reserve space for the edges
    EdgesList edges((dir1_size - 1) * (dir2_size) +
                    (dir1_size) * (dir2_size - 1));

    // Fill the edges

    // First the edges in direction 1
    MeshIndex e_idx = 0;
    p_idx = 0;

    // Loop over dir1 lines (edges in dir1)
    for (MeshIndex j = 0; j < dir2_size; ++j) {
        for (MeshIndex i = 0; i < dir1_size - 1; ++i) {
            const MeshIndex num_edge_points = additional_points_dir1[j][i] + 2;
            Edges edge(num_edge_points);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
            // num_edge_points where a compiled constant, it would be fine
            edge[0] = p_idx;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
            for (MeshIndex i_add = 0; i_add <= additional_points_dir1[j][i];
                 ++i_add) {
                edge[i_add + 1] = p_idx + i_add + 1;
            }
            edges[e_idx] = edge;
            e_idx++;
            p_idx += num_edge_points - 1;
        }
        p_idx++;
    }

    // Loop over dir2 lines (edges in dir2)
    for (MeshIndex j = 0; j < dir1_size - 1; ++j) {
        for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
            const MeshIndex lower_edge_idx = (dir1_size - 1) * i + j;
            const MeshIndex upper_edge_idx = (dir1_size - 1) * (i + 1) + j;
            const MeshIndex num_edge_points = additional_points_dir2[j][i] + 2;
            Edges edge(num_edge_points);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
            // num_edge_points where a compiled constant, it would be fine
            edge[0] = edges[lower_edge_idx][0];
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
            edge[num_edge_points - 1] = edges[upper_edge_idx][0];
            for (MeshIndex i_add = 0; i_add < additional_points_dir2[j][i];
                 ++i_add) {
                edge[i_add + 1] = p_idx;
                p_idx++;
            }
            edges[e_idx] = edge;
            e_idx++;
        }
    }
    // Last edge
    for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
        const MeshIndex lower_edge_idx = (dir1_size - 1) * i + dir1_size - 2;
        const MeshIndex upper_edge_idx =
            (dir1_size - 1) * (i + 1) + dir1_size - 2;
        const MeshIndex num_edge_points =
            additional_points_dir2[dir1_size - 1][i] + 2;
        Edges edge(num_edge_points);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
        edge[0] = edges[lower_edge_idx][edges[lower_edge_idx].size() - 1];
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
        edge[num_edge_points - 1] =
            edges[upper_edge_idx][edges[upper_edge_idx].size() - 1];
        for (MeshIndex i_add = 0;
             i_add < additional_points_dir2[dir1_size - 1][i]; ++i_add) {
            edge[i_add + 1] = p_idx;
            p_idx++;
        }
        edges[e_idx] = edge;
        e_idx++;
    }

    // 4. Create the permiter edges. Anti-clockwise order starting from P01
    // -> dir1 edge
    const MeshIndex num_perimeter_edges = 2 * (dir1_size + dir2_size - 2);
    EdgesIdsList perimeter_edges(num_perimeter_edges);
    MeshIndex per_edge_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        perimeter_edges[per_edge_idx] = i_dir1;
        ++per_edge_idx;
    }
    MeshIndex skip_horizontal_edges = (dir2_size) * (dir1_size - 1);
    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
        perimeter_edges[per_edge_idx] =
            i_dir2 + (dir2_size - 1) * (dir1_size - 1) + skip_horizontal_edges;
        ++per_edge_idx;
    }
    skip_horizontal_edges = (dir2_size - 1) * (dir1_size - 1);
    // Reverse order loop for anti-clockwise edges order
    // To loop in reverse order with an unsigned type, we need to use a
    // trick. See: https://stackoverflow.com/a/5458251
    for (MeshIndex i_dir1 = dir1_size - 2; i_dir1 != static_cast<MeshIndex>(-1);
         --i_dir1) {
        perimeter_edges[per_edge_idx] = i_dir1 + skip_horizontal_edges;
        ++per_edge_idx;
    }

    skip_horizontal_edges = (dir2_size) * (dir1_size - 1);
    // Reverse order loop for anti-clockwise edges order
    // To loop in reverse order with an unsigned type, we need to use a
    // trick. See: https://stackoverflow.com/a/5458251
    for (MeshIndex i_dir2 = dir2_size - 2; i_dir2 != static_cast<MeshIndex>(-1);
         --i_dir2) {
        perimeter_edges[per_edge_idx] = i_dir2 + skip_horizontal_edges;
        ++per_edge_idx;
    }

    // 5. Create the faces edges
    const MeshIndex num_faces = (dir1_size - 1) * (dir2_size - 1);
    FaceEdges faces_edges(num_faces, EdgesIdsList(4));
    MeshIndex face_idx = 0;
    skip_horizontal_edges = (dir2_size) * (dir1_size - 1);
    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
        for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
            faces_edges[face_idx][0] = i_dir1 + (dir1_size - 1) * i_dir2;
            faces_edges[face_idx][1] =
                i_dir2 + (dir2_size - 1) * (i_dir1 + 1) + skip_horizontal_edges;
            faces_edges[face_idx][2] = i_dir1 + (dir1_size - 1) * (i_dir2 + 1);
            faces_edges[face_idx][3] =
                i_dir2 + (dir2_size - 1) * (i_dir1) + skip_horizontal_edges;
            ++face_idx;
        }
    }

    // 2.5 Create the interior points
    // Loop over the faces adding the interior points (not used in the edges)
    if (num_interior_points > 0) {
        p_idx = num_points_edges;
        for (MeshIndex j = 0; j < dir2_size - 1; ++j) {
            for (MeshIndex i = 0; i < dir1_size - 1; ++i) {
                const MeshIndex i_face = j * (dir1_size - 1) + i;
                // Copying the points (instead of referencing) because points is
                // col-major
                const Point2D p1_f(
                    points.row(edges[faces_edges[i_face][0]][0]).head<2>());
                const Point2D p2_f(
                    points.row(edges[faces_edges[i_face][1]][0]).head<2>());

                const Point2D p3_f(
                    points.row(edges[faces_edges[i_face][2]].tail<1>().value())
                        .head<2>());
                const Point2D p4_f(
                    points.row(edges[faces_edges[i_face][3]].tail<1>().value())
                        .head<2>());

                const MeshIndex dir1_add_points =
                    additional_points_faces_dir1(j, i);
                const MeshIndex dir2_add_points =
                    additional_points_faces_dir2(j, i);

                Vector2D v14_f = p4_f - p1_f;
                v14_f = v14_f / (dir2_add_points + 1);
                Vector2D v23_f = p3_f - p2_f;
                v23_f = v23_f / (dir2_add_points + 1);

                for (MeshIndex ip_dir2 = 1; ip_dir2 <= dir2_add_points;
                     ++ip_dir2) {
                    auto start_point = p1_f + v14_f * (ip_dir2);
                    auto end_point = p2_f + v23_f * (ip_dir2);
                    Vector2D vdir1 = end_point - start_point;
                    vdir1 = vdir1 / (dir1_add_points + 1);
                    for (MeshIndex ip_dir1 = 1; ip_dir1 <= dir1_add_points;
                         ++ip_dir1) {
                        points(p_idx, 0) =
                            start_point.x() + vdir1.x() * ip_dir1;
                        points(p_idx, 1) =
                            start_point.y() + vdir1.y() * ip_dir1;
                        points(p_idx, 2) = 0.0;
                        ++p_idx;
                    }
                }
            }
        }
    }

    // 6. Create the trimesh

    TriMesh trimesh;

    trimesh.set_vertices(std::move(points));
    trimesh.set_edges(std::move(edges));
    trimesh.set_perimeter_edges(std::move(perimeter_edges));
    trimesh.set_faces_edges(std::move(faces_edges));
    trimesh.set_number_of_faces(num_faces * 2);

    return trimesh;
}
// NOLINTEND(readability-function-cognitive-complexity)

// Create the mesh of a triangle
// It is assumed that there are no divisions in the dir1 direction
// so dir1_mesh is not necessary. Use the general create_2d_triangular_mesh
// function otherwise

// TODO: Refactor this function to reduce its complexity. clang-tidy warning.
// NOLINTBEGIN(readability-function-cognitive-complexity)
inline TriMesh create_2d_triangular_only_mesh(
    const Eigen::VectorXd& dir2_mesh_normalized, const Point2D& p1,
    const Point2D& p2, const Point2D& p3, double max_distance_points_hdir,
    double max_distance_points_vdir) {
    // Assert preconditions (only in debug mode)
    assert(dir2_mesh_normalized.size() >= 2);
    assert(2 * dir2_mesh_normalized.size() <=
           std::numeric_limits<MeshIndex>::max());
    assert(utils::is_sorted(dir2_mesh_normalized));

    // Normalize the mesh
    // Eigen::VectorXd dir2_mesh_normalized =
    //     dir2_mesh / (dir2_mesh[dir2_mesh.size() - 1]);
    auto v23 = p3 - p2;
    // std::cout << "v23: " << v23.norm() << std::endl;
    Eigen::VectorXd dir2_mesh =
        dir2_mesh_normalized * v23.norm();  // Mesh of the current line
    // auto v21 = p2 - p1;
    // auto v31 = p3 - p1;

    // 1. Determine the number of points to reserve space
    const auto dir2_size = static_cast<MeshIndex>(dir2_mesh.size());

    MeshIndex num_points_edges = dir2_size + 1;  // Minimum number of points

    // For each of the lines that divides the triangle in dir2, additional
    // points to be placed
    std::vector<MeshIndex> additional_points_dir1(dir2_size, 0);

    // For each of the divisions in dir2, additional points to be placed
    std::vector<MeshIndex> additional_points_dir2(dir2_size - 1, 0);

    // Now check if additional points are needed based on
    // max_distance_points If max_distance is negative or zero, then we
    // don't add additional points
    if (max_distance_points_hdir > LENGTH_TOL) {
        // We loop over all the dir1 ("horizontal") lines
        for (MeshIndex i = 0; i < dir2_size; ++i) {
            // Get the line P1 - (P2+mesh[i]*v23)
            const Vector2D dir = p2 + v23 * dir2_mesh_normalized[i] - p1;
            const double hdistance = dir.norm();
            const auto num_additional_points = static_cast<MeshIndex>(
                std::floor(hdistance / max_distance_points_hdir));
            num_points_edges += num_additional_points;
            additional_points_dir1[i] = num_additional_points;
        }
    }

    const Vector2D v23_dir = v23.normalized();
    if (max_distance_points_hdir > LENGTH_TOL) {
        // We loop over all divisions of dir2
        for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
            const double vdistance =
                v23_dir.norm() * (dir2_mesh[i + 1] - dir2_mesh[i]);
            const auto num_additional_points = static_cast<MeshIndex>(
                std::floor(vdistance / max_distance_points_vdir));
            num_points_edges += num_additional_points;
            additional_points_dir2[i] = num_additional_points;
        }
    }

    // Loop over the faces to determine the number of interior points to be
    // added
    std::vector<MeshIndex> additional_points_faces_dir1((dir2_size - 1), 0);

    MeshIndex num_interior_points = 0;
    for (MeshIndex j = 0; j < dir2_size - 1; ++j) {
        additional_points_faces_dir1[j] =
            std::max(additional_points_dir1[j], additional_points_dir1[j + 1]);
        num_interior_points +=
            additional_points_faces_dir1[j] * additional_points_dir2[j];
    }

    // 2. Create the mesh points (3D points with z = 0)

    // Fill the points array

    VerticesList points(num_points_edges + num_interior_points, 3);
    MeshIndex p_idx = 0;

    // Add p1
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
    points(p_idx, 0) = p1.x();
    points(p_idx, 1) = p1.y();
    points(p_idx, 2) = 0.0;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    ++p_idx;

    // Loop over dir1 lines
    for (MeshIndex j = 0; j < dir2_size; ++j) {
        const Vector2D dir = p2 + v23 * dir2_mesh_normalized[j] - p1;
        for (MeshIndex i = 1; i <= additional_points_dir1[j] + 1; ++i) {
            Point2D padd = p1 + dir * (i) / (additional_points_dir1[j] + 1);
            points(p_idx, 0) = padd.x();
            points(p_idx, 1) = padd.y();
            points(p_idx, 2) = 0.0;
            ++p_idx;
        }
    }
    const MeshIndex additional_points_dir2_start = p_idx;
    // Additional points in dir2 line
    for (MeshIndex j = 0; j < dir2_size - 1; ++j) {
        const Point2D start_point = p2 + v23 * dir2_mesh_normalized[j];
        const Point2D end_point = p2 + v23 * dir2_mesh_normalized[j + 1];
        const Vector2D dir =
            (end_point - start_point) / (additional_points_dir2[j] + 1);
        for (MeshIndex i = 1; i <= additional_points_dir2[j]; ++i) {
            Point2D padd = start_point + dir * i;
            points(p_idx, 0) = padd.x();
            points(p_idx, 1) = padd.y();
            points(p_idx, 2) = 0.0;
            ++p_idx;
        }
    }

    // 3. Create the edges

    // Reserve space for the edges
    EdgesList edges(dir2_size * 2 - 1);

    // Fill the edges

    // First the edges in direction 1
    MeshIndex e_idx = 0;
    p_idx = 0;

    // Loop over dir1 lines (edges in dir1)
    for (MeshIndex j = 0; j < dir2_size; ++j) {
        const MeshIndex num_edge_points = additional_points_dir1[j] + 2;
        Edges edge(num_edge_points);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
        edge[0] = 0;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
        for (MeshIndex i_add = 1; i_add <= additional_points_dir1[j] + 1;
             ++i_add) {
            p_idx++;
            edge[i_add] = p_idx;
        }
        edges[e_idx] = edge;
        e_idx++;
    }

    //     // Loop over dir2 line
    //     MeshIndex start_p_idx = 0;
    //     MeshIndex end_p_idx = additional_points_dir1[0] + 1;
    //     MeshIndex start_padd_idx = additional_points_dir2_start;
    //     for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
    //         start_p_idx = end_p_idx;
    //         end_p_idx += additional_points_dir1[i + 1] + 1;
    //         const MeshIndex num_edge_points = additional_points_dir2[i] + 2;
    //         Edges edge(num_edge_points);
    // #if defined(__GNUC__)
    // #pragma GCC diagnostic push
    // #pragma GCC diagnostic ignored "-Wnull-dereference"
    // #endif
    //         edge[0] = start_p_idx;
    //         edge.tail<1>()(0) = end_p_idx;
    // #if defined(__GNUC__)
    // #pragma GCC diagnostic pop
    // #endif
    //         for (MeshIndex i_add = 0; i_add < additional_points_dir2[i];
    //         ++i_add) {
    //             edge[i_add + 1] = start_padd_idx;
    //             start_padd_idx++;
    //         }
    //         edges[e_idx] = edge;
    //         e_idx++;
    //     }

    MeshIndex start_p_idx = additional_points_dir1[0] + 1;
    MeshIndex start_padd_idx = additional_points_dir2_start;
    for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
        const MeshIndex end_p_idx =
            start_p_idx + additional_points_dir1[i + 1] + 1;
        const MeshIndex num_edge_points = additional_points_dir2[i] + 2;
        Edges edge(num_edge_points);
        edge[0] = start_p_idx;
        edge.tail<1>()(0) = end_p_idx;

        for (MeshIndex i_add = 0; i_add < additional_points_dir2[i]; ++i_add) {
            edge[i_add + 1] = start_padd_idx;
            start_padd_idx++;
        }
        edges[e_idx] = edge;
        e_idx++;
        start_p_idx = end_p_idx;
    }

    // 4. Create the permiter edges. Anti-clockwise order starting from P01
    // -> dir1 edge
    const MeshIndex num_perimeter_edges = dir2_size - 1 + 2;
    EdgesIdsList perimeter_edges(num_perimeter_edges);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
    perimeter_edges[0] = 0;
    perimeter_edges.tail<1>()(0) = dir2_size - 1;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
        perimeter_edges[i + 1] = perimeter_edges.tail<1>()(0) + i + 1;
    }

    // 5. Create the faces edges
    const MeshIndex num_faces = (dir2_size - 1);
    FaceEdges faces_edges(num_faces, EdgesIdsList(3));

    for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
        faces_edges[i][0] = i;
        faces_edges[i][1] = dir2_size + i;
        faces_edges[i][2] = i + 1;
    }

    // 2.5 Create the interior points
    // Loop over the faces adding the interior points (not used in the edges)
    if (num_interior_points > 0) {
        p_idx = num_points_edges;
        Point2D start_point = p1;
        MeshIndex end_point_idx = additional_points_dir2_start;
        for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
            for (MeshIndex ip_dir2 = 0; ip_dir2 < additional_points_dir2[i];
                 ++ip_dir2) {
                const Point2D end_point(points.row(end_point_idx).head<2>());
                end_point_idx++;
                Vector2D vdir1 = end_point - start_point;
                vdir1 = vdir1 / (additional_points_faces_dir1[i] + 1);
                for (MeshIndex ip_dir1 = 1;
                     ip_dir1 <= additional_points_faces_dir1[i]; ++ip_dir1) {
                    points(p_idx, 0) = start_point.x() + vdir1.x() * ip_dir1;
                    points(p_idx, 1) = start_point.y() + vdir1.y() * ip_dir1;
                    points(p_idx, 2) = 0.0;
                    ++p_idx;
                }
            }
        }
    }

    // 6. Create the trimesh

    TriMesh trimesh;

    trimesh.set_vertices(std::move(points));
    trimesh.set_edges(std::move(edges));
    trimesh.set_perimeter_edges(std::move(perimeter_edges));
    trimesh.set_faces_edges(std::move(faces_edges));
    trimesh.set_number_of_faces(num_faces * 2);

    return trimesh;
}
// NOLINTEND(readability-function-cognitive-complexity)

// Create the mesh of a general triangle.
// It uses the functions create_2d_triangular_only_mesh and
// create_2d_quadrilateral_mesh and then merges the two meshes

// TODO: Refactor this function to reduce its complexity. clang-tidy warning.
// NOLINTBEGIN(readability-function-cognitive-complexity)
inline TriMesh create_2d_triangular_mesh(
    const Eigen::VectorXd& dir1_mesh_normalized,
    const Eigen::VectorXd& dir2_mesh_normalized, const Point2D& p1,
    const Point2D& p2, const Point2D& p3, double max_distance_points_hdir,
    double max_distance_points_vdir) {
    // Assert preconditions (only in debug mode)
    assert(dir1_mesh_normalized.size() >= 2);
    assert(dir2_mesh_normalized.size() >= 2);
    assert(2 * dir1_mesh_normalized.size() * dir2_mesh_normalized.size() <=
           std::numeric_limits<MeshIndex>::max());
    assert(utils::is_sorted(dir1_mesh_normalized));
    assert(utils::is_sorted(dir2_mesh_normalized));

    // If there are no divisions in the dir1 direction, then we use the
    // triangular only mesh
    if (dir1_mesh_normalized.size() == 2) {
        return create_2d_triangular_only_mesh(dir2_mesh_normalized, p1, p2, p3,
                                              max_distance_points_hdir,
                                              max_distance_points_vdir);
    }

    // Normalize the mesh
    // Eigen::VectorXd dir1_mesh_normalized =
    //     dir1_mesh / (dir1_mesh[dir1_mesh.size() - 1]);
    // const Eigen::VectorXd dir2_mesh_normalized =
    //     dir2_mesh / (dir2_mesh[dir2_mesh.size() - 1]);
    auto v21 = p2 - p1;
    // auto v23 = p3 - p2;
    auto v31 = p3 - p1;
    // Scale the mesh
    const Eigen::VectorXd dir1_mesh =
        dir1_mesh_normalized * v21.norm();  // Mesh of the current line
    const Eigen::VectorXd dir2_mesh =
        dir2_mesh_normalized * v31.norm();  // Mesh of the current line

    auto dir1_size = static_cast<MeshIndex>(dir1_mesh.size());
    auto dir2_size = static_cast<MeshIndex>(dir2_mesh.size());

    // Otherwise we need to create a meshed triangle and a meshed quadrilateral
    // and join them together
    const Eigen::VectorXd triangle_dir1_mesh = dir1_mesh.head(2);
    const auto& triangle_p1 = p1;
    const auto triangle_p2 = p1 + v21 * dir1_mesh_normalized[1];
    const auto triangle_p3 = p1 + v31 * dir1_mesh_normalized[1];
    const Eigen::VectorXd triangle_dir2_mesh =
        dir2_mesh_normalized * (triangle_p3 - triangle_p2).norm();

    Eigen::VectorXd quad_dir1_mesh = dir1_mesh(seq(1, dir1_mesh.size() - 1));
    // auto quad_dir1_mesh_0 = quad_dir1_mesh[0];
    // for (auto& quad_dir1 : quad_dir1_mesh) {
    //     quad_dir1 = quad_dir1 - quad_dir1_mesh_0;
    // }
    quad_dir1_mesh.array() -= quad_dir1_mesh[0];

    // auto first_value = quad_dir1_mesh[0];
    // std::transform(
    //     quad_dir1_mesh.begin(), quad_dir1_mesh.end(), quad_dir1_mesh.begin(),
    //     [first_value](auto quad_dir1) { return quad_dir1 - first_value; });

    const Eigen::VectorXd& quad_dir2_mesh = dir2_mesh;
    const auto& quad_p1 = triangle_p2;
    const auto& quad_p2 = p2;
    const auto& quad_p3 = p3;
    const auto& quad_p4 = triangle_p3;

    // Normalize the meshes
    const Eigen::VectorXd triangle_dir2_mesh_normalized =
        triangle_dir2_mesh /
        (triangle_dir2_mesh[triangle_dir2_mesh.size() - 1]);
    const Eigen::VectorXd quad_dir1_mesh_normalized =
        quad_dir1_mesh / (quad_dir1_mesh[quad_dir1_mesh.size() - 1]);
    const Eigen::VectorXd quad_dir2_mesh_normalized =
        quad_dir2_mesh / (quad_dir2_mesh[quad_dir2_mesh.size() - 1]);

    // Create the triangular mesh
    TriMesh tri_mesh = create_2d_triangular_only_mesh(
        triangle_dir2_mesh_normalized, triangle_p1, triangle_p2, triangle_p3,
        max_distance_points_hdir, max_distance_points_vdir);

    // Create the quadrilateral mesh
    TriMesh quad_mesh = create_2d_quadrilateral_mesh(
        quad_dir1_mesh_normalized, quad_dir2_mesh_normalized, quad_p1, quad_p2,
        quad_p3, quad_p4, max_distance_points_hdir, max_distance_points_vdir);

    auto tri_vertices = tri_mesh.get_vertices();
    auto quad_vertices = quad_mesh.get_vertices();
    auto tri_edges = tri_mesh.get_edges();
    auto quad_edges = quad_mesh.get_edges();
    auto tri_perimeter_edges = tri_mesh.get_perimeter_edges();
    auto quad_perimeter_edges = quad_mesh.get_perimeter_edges();
    auto tri_faces_edges = tri_mesh.get_faces_edges();
    auto quad_faces_edges = quad_mesh.get_faces_edges();

    // Edges to remove from quad
    // (dir1_size-2)*dir2_size, (dir1_size-2)*dir2_size+1, ...,
    // (dir1_size-2)*dir2_size+dir2_size-2
    std::vector<MeshIndex> quad_edges_to_remove(dir2_size - 1);
    for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
        quad_edges_to_remove[i] = (dir1_size - 2) * dir2_size + i;
    }

    // Border edges of the pure triangle
    std::vector<MeshIndex> tri_edges_border(dir2_size - 1);
    for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
        tri_edges_border[i] = dir2_size + i;
    }

    // int edge_idx = 0;
    std::set<MeshIndex> quad_edges_to_remove_set;
    for (const auto& edge_idx : quad_edges_to_remove) {
        auto edge = quad_edges[edge_idx];
        for (const auto& vert_idx : edge) {
            quad_edges_to_remove_set.insert(vert_idx);
        }
    }

    // Create a new matrix without the contact points
    const auto num_rows_to_remove =
        static_cast<MeshIndex>(quad_edges_to_remove_set.size());
    const auto num_rows_original_matrix =
        static_cast<MeshIndex>(quad_vertices.rows());
    const auto num_rows_new_matrix =
        num_rows_original_matrix - num_rows_to_remove;

    Eigen::MatrixXd reduced_quad_vertices(num_rows_new_matrix,
                                          quad_vertices.cols());
    MeshIndex new_row = 0;

    for (MeshIndex row = 0; row < num_rows_original_matrix; ++row) {
        if (quad_edges_to_remove_set.find(row) ==
            quad_edges_to_remove_set.end()) {
            reduced_quad_vertices.row(new_row) = quad_vertices.row(row);
            ++new_row;
        }
    }

    // Remove the contact edges
    quad_edges.erase(quad_edges.begin() + quad_edges_to_remove[0],
                     quad_edges.begin() +
                         quad_edges_to_remove[quad_edges_to_remove.size() - 1] +
                         1);

    auto last_idx_edge_tri = static_cast<MeshIndex>(tri_edges.size());

    // Update the faces edges of the triangle
    for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
        for (MeshIndex j = 0; j < dir1_size - 2; ++j) {
            // Faces in contact with the pure triangle
            if (j == 0) {
                quad_faces_edges[i * (dir1_size - 2)][0] += last_idx_edge_tri;
                quad_faces_edges[i * (dir1_size - 2)][1] +=
                    (last_idx_edge_tri - (dir2_size - 1));
                quad_faces_edges[i * (dir1_size - 2)][2] += last_idx_edge_tri;
                quad_faces_edges[i * (dir1_size - 2)][3] = tri_edges_border[i];
                // Other faces
            } else {
                quad_faces_edges[i * (dir1_size - 2) + j][0] +=
                    last_idx_edge_tri;
                quad_faces_edges[i * (dir1_size - 2) + j][1] +=
                    (last_idx_edge_tri - (dir2_size - 1));
                quad_faces_edges[i * (dir1_size - 2) + j][2] +=
                    last_idx_edge_tri;
                quad_faces_edges[i * (dir1_size - 2) + j][3] +=
                    (last_idx_edge_tri - (dir2_size - 1));
            }
        }
    }

    // Remove the contact edges from the pure triangle perimeter
    if (tri_perimeter_edges.size() > 2) {
        // Temporarily store the last element
        auto last_element = tri_perimeter_edges(tri_perimeter_edges.size() - 1);
        // Resize the matrix to keep only two elements
        tri_perimeter_edges.conservativeResize(2);
        // Set the last element as we've stored it before resizing
        tri_perimeter_edges(1) = last_element;
    }
    // Remove the contact edges from the quadrilateral perimeter
    if (quad_perimeter_edges.size() > dir2_size - 1) {
        // Calculate the new size
        const Index new_size = quad_perimeter_edges.size() - (dir2_size - 1);
        // Resize the matrix to exclude the last 'dir2_size - 1' elements
        quad_perimeter_edges.conservativeResize(new_size);
    }

    // Update the perimeter edges of the quadrilateral
    for (MeshIndex i = 0; i < dir1_size - 2; ++i) {
        quad_perimeter_edges[i] += last_idx_edge_tri;
        quad_perimeter_edges[quad_perimeter_edges.size() - 1 - i] +=
            last_idx_edge_tri;
    }
    for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
        quad_perimeter_edges[dir1_size - 2 + i] +=
            (last_idx_edge_tri - (dir2_size - 1));
    }

    auto last_idx_vert_tri = static_cast<MeshIndex>(tri_vertices.rows());
    // Update the edges' horizontal vertices
    for (MeshIndex i = 0; i < dir1_size - 2; ++i) {
        for (MeshIndex j = 0; j < dir2_size; ++j) {
            if (i == 0) {
                if (j < dir2_size - 1) {
                    quad_edges[(dir1_size - 2) * j][0] =
                        tri_edges[tri_edges_border[j]][0];
                } else {
                    quad_edges[(dir1_size - 2) * j][0] =
                        tri_edges[tri_edges_border[j - 1]].tail<1>()(0);
                }

                for (Index k = 1;
                     k < static_cast<Index>(
                             quad_edges[i + (dir1_size - 2) * j].size());
                     ++k) {
                    quad_edges[(dir1_size - 2) * j][k] +=
                        last_idx_vert_tri - 1 - j;
                }
            } else {
                for (Index k = 0;
                     k < static_cast<Index>(
                             quad_edges[i + (dir1_size - 2) * j].size());
                     ++k) {
                    quad_edges[i + (dir1_size - 2) * j][k] +=
                        last_idx_vert_tri - 1 - j;
                }
            }
        }
    }
    // Update the edges' vertical vertices
    const auto quad_removed_vert =
        static_cast<MeshIndex>(quad_edges_to_remove_set.size());
    for (MeshIndex i = 0; i < dir1_size - 2; ++i) {
        for (MeshIndex j = 0; j < dir2_size - 1; ++j) {
            quad_edges[dir2_size * (dir1_size - 2) + i * (dir2_size - 1) + j]
                      [0] = quad_edges[i + (dir1_size - 2) * j].tail<1>()(0);
            quad_edges[dir2_size * (dir1_size - 2) + i * (dir2_size - 1) + j]
                .tail<1>()(0) =
                quad_edges[i + (dir1_size - 2) * (j + 1)].tail<1>()(0);
            for (Index k = 1;
                 k < static_cast<Index>(quad_edges[dir2_size * (dir1_size - 2) +
                                                   i * (dir2_size - 1) + j]
                                            .size()) -
                         1;
                 ++k) {
                quad_edges[dir2_size * (dir1_size - 2) + i * (dir2_size - 1) +
                           j][k] += last_idx_vert_tri - quad_removed_vert;
            }
        }
    }

    // Merge vertices
    VerticesList points(tri_vertices.rows() + reduced_quad_vertices.rows(), 3);
    points(Eigen::seq(0, tri_vertices.rows() - 1), Eigen::all) = tri_vertices;
    points(Eigen::seq(tri_vertices.rows(), points.rows() - 1), Eigen::all) =
        reduced_quad_vertices;

    // Merge edges
    EdgesList edges;
    edges.reserve(tri_edges.size() + quad_edges.size());
    edges.insert(edges.end(), tri_edges.begin(), tri_edges.end());
    edges.insert(edges.end(), quad_edges.begin(), quad_edges.end());

    // Merge face edges
    EdgesList faces_edges;
    faces_edges.reserve(tri_faces_edges.size() + quad_faces_edges.size());
    faces_edges.insert(faces_edges.end(), tri_faces_edges.begin(),
                       tri_faces_edges.end());
    faces_edges.insert(faces_edges.end(), quad_faces_edges.begin(),
                       quad_faces_edges.end());

    // Merge perimeter edges
    EdgesIdsList perimeter_edges(tri_perimeter_edges.size() +
                                 quad_perimeter_edges.size());

    // Insert elements
    Index current_index = 0;

    perimeter_edges(current_index) = tri_perimeter_edges(0);
    current_index += 1;

    // Inserting all elements from quad_perimeter_edges
    perimeter_edges.block(current_index, 0, quad_perimeter_edges.size(), 1) =
        quad_perimeter_edges;
    current_index += quad_perimeter_edges.size();

    // Inserting the rest of tri_perimeter_edges
    perimeter_edges.block(current_index, 0, tri_perimeter_edges.size() - 1, 1) =
        tri_perimeter_edges.block(1, 0, tri_perimeter_edges.size() - 1, 1);

    // Merge number of faces
    auto tri_n_faces = tri_mesh.get_number_of_faces();
    auto quad_n_faces = quad_mesh.get_number_of_faces();
    auto num_faces = tri_n_faces + quad_n_faces;

    TriMesh trimesh;

    trimesh.set_vertices(std::move(points));
    trimesh.set_edges(std::move(edges));
    trimesh.set_perimeter_edges(std::move(perimeter_edges));
    trimesh.set_faces_edges(std::move(faces_edges));
    trimesh.set_number_of_faces(num_faces);

    // cdt_trimesher(trimesh);

    return trimesh;
}

// NOLINTEND(readability-function-cognitive-complexity)

// TODO: Remove this when an update of MSVC is available
#if defined(_MSC_VER)
#pragma optimize("", off)
#endif
// Create the mesh of a disc
/**
 * @brief Create a 2d mesh of a disc
 *
 * @param center The center of the disc
 * @param outer_point The point that defines the outer perimeter (could be
 * changed in the future)
 * @param dir1_mesh_normalized Normalized position of the points in the radial
 * direction. The values can be between 0 and 1, both included (0 and 1 are the
 * center and the outer perimeter). The first value can be greater than 0. The
 * values have to be sorted in ascending order.
 * @param dir2_mesh_normalized Normalized position of the points in the angular
 * direction. The values can be between 0 and 1, both included but the first
 * value has to be 0. The values have to be sorted in ascending order.
 * @param max_distance_points_dir1 Maximum absolute distance between points in
 * the radial direction.
 * @param max_distance_points_dir2 Maximum absolute distance along the arc
 * between points in the angular direction.
 * @return TriMesh
 */

// TODO: Refactor this function to reduce its complexity. clang-tidy warning.
// NOLINTBEGIN(readability-function-cognitive-complexity)
inline TriMesh create_2d_disc_mesh(const Eigen::VectorXd& dir1_mesh_normalized,
                                   const Eigen::VectorXd& dir2_mesh_normalized,
                                   const Point2D& center,       // Center of the
                                                                // disc
                                   const Point2D& outer_point,  // Point in the
                                                                // outer
                                                                // perimeter
                                   double max_distance_points_dir1,
                                   double max_distance_points_dir2) {
    using std::numbers::pi;

    // Assert preconditions (only in debug mode)
    assert(dir1_mesh_normalized.size() >= 2);
    assert(dir2_mesh_normalized.size() >= 2);
    assert(2 * dir1_mesh_normalized.size() * dir2_mesh_normalized.size() <=
           std::numeric_limits<MeshIndex>::max());
    assert(utils::is_sorted(dir1_mesh_normalized));
    assert(utils::is_sorted(dir2_mesh_normalized));

    // Extract the radius and the maximum angle
    const double radius = (outer_point - center).norm();

    // Set the conditions for the mesh
    const bool inner_radius = (dir1_mesh_normalized[0] != 0.0);
    const bool full_circle =
        (dir2_mesh_normalized[dir2_mesh_normalized.size() - 1] == 1.0);

    // Check that the mesh is normalized in dir1
    if (dir1_mesh_normalized[dir1_mesh_normalized.size() - 1] > 1.0) {
        std::cout << "dir1_mesh is not normalized, but last is value: "
                  << dir1_mesh_normalized[dir1_mesh_normalized.size() - 1]
                  << "\n";
        throw std::runtime_error("dir1_mesh is not normalized.");
    }

    // Check that the mesh is normalized in dir2
    if (dir2_mesh_normalized[dir2_mesh_normalized.size() - 1] > 1.0) {
        std::cout << "dir2_mesh is not normalized, but last is value: "
                  << dir2_mesh_normalized[dir2_mesh_normalized.size() - 1]
                  << "\n";
        throw std::runtime_error("dir2_mesh is not normalized.");
    }

    // Check that the mesh in dir2 starts in 0.0
    double theta_0 = 0.0;
    if (dir2_mesh_normalized[0] != 0.0) {
        theta_0 = dir2_mesh_normalized[0] * 2 * pi;
    }

    // 1. Determine the number of points to reserve space
    const auto dir1_size = static_cast<MeshIndex>(dir1_mesh_normalized.size());
    const auto dir2_size = static_cast<MeshIndex>(dir2_mesh_normalized.size());

    MeshIndex num_points = 0;

    // Dir1 is the radial direction. All the radius will have the same number of
    // points.
    std::vector<MeshIndex> additional_points_dir1(dir1_size - 1, 0);

    // Dir2 is the angular direction. Because the length of the circumference
    // changes for each of the divisions in dir1, a different number of points
    // might be needed

    // If the disc includes the center, the count start in 1 to skip it
    MeshIndex dir1_start = 0;
    if (!inner_radius) {
        dir1_start = 1;
    }
    std::vector<std::vector<MeshIndex>> additional_points_dir2(
        dir1_size - dir1_start, std::vector<MeshIndex>(dir2_size - 1));

    // Now check if additional points are needed based on
    // max_distance_points If max_distance is negative or zero, then we
    // don't add additional points

    MeshIndex num_points_row_dir1 = dir1_size;
    // MeshIndex num_points_dir2 = dir2_size*(dir1_size-1);
    MeshIndex num_points_dir2 = 0;

    // Points in direction 1, the same for each radius
    if (max_distance_points_dir1 > LENGTH_TOL) {
        for (MeshIndex i = 0; i < dir1_size - 1; ++i) {
            const double distance =
                (dir1_mesh_normalized[i + 1] - dir1_mesh_normalized[i]) *
                radius;
            const auto num_additional_points = static_cast<MeshIndex>(
                std::floor(distance / max_distance_points_dir1));
            num_points_row_dir1 += num_additional_points;
            additional_points_dir1[i] = num_additional_points;
        }
    }

    // Points in direction 2, different for each radius and angle
    if (max_distance_points_dir2 > LENGTH_TOL) {
        for (MeshIndex i = 0; i < dir2_size - 1; ++i) {
            const double angle_i =
                (dir2_mesh_normalized[i + 1] - dir2_mesh_normalized[i]) * 2 *
                pi;
            for (MeshIndex j = dir1_start; j < dir1_size; ++j) {
                const double distance =
                    angle_i * dir1_mesh_normalized[j] * radius;
                const auto num_additional_points = static_cast<MeshIndex>(
                    std::floor(distance / max_distance_points_dir2));
                num_points_dir2 += num_additional_points;
                additional_points_dir2[j - dir1_start][i] =
                    num_additional_points;
            }
        }
    }

    // Determine if there are interior points. For the disc
    // only the dir1 points generate interior points and only
    // they are the same for a given radius
    // const std::vector<std::vector<MeshIndex>> interior_points_dir2(
    //     dir1_size - 1, std::vector<MeshIndex>(dir2_size - 1));
    MeshIndex num_interior_points = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        if (additional_points_dir1[i_dir1] > 0) {
            auto add_pi1 = additional_points_dir1[i_dir1];
            for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
                MeshIndex num_interior_points_i = 0;
                if (!inner_radius) {
                    num_interior_points_i =
                        add_pi1 * additional_points_dir2[i_dir1][j_dir2];
                } else {
                    num_interior_points_i =
                        add_pi1 * additional_points_dir2[i_dir1 + 1][j_dir2];
                }
                num_interior_points += num_interior_points_i;
                // interior_points_dir2[i_dir1][j_dir2] = num_interior_points_i;
            }
        }
    }

    // Calculate the total number of points
    MeshIndex num_points_dir1 = num_points_row_dir1;
    if (!inner_radius) {
        if (full_circle) {
            num_points_dir1 = (num_points_dir1 - 1) * (dir2_size - 1) + 1;
        } else {
            num_points_dir1 = (num_points_dir1 - 1) * dir2_size + 1;
        }
    } else {
        if (full_circle) {
            num_points_dir1 = num_points_dir1 * (dir2_size - 1);
        } else {
            num_points_dir1 = num_points_dir1 * dir2_size;
        }
    }
    num_points = num_points_dir1 + num_points_dir2 + num_interior_points;
    VerticesList points(num_points, 3);
    std::vector<double> full_dir1_mesh(num_points_row_dir1);
    std::vector<double> full_dir2_mesh(num_points_dir2);

    // Fill the full mesh arrays with the original and additional mesh
    // points
    // Dir 1
    MeshIndex p_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        const double distance =
            (dir1_mesh_normalized[i_dir1 + 1] - dir1_mesh_normalized[i_dir1]) *
            radius / (additional_points_dir1[i_dir1] + 1);
        if ((!inner_radius) && (i_dir1 == 0)) {
            full_dir1_mesh[0] = 0.0;
            ++p_idx;
            for (MeshIndex i_add1 = 1; i_add1 <= additional_points_dir1[i_dir1];
                 ++i_add1) {
                full_dir1_mesh[p_idx] =
                    dir1_mesh_normalized[i_dir1] * radius + i_add1 * distance;
                ++p_idx;
            }
        } else {
            for (MeshIndex i_add1 = 0; i_add1 <= additional_points_dir1[i_dir1];
                 ++i_add1) {
                full_dir1_mesh[p_idx] =
                    dir1_mesh_normalized[i_dir1] * radius + i_add1 * distance;
                ++p_idx;
            }
        }
    }
    full_dir1_mesh[full_dir1_mesh.size() - 1] =
        dir1_mesh_normalized[dir1_size - 1] * radius;

    // Dir 2
    p_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - dir1_start; ++i_dir1) {
        // Angle of the dir2 division
        for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
            const double angle_i = (dir2_mesh_normalized[j_dir2 + 1] -
                                    dir2_mesh_normalized[j_dir2]) *
                                   2 * pi;

            // Angle between points of the dir2 division
            const double distance =
                angle_i / (additional_points_dir2[i_dir1][j_dir2] + 1);
            for (MeshIndex i_add1 = 1;
                 i_add1 <= additional_points_dir2[i_dir1][j_dir2]; ++i_add1) {
                full_dir2_mesh[p_idx] =
                    dir2_mesh_normalized[j_dir2] * 2 * pi + i_add1 * distance;
                ++p_idx;
            }
        }
    }

    // Fill the points array in dir1
    p_idx = 0;
    if (!inner_radius) {
        points(0, 0) = center.x();
        points(0, 1) = center.y();
        points(0, 2) = 0.0;
        ++p_idx;
    }
    MeshIndex dir2_stop = dir2_size;
    if (full_circle) {
        dir2_stop = dir2_size - 1;
    }

    const auto full_dir1_mesh_size = static_cast<Index>(full_dir1_mesh.size());

    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_stop; ++i_dir2) {
        auto angle_i = dir2_mesh_normalized[i_dir2] * 2 * pi;
        // If i_dir1 is declared MeshIndex, the loop does not work on release
        for (MeshIndex i_dir1 = dir1_start; i_dir1 < full_dir1_mesh_size;
             ++i_dir1) {
            points(p_idx, 0) = center.x() + round(full_dir1_mesh[i_dir1] *
                                                  cos(angle_i) / LENGTH_TOL) *
                                                LENGTH_TOL;
            points(p_idx, 1) = center.y() + round(full_dir1_mesh[i_dir1] *
                                                  sin(angle_i) / LENGTH_TOL) *
                                                LENGTH_TOL;
            points(p_idx, 2) = 0.0;
            ++p_idx;
        }
    }

    // Fill the points array in dir2
    MeshIndex dir2_idx = 0;
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - dir1_start; ++i_dir1) {
        for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
            for (MeshIndex i_add2 = 0;
                 i_add2 < additional_points_dir2[i_dir1][i_dir2]; ++i_add2) {
                auto angle_i = full_dir2_mesh[dir2_idx];
                points(p_idx, 0) =
                    center.x() +
                    round(dir1_mesh_normalized[i_dir1 + dir1_start] * radius *
                          cos(angle_i) / LENGTH_TOL) *
                        LENGTH_TOL;
                points(p_idx, 1) =
                    center.y() +
                    round(dir1_mesh_normalized[i_dir1 + dir1_start] * radius *
                          sin(angle_i) / LENGTH_TOL) *
                        LENGTH_TOL;
                points(p_idx, 2) = 0.0;
                ++p_idx;
                ++dir2_idx;
            }
        }
    }

    // Fill the interior points
    MeshIndex extra_start = 0;
    if (inner_radius) {
        extra_start = 1;
    }
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        if (additional_points_dir1[i_dir1] > 0) {
            auto add_pi1 = additional_points_dir1[i_dir1];
            for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
                for (MeshIndex p = 1; p <= add_pi1; ++p) {
                    const double rad_i = (dir1_mesh_normalized[i_dir1] +
                                          (dir1_mesh_normalized[i_dir1 + 1] -
                                           dir1_mesh_normalized[i_dir1]) *
                                              p / (1 + add_pi1)) *
                                         radius;

                    for (MeshIndex a = 1;
                         a <=
                         additional_points_dir2[i_dir1 + extra_start][j_dir2];
                         ++a) {
                        auto angle_a =
                            ((dir2_mesh_normalized[j_dir2 + 1] -
                              dir2_mesh_normalized[j_dir2]) /
                                 (1 +
                                  additional_points_dir2[i_dir1 + extra_start]
                                                        [j_dir2]) *
                                 a +
                             dir2_mesh_normalized[j_dir2]) *
                            2 * pi;
                        points(p_idx, 0) =
                            center.x() +
                            round(rad_i * cos(angle_a - theta_0) / LENGTH_TOL) *
                                LENGTH_TOL;
                        points(p_idx, 1) =
                            center.y() +
                            round(rad_i * sin(angle_a - theta_0) / LENGTH_TOL) *
                                LENGTH_TOL;
                        points(p_idx, 2) = 0.0;
                        ++p_idx;
                    }
                }
                // const MeshIndex num_interior_points_i =
                //     add_pi1 *
                //     additional_points_dir2[i_dir1 + extra_start][j_dir2];
                // num_interior_points += num_interior_points_i;
                // interior_points_dir2[i_dir1][j_dir2] = num_interior_points_i;
            }
        }
    }

    // 3. Create the edges

    // Reserve space for the edges
    MeshIndex edges_size =
        (dir1_size - 1) * (dir2_size) + (dir1_size) * (dir2_size - 1);
    if (!inner_radius) {
        edges_size -= dir2_size - 1;
    }
    if (full_circle) {
        edges_size -= dir1_size - 1;
    }
    EdgesList edges(edges_size);

    // Fill the edges
    // First the edges in direction 1
    MeshIndex e_idx = 0;
    p_idx = 0;
    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_stop; ++i_dir2) {
        for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
            const MeshIndex num_edge_points =
                additional_points_dir1[i_dir1] + 2;

            Edges edge(num_edge_points);
            if (!inner_radius && i_dir1 == 0) {
                edge[0] = 0;
            } else {
                edge[0] = p_idx;
            }
            for (MeshIndex i_add1 = 0; i_add1 <= additional_points_dir1[i_dir1];
                 ++i_add1) {
                edge[i_add1 + 1] = p_idx + i_add1 + 1;
            }
            edges[e_idx] = edge;
            e_idx++;
            p_idx += num_edge_points - 1;
        }
        if (inner_radius) {
            p_idx += 1;
        }
    }
    if (!inner_radius) {
        p_idx += 1;
    }

    // Then the edges in direction 2
    MeshIndex dir1_idx = 0;
    if (!inner_radius) {
        dir1_idx = 1 + additional_points_dir1[0];
    }
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - dir1_start; ++i_dir1) {
        MeshIndex end_idx = dir1_idx;
        for (MeshIndex j_dir2 = 0; j_dir2 < dir2_size - 1; ++j_dir2) {
            const MeshIndex num_edge_points =
                additional_points_dir2[i_dir1][j_dir2] + 2;
            Edges edge(num_edge_points);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
            edge[0] = end_idx;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
            for (MeshIndex i_add2 = 0;
                 i_add2 < additional_points_dir2[i_dir1][j_dir2]; ++i_add2) {
                edge[i_add2 + 1] = p_idx;
                ++p_idx;
            }
            end_idx += num_points_dir1 / dir2_stop;
            if ((full_circle) && (j_dir2 == dir2_stop - 1)) {
                edge[num_edge_points - 1] = dir1_idx;
            } else {
                edge[num_edge_points - 1] = end_idx;
            }
            edges[e_idx] = edge;
            e_idx++;
        }
        if (i_dir1 < dir1_size - dir1_start - 1) {
            if (!inner_radius) {
                dir1_idx += additional_points_dir1[i_dir1 + 1] + 1;
            } else {
                dir1_idx += additional_points_dir1[i_dir1] + 1;
            }
        }
    }

    // 4. Create the permiter edges. Anti-clockwise order starting from P01
    // -> dir1 edge
    MeshIndex num_perimeter_edges = dir2_size - 1;
    if (inner_radius) {
        num_perimeter_edges *= 2;
    }
    if (!full_circle) {
        num_perimeter_edges += (dir1_size - 1) * 2;
    }
    EdgesIdsList perimeter_edges(num_perimeter_edges);
    MeshIndex per_edge_idx = 0;

    if (!full_circle) {
        for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
            perimeter_edges[per_edge_idx] = i_dir1;
            ++per_edge_idx;
        }
        for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
            perimeter_edges[per_edge_idx] =
                i_dir1 + (dir1_size - 1) * (dir2_size - 1);
            ++per_edge_idx;
        }
    }

    if (inner_radius) {
        for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
            perimeter_edges[per_edge_idx] =
                i_dir2 + dir2_stop * (dir1_size - 1);
            ++per_edge_idx;
        }
    }

    for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
        perimeter_edges[per_edge_idx] =
            i_dir2 + dir2_stop * (dir1_size - 1) +
            (dir2_size - 1) * (dir1_size - dir1_start - 1);
        ++per_edge_idx;
    }

    // 5. Create the faces edges
    const MeshIndex num_faces = (dir1_size - 1) * (dir2_size - 1);
    FaceEdges faces_edges(num_faces);
    MeshIndex face_idx = 0;
    const MeshIndex skip_horizontal_edges = dir2_stop * (dir1_size - 1);
    for (MeshIndex i_dir1 = 0; i_dir1 < dir1_size - 1; ++i_dir1) {
        for (MeshIndex i_dir2 = 0; i_dir2 < dir2_size - 1; ++i_dir2) {
            if (!inner_radius && (i_dir1 == 0)) {
                EdgesIdsList face(3);
                face[0] = i_dir1 + (dir1_size - 1) * i_dir2;
                face[1] = i_dir2 + skip_horizontal_edges;
                face[2] = (full_circle && i_dir2 == dir2_size - 2)
                              ? 0
                              : i_dir1 + (dir1_size - 1) * (i_dir2 + 1);
                faces_edges[face_idx] = face;
            } else {
                EdgesIdsList face(4);
                face[0] = i_dir1 + (dir1_size - 1) * i_dir2;
                face[1] = i_dir2 + (dir2_size - 1) * (i_dir1 - dir1_start + 1) +
                          skip_horizontal_edges;
                face[2] = (full_circle && i_dir2 == dir2_size - 2)
                              ? i_dir1
                              : i_dir1 + (dir1_size - 1) * (i_dir2 + 1);
                face[3] = i_dir2 + (dir2_size - 1) * (i_dir1 - dir1_start) +
                          skip_horizontal_edges;
                faces_edges[face_idx] = face;
            }

            ++face_idx;
        }
    }

    // 6. Create the trimesh

    TriMesh trimesh;

    trimesh.set_vertices(std::move(points));
    trimesh.set_edges(std::move(edges));
    trimesh.set_perimeter_edges(std::move(perimeter_edges));
    trimesh.set_faces_edges(std::move(faces_edges));
    trimesh.set_number_of_faces(num_faces * 2);

    return trimesh;
}
#if defined(_MSC_VER)
#pragma optimize("", on)
#endif
// NOLINTEND(readability-function-cognitive-complexity)

}  // namespace trimesher
}  // namespace pycanha::gmm
