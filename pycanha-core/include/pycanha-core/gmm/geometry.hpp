#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "./id.hpp"
#include "./primitives.hpp"
#include "./thermalmesh.hpp"
#include "./transformations.hpp"
#include "./trimesh.hpp"

namespace pycanha::gmm {

// using PrimitivePtr =
//     std::variant<std::shared_ptr<Triangle>, std::shared_ptr<Rectangle>,
//                  std::shared_ptr<Quadrilateral>, std::shared_ptr<Disc>,
//                  std::shared_ptr<Cylinder>, std::shared_ptr<Sphere>>;

using PrimitivePtr = std::shared_ptr<Primitive>;

using ThermalMeshPtr = std::shared_ptr<ThermalMesh>;
using TransformationPtr = std::shared_ptr<CoordinateTransformation>;

// A list of GeometryItems and/or GeometryGroups
class GeometryItem;
class GeometryMeshedItem;
class GeometryGroup;
class GeometryGroupCutted;
using GeometryPtrList =
    std::vector<std::variant<std::shared_ptr<GeometryMeshedItem>,
                             std::shared_ptr<GeometryGroup>,
                             std::shared_ptr<GeometryGroupCutted>>>;

using GeometryItemPtrList = std::vector<std::shared_ptr<GeometryItem>>;

/**
 * @class Geometry
 * @brief Base class for geometric items.
 *
 * TODO: Make contructors protected?
 */
class Geometry : public UniqueID {
  protected:
    std::string _name;                    /**< The name of the object. */
    TransformationPtr _transformation;    /**< The transformation. */
    std::weak_ptr<GeometryGroup> _parent; /**< The parent GeometryGroup. */

  public:
    /**
     * @brief Default constructor.
     */
    Geometry() : _transformation(std::make_shared<CoordinateTransformation>()) {
        _name = "Geometry_" + std::to_string(get_id());
    }

    /**
     * @brief Constructor with only a name.
     * @param name The name to set.
     */
    explicit Geometry(std::string name)
        : _name(std::move(name)),
          _transformation(std::make_shared<CoordinateTransformation>()) {}

    /**
     * @brief Constructor that sets the name.
     * @param name The name to set.
     * @param transformation The transformation to set.
     */
    explicit Geometry(std::string name, TransformationPtr transformation)
        : _name(std::move(name)), _transformation(std::move(transformation)) {}

    virtual ~Geometry() = default;
    Geometry(const Geometry&) = delete;
    Geometry& operator=(const Geometry&) = delete;
    Geometry(Geometry&&) = delete;
    Geometry& operator=(Geometry&&) = delete;

    /**
     * @brief Gets the name.
     * @return The name.
     */
    [[nodiscard]] std::string get_name() const { return _name; }

    /**
     * @brief Sets the name.
     * @param name The name to set.
     */
    void set_name(std::string name) { _name = std::move(name); }

    /**
     * @brief Gets the transformation.
     * @return The transformation.
     */
    [[nodiscard]] TransformationPtr get_transformation() const {
        return _transformation;
    }

    /**
     * @brief Sets the transformation.
     * @param transformation The transformation to set.
     */
    void set_transformation(TransformationPtr transformation) {
        _transformation = std::move(transformation);
    }

    /**
     * @brief Gets the parent GeometryGroup.
     * @return The parent GeometryGroup.
     */
    [[nodiscard]] std::weak_ptr<GeometryGroup> get_parent() const {
        return _parent;
    }

    /**
     * @brief Sets the parent GeometryGroup.
     * @param parent The parent GeometryGroup to set.
     */
    void set_parent(std::weak_ptr<GeometryGroup> parent) {
        _parent = std::move(parent);
    }
};

/**
 * @brief The base class for all geometry items.
 */
class GeometryItem : public Geometry {
  protected:
    PrimitivePtr _primitive; /**< The primitive. */

  public:
    /**
     * @brief Default constructor.
     */
    GeometryItem() = default;

    /**
     * @brief Constructor that sets the primitive and transformation.
     * @param name The name of the object.
     * @param primitive The primitive to set.
     * @param transformation The transformation to set.
     */
    GeometryItem(std::string name, PrimitivePtr primitive,
                 TransformationPtr transformation)
        : Geometry(std::move(name), std::move(transformation)),
          _primitive(std::move(primitive)) {}

    /**
     * @brief Gets the primitive.
     * @return The primitive.
     */
    [[nodiscard]] PrimitivePtr get_primitive() const { return _primitive; }

    /**
     * @brief Sets the primitive.
     * @param primitive The primitive to set.
     */
    void set_primitive(PrimitivePtr primitive) {
        _primitive = std::move(primitive);
    }
};

/**
 * @brief A geometry item that contains mesh data.
 */
class GeometryMeshedItem : public GeometryItem {
    // Private members
    ThermalMeshPtr _thermal_mesh; /**< The thermal mesh. */
    TriMeshPtr _tri_mesh;         /**< The triangle mesh. */

  public:
    /**
     * @brief Default constructor.
     */
    GeometryMeshedItem() = default;

    /**
     * @brief Constructor that sets the primitive, transformation, thermal mesh,
     * and triangle mesh.
     * @param name The name of the object.
     * @param primitive The primitive to set.
     * @param transformation The transformation to set.
     * @param thermal_mesh The thermal mesh to set.
     */
    GeometryMeshedItem(std::string name, PrimitivePtr primitive,
                       TransformationPtr transformation,
                       ThermalMeshPtr thermal_mesh)
        : GeometryItem(std::move(name), std::move(primitive),
                       std::move(transformation)),
          _thermal_mesh(std::move(thermal_mesh)) {}

    /**
     * @brief Gets the thermal mesh.
     * @return The thermal mesh.
     */
    [[nodiscard]] ThermalMeshPtr get_thermal_mesh() const {
        return _thermal_mesh;
    }

    /**
     * @brief Sets the thermal mesh.
     * @param thermal_mesh The thermal mesh to set.
     */
    void set_thermal_mesh(ThermalMeshPtr thermal_mesh) {
        _thermal_mesh = std::move(thermal_mesh);
    }

    /**
     * @brief Gets the triangle mesh.
     * @return The triangle mesh.
     */
    [[nodiscard]] TriMeshPtr get_tri_mesh() const { return _tri_mesh; }

    /**
     * @brief Sets the triangle mesh.
     * @param tri_mesh The triangle mesh to set.
     */
    void set_tri_mesh(TriMeshPtr tri_mesh) { _tri_mesh = std::move(tri_mesh); }

    /**
     * @brief Creates the triangular mesh for the primitive.
     * @param tolerance The tolerance to use (only used for non-planar
     * surfaces).
     */
    void create_mesh(double tolerance) {
        _tri_mesh = std::make_shared<TriMesh>(
            _primitive->create_mesh(*_thermal_mesh, tolerance));
    }

    /**
     * @brief Triangulate a cutted primitive. This function
     * is called after post-processing the cut in python.
     */
    void triangulate_post_processed_cutted_mesh() {
        // Call the CDT mesher to resolve the triangulation
        trimesher::cdt_trimesher(*_tri_mesh);

        // Reconstruct the face edges
        _primitive->reconstruct_face_edges_2d(_tri_mesh, *_thermal_mesh);

        // Assign face ids
        for (MeshIndex i = 0; i < _tri_mesh->get_triangles().rows(); i++) {
            auto tri = _tri_mesh->get_triangles().row(i);
            const Point2D p0_2d{_tri_mesh->get_vertices()(tri[0], 0),
                                _tri_mesh->get_vertices()(tri[0], 1)};
            const Point2D p1_2d{_tri_mesh->get_vertices()(tri[1], 0),
                                _tri_mesh->get_vertices()(tri[1], 1)};
            const Point2D p2_2d{_tri_mesh->get_vertices()(tri[2], 0),
                                _tri_mesh->get_vertices()(tri[2], 1)};

            const Point2D centroid = (p0_2d + p1_2d + p2_2d) / 3.0;

            // Get the face id
            _tri_mesh->get_face_ids()[i] =
                _primitive->get_faceid_from_uv(*_thermal_mesh, centroid);
        }

        // Transform 2D points to 3D
        for (int i = 0; i < _tri_mesh->get_vertices().rows(); i++) {
            const Point2D point2d{_tri_mesh->get_vertices()(i, 0),
                                  _tri_mesh->get_vertices()(i, 1)};
            _tri_mesh->get_vertices().row(i) =
                _primitive->from_2d_to_3d(point2d);
        }
        // sort the mesh
        _tri_mesh->sort_triangles();
        // compute areas
        _tri_mesh->compute_areas();
    }
};

/**
 * @class GeometryGroup
 * @brief Class representing a group of geometric items. Inherits from Geometry.
 *
 * @note Names or the GeometryItems and GeometryGroups can be repeated. It is
 * not checked by the class, but GeometricalModel checks it.
 */
class GeometryGroup : public Geometry {
  protected:
    std::vector<std::shared_ptr<GeometryMeshedItem>> _geometry_items;
    std::vector<std::shared_ptr<GeometryGroup>> _geometry_groups;
    std::vector<std::shared_ptr<GeometryGroupCutted>> _geometry_groups_cutted;

  public:
    /**
     * @brief Constructor for the GeometryGroup class that accepts only a name.
     *
     * @param name - Name of the geometric group.
     */
    explicit GeometryGroup(std::string name) : Geometry(std::move(name)) {}

    /**
     * @brief Default constructor for the GeometryGroup class.
     */
    GeometryGroup() = default;

    /**
     * @brief Constructor for the GeometryGroup class.
     *
     * @param name - Name of the geometric group.
     * @param geometries - List of GeometryItem and GeometryGroup pointers
     * belonging to the group.
     * @param transformation - Transformation applied to the group.
     */
    GeometryGroup(std::string name, const GeometryPtrList& geometries,
                  TransformationPtr transformation)
        : Geometry(std::move(name), std::move(transformation)) {
        for (const auto& item : geometries) {
            if (std::holds_alternative<std::shared_ptr<GeometryGroup>>(item)) {
                _geometry_groups.push_back(
                    std::get<std::shared_ptr<GeometryGroup>>(item));
            } else if (std::holds_alternative<
                           std::shared_ptr<GeometryGroupCutted>>(item)) {
                _geometry_groups_cutted.push_back(
                    std::get<std::shared_ptr<GeometryGroupCutted>>(item));
            } else {
                _geometry_items.push_back(
                    std::get<std::shared_ptr<GeometryMeshedItem>>(item));
            }
        }
    }

    /**
     * @brief Getter for the geometry items.
     *
     * @return Reference to the geometry items vector.
     */
    std::vector<std::shared_ptr<GeometryMeshedItem>>& get_geometry_items() {
        return _geometry_items;
    }

    /**
     * @brief Setter for the geometry items.
     *
     * @param items New geometry items vector.
     */
    void set_geometry_items(
        const std::vector<std::shared_ptr<GeometryMeshedItem>>& items) {
        _geometry_items = items;
    }
    /**
     * @brief Getter for the geometry groups.
     *
     * @return Reference to the geometry groups vector.
     */
    std::vector<std::shared_ptr<GeometryGroup>>& get_geometry_groups() {
        return _geometry_groups;
    }

    /**
     * @brief Setter for the geometry groups.
     *
     * @param groups New geometry groups vector.
     */
    void set_geometry_groups(
        const std::vector<std::shared_ptr<GeometryGroup>>& groups) {
        _geometry_groups = groups;
    }

    /**
     * @brief Getter for the geometry groups cutted.
     *
     * @return Reference to the geometry groups cutted vector.
     */
    std::vector<std::shared_ptr<GeometryGroupCutted>>&
    get_geometry_groups_cutted() {
        return _geometry_groups_cutted;
    }

    /**
     * @brief Setter for the geometry groups cutted.
     *
     * @param groups New geometry groups cutted vector.
     */
    void set_geometry_groups_cutted(
        const std::vector<std::shared_ptr<GeometryGroupCutted>>& groups) {
        _geometry_groups_cutted = groups;
    }

    /**
     * @brief Adds a geometry item to the geometry items vector.
     *
     * @param item Geometry item to be added.
     */
    void add_geometry_item(std::shared_ptr<GeometryMeshedItem> item) {
        _geometry_items.push_back(std::move(item));
    }

    /**
     * @brief Adds a geometry group to the geometry groups vector.
     *
     * @param group Geometry group to be added.
     */
    void add_geometry_group(std::shared_ptr<GeometryGroup> group) {
        _geometry_groups.push_back(std::move(group));
    }

    /**
     * @brief Removes a geometry item from the geometry items vector.
     *
     * @param item Geometry item to be removed.
     */

    void remove_geometry_item(const std::shared_ptr<GeometryItem>& item) {
        _geometry_items.erase(
            std::remove(_geometry_items.begin(), _geometry_items.end(), item),
            _geometry_items.end());
    }

    /**
     * @brief Removes a geometry group from the geometry groups vector.
     *
     * @param group Geometry group to be removed.
     */

    void remove_geometry_group(const std::shared_ptr<GeometryGroup>& group) {
        _geometry_groups.erase(std::remove(_geometry_groups.begin(),
                                           _geometry_groups.end(), group),
                               _geometry_groups.end());
    }

    /**
     * @brief Adds a cut geometry group to the geometry groups cutted vector.
     *
     * @param group_cutted Geometry group cutted to be added.
     */
    void add_geometry_group_cutted(
        std::shared_ptr<GeometryGroupCutted> group_cutted) {
        _geometry_groups_cutted.push_back(std::move(group_cutted));
    }

    /**
     * @brief Removes a cut geometry group from the geometry groups cutted
     * vector.
     *
     * @param group_cutted Geometry group cutted to be removed.
     */
    void remove_geometry_group_cutted(
        const std::shared_ptr<GeometryGroupCutted>& group_cutted) {
        _geometry_groups_cutted.erase(
            std::remove(_geometry_groups_cutted.begin(),
                        _geometry_groups_cutted.end(), group_cutted),
            _geometry_groups_cutted.end());
    }

    /**
     * @brief Creates the triangular meshes for all the geometry items in the
     * group.
     * @param tolerance The tolerance to use (only used for non-planar
     * surfaces).
     */
    void create_meshes(double tolerance);
    /**
     * @brief Recursively iterates over all GeometryMeshedItem in the
     * GeometryGroup. Skips recursive GeometryGroupCutted.
     *
     * @param global_transformation Chain of transformations applied to the
     * group
     * @param func Function to execute for each GeometryMeshedItem.
     */
    void iterate_geometry_meshed_items(
        const CoordinateTransformation& global_transformation,
        const std::function<void(const CoordinateTransformation&,
                                 const std::shared_ptr<GeometryMeshedItem>&)>&
            func);

    /**
     * @brief Recursively iterates over all GeometryMeshedItem in the
     * GeometryGroup. Does NOT skip GeometryGroupCutted.
     *
     * @param global_transformation Chain of transformations applied to the
     * group
     * @param func Function to execute for each GeometryMeshedItem.
     */
    void iterate_all_geometry_meshed_items(
        const CoordinateTransformation& global_transformation,
        const std::function<void(const CoordinateTransformation&,
                                 const std::shared_ptr<GeometryMeshedItem>&)>&
            func);

    /**
     * @brief Recursively iterates over all GeometryItem in the
     * GeometryGroupCutted that are used . Does NOT skip GeometryGroupCutted.
     *
     * @param global_transformation Chain of transformations applied to the
     * group
     * @param func Function to execute for each GeometryMeshedItem.
     */
    void iterate_all_cutting_geometries(
        const CoordinateTransformation& global_transformation,
        const std::function<void(const CoordinateTransformation&,
                                 const std::shared_ptr<GeometryItem>&)>& func);

    /**
     * @brief Recursively iterates over all geometry below a
     * GeometryGroupCutted. Call this function only from top level
     * GeometryGroupCutted objects. After returning, two vectors are filled
     * The vector cutted_geometry_meshed_items contains a transformed copy
     * of the GeometryMeshedItem that needs to be cutted
     *
     * For each GeometryMeshedItem, a vector in cutting_primitives contains
     * a copy of the transformed primitive that cuts that GeometryMeshedItem.
     *
     * @param global_transformation Chain of transformations applied to the
     * group
     * @param cutted_geometry_meshed_items Vector of GeometryMeshedItem that
     * need to be cutted.
     * @param cutting_primitives Vector of vectors of primitives that cut the
     * GeometryMeshedItem.
     */
    void iterate_create_cut_groups(
        const CoordinateTransformation& global_transformation,
        std::vector<std::shared_ptr<GeometryMeshedItem>>&
            cutted_geometry_meshed_items,
        std::vector<std::vector<std::shared_ptr<Primitive>>>&
            cutting_primitives);
};

/**
 * @class GeometryGroupCutted
 * @brief Class representing a cutted geometry.
 *
 */
class GeometryGroupCutted : public GeometryGroup {
    std::vector<std::shared_ptr<GeometryItem>> _cutting_geometry_items;

    // Holds the resultant cutted mesh.
    // This class owns this objects, so no need to use smart pointers.
    std::vector<std::shared_ptr<GeometryMeshedItem>>
        _cutted_geometry_meshed_items;
    std::vector<std::vector<std::shared_ptr<Primitive>>> _cutting_primitives;

  public:
    /**
     * @brief Constructor for the GeometryGroupCutted class.
     *
     * @param name - Unique name of the geometric group.
     * @param geometries - List of GeometryItem and GeometryGroupCutted pointers
     * belonging to the group.
     * @param transformation - Transformation applied to the group.
     */

    GeometryGroupCutted(std::string name, const GeometryPtrList& geometries,
                        GeometryItemPtrList cutting_geometry_items,
                        TransformationPtr transformation)
        : GeometryGroup(std::move(name), geometries, std::move(transformation)),
          _cutting_geometry_items(std::move(cutting_geometry_items)) {}
    /**
     * @brief Getter for the cutting geometry items.
     *
     * @return Reference to the cutting geometry items vector.
     */
    std::vector<std::shared_ptr<GeometryItem>>& get_cutting_geometry_items() {
        return _cutting_geometry_items;
    }

    /**
     * @brief Setter for the cutting geometry items.
     *
     * @param items New cutting geometry items vector.
     */
    void set_cutting_geometry_items(
        const std::vector<std::shared_ptr<GeometryItem>>& items) {
        _cutting_geometry_items = items;
    }

    /**
     * @brief Adds a geometry item to the cutting geometry items vector.
     *
     * @param item Geometry item to be added.
     */
    void add_cutting_geometry_item(std::shared_ptr<GeometryItem> item) {
        _cutting_geometry_items.push_back(std::move(item));
    }

    /**
     * @brief Removes a geometry item from the cutting geometry items
     * vector.
     *
     * @param item Geometry item to be removed.
     */
    void remove_cutting_geometry_item(
        const std::shared_ptr<GeometryItem>& item) {
        _cutting_geometry_items.erase(
            std::remove(_cutting_geometry_items.begin(),
                        _cutting_geometry_items.end(), item),
            _cutting_geometry_items.end());
    }

    /**
     * @brief Getter for the cutted geometry meshed items.
     *
     * @return Reference to the cutted geometry meshed items vector.
     */
    std::vector<std::shared_ptr<GeometryMeshedItem>>&
    get_cutted_geometry_meshed_items() {
        return _cutted_geometry_meshed_items;
    }

    /**
     * @brief Setter for the cutted geometry meshed items.
     *
     * @param items New cutted geometry meshed items vector.
     */
    void set_cutted_geometry_meshed_items(
        const std::vector<std::shared_ptr<GeometryMeshedItem>>& items) {
        _cutted_geometry_meshed_items = items;
    }

    /**
     * @brief Clears the resultant cutting meshes.
     */
    void clear_cutted_mesh() {
        _cutted_geometry_meshed_items.clear();
        _cutting_primitives.clear();
    }

    /**
     * @brief Creates the cutted mesh.
     * @param tol - Tolerance used to create the mesh.
     */
    void create_cutted_mesh(double tol) {
        std::cout << "Creating cutted mesh (experimental)" << tol << '\n';

        // TODO:
        // Reset all the cutting information vectors

        //  Iterate over all geometry items in and below this group
        //  For each item, create a new geometry item with an identity
        //  transformation and the primitive transformed accordingly.
        //  Then, create the mesh for the new geometry item.
        // Iterate over all cutting primitives (not only in this group, also in
        // the groups below)
        // For each primitive, copy and transform the primitive accordingly.
        // Store the new primitive in a vector.
        clear_cutted_mesh();

        iterate_create_cut_groups(CoordinateTransformation(),
                                  _cutted_geometry_meshed_items,
                                  _cutting_primitives);

        // Iterate over the cutted geometry and crete the mesh
        for (auto& item : _cutted_geometry_meshed_items) {
            item->create_mesh(tol);
        }

        // For each geometry item, call python to perform the cut.
        // Cut steps:
        // 1. Calculate the intersection between the primitive in
        // the geometry item and all of the cutting primitives.
        // 2. Refine, check and transform the intersections into
        // 2D coordinates of the primitive in the geometry item.
        // 3. Solve the 2D intersections with the perimeter and
        // with the
    }
};
// NOLINTBEGIN(misc-no-recursion)
inline void GeometryGroup::create_meshes(double tolerance) {
    for (auto& item : _geometry_items) {
        item->create_mesh(tolerance);
    }
    // TODO: This is recursive! Problems with deep groups?
    for (auto& group : _geometry_groups) {
        group->create_meshes(tolerance);
    }

    for (auto& group_cutted : _geometry_groups_cutted) {
        group_cutted->create_cutted_mesh(tolerance);
    }
}

inline void GeometryGroup::iterate_all_geometry_meshed_items(
    const CoordinateTransformation& global_transformation,
    const std::function<void(const CoordinateTransformation&,
                             const std::shared_ptr<GeometryMeshedItem>&)>&
        func) {
    for (const auto& item : _geometry_items) {
        func(global_transformation, item);
    }

    for (const auto& child_group : _geometry_groups) {
        child_group->iterate_all_geometry_meshed_items(
            global_transformation.chain(*(child_group->get_transformation())),
            func);
    }

    for (const auto& child_group_cutted : _geometry_groups_cutted) {
        child_group_cutted->iterate_all_geometry_meshed_items(
            global_transformation.chain(
                *(child_group_cutted->get_transformation())),
            func);
    }
}

inline void GeometryGroup::iterate_geometry_meshed_items(
    const CoordinateTransformation& global_transformation,
    const std::function<void(const CoordinateTransformation&,
                             const std::shared_ptr<GeometryMeshedItem>&)>&
        func) {
    for (const auto& item : _geometry_items) {
        func(global_transformation, item);
    }

    for (const auto& child_group : _geometry_groups) {
        child_group->iterate_geometry_meshed_items(
            global_transformation.chain(*(child_group->get_transformation())),
            func);
    }
    for (const auto& child_group_cutted : _geometry_groups_cutted) {
        for (const auto& meshed_item :
             child_group_cutted->get_cutted_geometry_meshed_items()) {
            func(global_transformation, meshed_item);
        }
    }
}

inline void GeometryGroup::iterate_create_cut_groups(
    const CoordinateTransformation& global_transformation,
    std::vector<std::shared_ptr<GeometryMeshedItem>>&
        cutted_geometry_meshed_items,
    std::vector<std::vector<std::shared_ptr<Primitive>>>& cutting_primitives) {
    auto index_geometry_below_this_group_starts =
        cutted_geometry_meshed_items.size();

    for (const auto& child_group : _geometry_groups) {
        child_group->iterate_create_cut_groups(
            global_transformation.chain(*(child_group->get_transformation())),
            cutted_geometry_meshed_items, cutting_primitives);
    }

    for (const auto& child_group_cutted : _geometry_groups_cutted) {
        child_group_cutted->iterate_create_cut_groups(
            global_transformation.chain(
                *(child_group_cutted->get_transformation())),
            cutted_geometry_meshed_items, cutting_primitives);
    }

    for (const auto& item : _geometry_items) {
        // Creates new GeometryMeshedItem
        const std::string new_name = item->get_name() + "_cutted";
        auto current_coord_transf =
            global_transformation.chain(*(item->get_transformation()));
        const PrimitivePtr new_primitive =
            item->get_primitive()->transform(current_coord_transf);
        const TransformationPtr new_coord_transf =
            std::make_shared<CoordinateTransformation>();
        auto th_mesh = item->get_thermal_mesh();

        auto cutted_item = std::make_shared<GeometryMeshedItem>(
            new_name, new_primitive, new_coord_transf, th_mesh);

        // Add the new GeometryMeshedItem to the vector
        cutted_geometry_meshed_items.push_back(cutted_item);

        // This adds an empty vector<std::shared_ptr<Primitive>> directly.
        cutting_primitives.emplace_back();
    }

    std::vector<std::shared_ptr<Primitive>> this_group_cutting_primitives;
    // Check if I'm actually a GeometryGroupCutted by trying to dynamic casting
    // *this to GeometryGroupCutted
    if (auto* group_cutted = dynamic_cast<GeometryGroupCutted*>(this)) {
        // If I'm a GeometryGroupCutted, iterate over all cutting primitives
        for (const auto& item : group_cutted->get_cutting_geometry_items()) {
            auto current_coord_transf =
                global_transformation.chain(*(item->get_transformation()));

            auto cutting_primitive =
                item->get_primitive()->transform(current_coord_transf);
            this_group_cutting_primitives.push_back(cutting_primitive);
        }

        // Insert the cutting primitives to all the items below and in this
        // group
        for (auto i = index_geometry_below_this_group_starts;
             i < cutting_primitives.size(); ++i) {
            cutting_primitives[i].insert(cutting_primitives[i].end(),
                                         this_group_cutting_primitives.begin(),
                                         this_group_cutting_primitives.end());
        }
    }
}

// NOLINTEND(misc-no-recursion)

}  // namespace pycanha::gmm
