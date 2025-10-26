#pragma once
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pycanha-core/gmm/geometry.hpp"
#include "pycanha-core/gmm/id.hpp"
#include "pycanha-core/gmm/transformations.hpp"

namespace pycanha::gmm {

/**
 * @class GeometryModel
 * @brief Class representing a geometrical mathematical model.
 */
class GeometryModel : public UniqueID,
                      public std::enable_shared_from_this<GeometryModel> {
    // Names should be unique, so we need something to track them
    std::unordered_set<std::string> _names;

    // A Geometry should also be unique within the Model.
    // Geometries can be tracked with their id. We can them map the id to other
    // useful information.

    // Map id - name
    std::unordered_map<GeometryIdType, std::string> _geometry_id_to_name;

    // Geometry
    std::unordered_set<std::shared_ptr<Geometry>> _geometries;

    // Root goemetry
    std::shared_ptr<GeometryGroup> _root_geometry_group;

    // For tracking dependencies

    // Track in which GeometryItem the Primitive is used
    std::unordered_map<GeometryIdType, std::vector<GeometryIdType>>
        _geometry_primitives_dependencies;

    // Root Trimesh
    TriMeshModel _trimesh_model;

  public:
    /**
     * @brief Default constructor.
     */
    GeometryModel() : _root_geometry_group(std::make_shared<GeometryGroup>()) {}

    /**
     * @brief Default constructor.
     */
    explicit GeometryModel(const std::string& name)
        : _root_geometry_group(std::make_shared<GeometryGroup>(name)) {}

    /**
     * @brief Method to create a new GeometryMeshedItem.
     *
     * @param name - Unique name of the geometric item.
     * @param primitive - Geometric primitive of the item.
     * @param transformation - Transformation applied to the item.
     * @param thermal_mesh - Thermal mesh of the item.
     *
     * @return Shared pointer to the created GeometryMeshedItem.
     */
    std::shared_ptr<GeometryMeshedItem> create_geometry_item(
        const std::string& name, const PrimitivePtr& primitive,
        const TransformationPtr& transformation,
        const ThermalMeshPtr& thermal_mesh) {
        auto item = std::make_shared<GeometryMeshedItem>(
            name, primitive, transformation, thermal_mesh);

        // Try to add the GeometryItem to the root GeometryGroup
        // Exception will be thrown if not possible (e.g. if the name already
        // exists)
        add_configure_geometry_item(item);
        return item;
    }

    /**
     * @brief Method to create a new GeometryGroup.
     *
     * @param name - Unique name of the geometric group.
     * @param geometries - Geometric items belonging to the group.
     * @param cutting_geometry_items - Cutting_geometry_items of the group.
     * @param transformation - Transformation applied to the group.
     *
     * @return Shared pointer to the created GeometryGroup.
     */
    std::shared_ptr<GeometryGroup> create_geometry_group(
        const std::string& name, const GeometryPtrList& geometries,
        const TransformationPtr& transformation) {
        auto group =
            std::make_shared<GeometryGroup>(name, geometries, transformation);

        add_configure_geometry_group(group);

        return group;
    }

    /**
     * @brief Method to create a new GeometryGroup.
     *
     * @param name - Unique name of the geometric group.
     * @param geometries - Geometric items belonging to the group.
     * @param cutting_geometry_items - Cutting_geometry_items of the group.
     * @param transformation - Transformation applied to the group.
     *
     * @return Shared pointer to the created GeometryGroup.
     */
    std::shared_ptr<GeometryGroup> create_geometry_group_cutted(
        const std::string& name, const GeometryPtrList& geometries,
        const GeometryItemPtrList& cutting_geometry_items,
        const TransformationPtr& transformation) {
        auto group = std::make_shared<GeometryGroupCutted>(
            name, geometries, cutting_geometry_items, transformation);

        add_configure_geometry_group(group);

        return group;
    }

    /**
     * @brief Method to remove a GeometryItem.
     *
     * @param item - Shared pointer to the GeometryItem to remove.
     */
    // void remove_geometry(const std::shared_ptr<GeometryItem>& item) {
    //     // item->get_name();
    //     std::cout << "Removing item not implemented yet" << std::endl;
    //     // _names.erase(item->get_name());
    //     //  Your deletion logic here
    // }
    //  Your methods here

    static void callback_primitive_changed(GeometryIdType primitive_id) {
        // Print
        std::cout << "Primitive with id: " << primitive_id
                  << " has been modified.\n";
    }

    /**
     * @brief Getter for the root geometry group.
     *
     * @return Shared pointer to the root geometry group.
     */
    std::shared_ptr<GeometryGroup> get_root_geometry_group() const {
        return _root_geometry_group;
    }

    void create_mesh(double tolerance) {
        // Create the mesh
        _root_geometry_group->create_meshes(tolerance);

        // Now that the meshes are created, we can copy them to the root mesh.
        // First, reserve space before copying, by counting the total number of
        // points, triangles, etc. (TODO !!)
        copy_mesh();
    }

    void copy_mesh() {
        _trimesh_model.clear();
        TriMeshModel& trimesh_model = _trimesh_model;
        _root_geometry_group->iterate_geometry_meshed_items(
            *_root_geometry_group->get_transformation(),
            [&trimesh_model](
                const CoordinateTransformation& global_transformation,
                const std::shared_ptr<GeometryMeshedItem>& item) {
                auto coord_transf =
                    global_transformation.chain(*item->get_transformation());

                auto transformed_trimesh =
                    coord_transf.transform_trimesh(*(item->get_tri_mesh()));

                trimesh_model.add_mesh(transformed_trimesh, item->get_id());
                // std::cout << " Num triangles: "
                //           << trimesh_model.get_triangles().rows() <<
                //           std::endl;
            });
    }

    /**
     * @brief Getter for the root trimesh.
     * @return Reference to the root trimesh.
     */
    TriMeshModel& get_trimesh_model() { return _trimesh_model; }

  private:
    void add_configure_geometry_item(
        const std::shared_ptr<GeometryMeshedItem>& item) {
        const std::string name = item->get_name();

        if (_names.find(name) != _names.end()) {
            throw std::runtime_error("Name already exists.");
        }

        item->set_parent(_root_geometry_group);
        _root_geometry_group->add_geometry_item(item);
        _names.insert(name);

        // Add the item to the set of geometries
        _geometries.insert(item);
        // Add the item to the map of id - name
        _geometry_id_to_name[item->get_id()] = item->get_name();

        // Add the item to the map of primitive - geometry items
        _geometry_primitives_dependencies[item->get_primitive()->get_id()]
            .push_back(item->get_id());

        // Set the callback of the primitive
        item->get_primitive()->add_callback(
            [wptr = std::weak_ptr<GeometryModel>(shared_from_this())](
                GeometryIdType primitive_id) {
                if (auto sptr = wptr.lock()) {
                    sptr->callback_primitive_changed(primitive_id);
                }
            },
            get_id());
    }

    void add_configure_geometry_group(
        const std::shared_ptr<GeometryGroup>& item) {
        const std::string name = item->get_name();

        if (_names.find(name) != _names.end()) {
            throw std::runtime_error("Name already exists.");
        }

        // To create a GeometryGroup within the model, the geometries
        // need to be already in the model in the root node
        // otherwise an exception is thrown
        // This new group is added to the root node, and the geometries
        // are moved from the root node to the this group

        // Check if all the geometries are in the root node
        // TODO: Implement geometry iterator like:
        // for (auto geo_ptr : item->get_geometries()) {...}
        for (const auto& geoitem : item->get_geometry_items()) {
            const bool found = std::any_of(
                _root_geometry_group->get_geometry_items().begin(),
                _root_geometry_group->get_geometry_items().end(),
                [&geoitem](const auto& root_geoitem) {
                    return geoitem->get_id() == root_geoitem->get_id();
                });

            if (!found) {
                throw std::runtime_error(
                    "Geometry not in the root node. Cannot create group.");
            }
        }

        for (const auto& geogroup : item->get_geometry_groups()) {
            const bool found = std::any_of(
                _root_geometry_group->get_geometry_groups().begin(),
                _root_geometry_group->get_geometry_groups().end(),
                [&geogroup](const auto& root_geogroup) {
                    return geogroup->get_id() == root_geogroup->get_id();
                });

            if (!found) {
                throw std::runtime_error(
                    "Geometry not in the root node. Cannot create group.");
            }
        }

        // Add the group to the root node

        // First check if item is GeometryGroup or GeometryGroupCutted
        const std::shared_ptr<GeometryGroupCutted> item_geometry_group_cutted =
            std::dynamic_pointer_cast<GeometryGroupCutted>(item);
        if (item_geometry_group_cutted) {
            // `item` is a GeometryGroupCutted
            _root_geometry_group->add_geometry_group_cutted(
                item_geometry_group_cutted);
        } else {
            // `item` is a GeometryGroup
            _root_geometry_group->add_geometry_group(item);
        }
        item->set_parent(_root_geometry_group);
        _names.insert(name);

        // Add the item to the set of geometries
        _geometries.insert(item);
        // Add the item to the map of id - name
        _geometry_id_to_name[item->get_id()] = item->get_name();

        // Remove the geometries from the root node and change the parent
        for (const auto& geoitem : item->get_geometry_items()) {
            _root_geometry_group->remove_geometry_item(geoitem);
            geoitem->set_parent(item);
        }
        for (const auto& geogroup : item->get_geometry_groups()) {
            _root_geometry_group->remove_geometry_group(geogroup);
            geogroup->set_parent(item);
        }

        // If item is a GeometryGroupCutted, iterate over all cutting
        // primitives
        const std::shared_ptr<GeometryGroupCutted> cutted_item =
            std::dynamic_pointer_cast<GeometryGroupCutted>(item);
        if (cutted_item) {
            for (const auto& geocutteditem :
                 cutted_item->get_cutting_geometry_items()) {
                _root_geometry_group->remove_geometry_item(geocutteditem);
                geocutteditem->set_parent(item);
            }
        }
    }
};

}  // namespace pycanha::gmm
