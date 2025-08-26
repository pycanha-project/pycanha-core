#include <catch2/catch_test_macros.hpp>
#include <memory>  // std::make_shared
#include <numbers>

#include "pycanha-core/gmm/geometry.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/primitives.hpp"
#include "pycanha-core/gmm/thermalmesh.hpp"
#include "pycanha-core/gmm/transformations.hpp"
#include "pycanha-core/parameters.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

using pycanha::Point3D;
using pycanha::gmm::GeometryModel;

TEST_CASE("Geometry Model", "[gmm][geometry][GeometryModel]") {
    // First we create the geometry items and groups to be tested in the
    // different sections

    using pycanha::gmm::Cylinder;
    using pycanha::gmm::Quadrilateral;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::Triangle;

    using pycanha::gmm::GeometryPtrList;

    // using CylPtr = std::shared_ptr<Cylinder>;
    // using QuadPtr = std::shared_ptr<Quadrilateral>;
    // using RectPtr = std::shared_ptr<Rectangle>;
    // using TriPtr = std::shared_ptr<Triangle>;

    // TODO: SPHERE
    // TODO: other geometry

    using std::numbers::pi;

    const Point3D p1(0.0, 0.0, 0.0);
    const Point3D p2(1.0, 0.0, 0.0);
    const Point3D p3(1.0, 1.0, 0.0);
    const Point3D p4(0.0, 1.0, 0.0);

    // Primitives
    auto tri = std::make_shared<Triangle>(p1, p2, p3);
    auto rect = std::make_shared<Rectangle>(p1, p2, p4);
    auto quad = std::make_shared<Quadrilateral>(p1, p2, p3, p4);
    auto cyl = std::make_shared<Cylinder>(p1, p2, p4, 1.0, 0.0, pi * 2.0);

    // Transformation
    auto transf_1 = std::make_shared<pycanha::gmm::CoordinateTransformation>(
        Vector3D(1.0, 2.0, 3.0), Vector3D(pi / 2.0, 0.0, 0.0),
        pycanha::gmm::TransformOrder::TRANSLATION_THEN_ROTATION);

    auto transf_2 = std::make_shared<pycanha::gmm::CoordinateTransformation>(
        Vector3D(1.0, 2.0, 3.0), Vector3D(pi / 2.0, 0.0, 0.0),
        pycanha::gmm::TransformOrder::TRANSLATION_THEN_ROTATION);

    // Thermal Meshes
    auto th_mesh_1 = std::make_shared<pycanha::gmm::ThermalMesh>();

    auto th_mesh_2 = std::make_shared<pycanha::gmm::ThermalMesh>();

    // Create Geometry Items

    SECTION("Check GeometryItem constructor and set/get methods") {
        auto geo_model = std::make_shared<GeometryModel>();
        auto geo_item_1 = geo_model->create_geometry_item("geo_item_1", rect,
                                                          transf_1, th_mesh_1);

        auto geo_item_2 = geo_model->create_geometry_item("geo_item_2", rect,
                                                          transf_1, th_mesh_1);

        auto geo_item_3 = geo_model->create_geometry_item("geo_item_3", rect,
                                                          transf_1, th_mesh_1);

        auto geo_item_4 = geo_model->create_geometry_item("geo_item_4", rect,
                                                          transf_1, th_mesh_1);

        auto geo_group_1 = geo_model->create_geometry_group(
            "geo_group_1", {geo_item_1, geo_item_2}, transf_1);
        auto geo_group_2 = geo_model->create_geometry_group(
            "geo_group_2", {geo_group_1, geo_item_3}, transf_1);

        geo_model->create_mesh(0.01);
    }
}
