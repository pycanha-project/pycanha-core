#include <catch2/catch_test_macros.hpp>
#include <memory>  // std::make_shared
#include <numbers>

#include "pycanha-core/gmm/geometry.hpp"
#include "pycanha-core/gmm/primitives.hpp"
#include "pycanha-core/gmm/thermalmesh.hpp"
#include "pycanha-core/gmm/transformations.hpp"
#include "pycanha-core/parameters.hpp"

using namespace pycanha;

using pycanha::Point3D;
using pycanha::gmm::GeometryGroup;
using pycanha::gmm::GeometryMeshedItem;

TEST_CASE("Geometry Items and Groups",
          "[gmm][geometry][GeometryItem][GeometryGroup]") {
    // First we create the geometry items and groups to be tested in the
    // different sections
    using pycanha::gmm::Cone;
    using pycanha::gmm::Cylinder;
    using pycanha::gmm::Disc;
    using pycanha::gmm::Quadrilateral;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::Sphere;
    using pycanha::gmm::Triangle;

    using pycanha::gmm::GeometryPtrList;

    // using CylPtr = std::shared_ptr<Cylinder>;
    // using QuadPtr = std::shared_ptr<Quadrilateral>;
    // using RectPtr = std::shared_ptr<Rectangle>;
    // using TriPtr = std::shared_ptr<Triangle>;
    // using DiscPtr = std::shared_ptr<Disc>;

    // TODO: SPHERE
    // TODO: other geometry

    using std::numbers::pi;

    const Point3D p1(0.0, 0.0, 0.0);
    const Point3D p2(1.0, 0.0, 0.0);
    const Point3D p3(1.0, 1.0, 0.0);
    const Point3D p4(0.0, 1.0, 0.0);
    const Point3D p5(0.0, 0.0, 1.0);

    // Primitives
    auto tri = std::make_shared<Triangle>(p1, p2, p3);
    auto rect = std::make_shared<Rectangle>(p1, p2, p4);
    auto quad = std::make_shared<Quadrilateral>(p1, p2, p3, p4);
    auto cyl = std::make_shared<Cylinder>(p1, p2, p4, 1.0, 0.0, pi * 2.0);
    auto disc = std::make_shared<Disc>(p1, p5, p3, 0.5, 1.0, 0.0, pi * 2.0);
    auto sph =
        std::make_shared<Sphere>(p1, p5, p2, 1.0, -1.0, 1.0, 0.0, pi * 2.0);
    auto cone = std::make_shared<Cone>(p1, p5, p2, 0.0, 1.0, 0.0, pi * 2.0);

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
    auto geo_item_1 = std::make_shared<GeometryMeshedItem>("geo_item_1", tri,
                                                           transf_1, th_mesh_1);
    auto geo_item_2 = std::make_shared<GeometryMeshedItem>("geo_item_2", rect,
                                                           transf_1, th_mesh_1);
    auto geo_item_3 = std::make_shared<GeometryMeshedItem>("geo_item_3", quad,
                                                           transf_2, th_mesh_1);
    auto geo_item_4 = std::make_shared<GeometryMeshedItem>("geo_item_4", cyl,
                                                           transf_2, th_mesh_1);
    auto geo_item_5 = std::make_shared<GeometryMeshedItem>("geo_item_5", disc,
                                                           transf_2, th_mesh_1);

    // Create GeometryGroup with two GeometryItems
    auto geo_group_1 = std::make_shared<GeometryGroup>(
        "geo_group_1", GeometryPtrList{geo_item_1, geo_item_2}, transf_2);

    // Create a GeometryGroup with an item and a group
    auto geo_group_2 = std::make_shared<GeometryGroup>(
        "geo_group_2", GeometryPtrList{geo_item_3, geo_group_1}, transf_1);

    // Create a GeometryGroup with just an item
    auto geo_group_3 = std::make_shared<GeometryGroup>(
        "geo_group_3", GeometryPtrList{geo_item_4}, transf_1);

    // Create a GeometryGroup with just a group
    auto geo_group_4 = std::make_shared<GeometryGroup>(
        "geo_group_4", GeometryPtrList{geo_group_3}, transf_1);

    // Create a GeometryGroup with two groups
    auto geo_group_5 = std::make_shared<GeometryGroup>(
        "geo_group_5", GeometryPtrList{geo_group_2, geo_group_4}, transf_1);

    /*
    SECTION("Check GeometryItem constructor and set/get methods") {
        REQUIRE(geo_item_1->get_name() == "geo_item_1");
        REQUIRE(geo_item_2->get_name() == "geo_item_2");

        // Check the correct primitive type
        REQUIRE_NOTHROW(std::get<TriPtr>(geo_item_1->get_primitive()));
        REQUIRE_NOTHROW(std::get<RectPtr>(geo_item_2->get_primitive()));
        REQUIRE_NOTHROW(std::get<QuadPtr>(geo_item_3->get_primitive()));
        REQUIRE_NOTHROW(std::get<CylPtr>(geo_item_4->get_primitive()));

        // Retrieving the wrong primitive type throws an exception
        CHECK_THROWS(std::get<RectPtr>(geo_item_1->get_primitive()));
        CHECK_THROWS(std::get<QuadPtr>(geo_item_2->get_primitive()));
        CHECK_THROWS(std::get<CylPtr>(geo_item_3->get_primitive()));
        CHECK_THROWS(std::get<TriPtr>(geo_item_4->get_primitive()));

        // Check the actual primitive is stored
        REQUIRE(std::get<TriPtr>(geo_item_1->get_primitive()) == tri);
        REQUIRE(std::get<RectPtr>(geo_item_2->get_primitive()) == rect);
        REQUIRE(std::get<QuadPtr>(geo_item_3->get_primitive()) == quad);
        REQUIRE(std::get<CylPtr>(geo_item_4->get_primitive()) == cyl);

        REQUIRE(geo_item_1->get_transformation() == transf_1);
        REQUIRE(geo_item_1->get_transformation() != transf_2);
        transf_1->set_rotation(Vector3D(0.0, pi / 2.0, 0.0));
        REQUIRE(geo_item_1->get_transformation() == transf_1);

        geo_item_1->set_name("new_geo_item_1");
        REQUIRE(geo_item_1->get_name() == "new_geo_item_1");

        geo_item_1->set_primitive(rect);
        REQUIRE(std::get<RectPtr>(geo_item_1->get_primitive()) == rect);

        geo_item_1->set_transformation(transf_2);
        REQUIRE(geo_item_1->get_transformation() == transf_2);
        REQUIRE(geo_item_2->get_transformation() !=
                geo_item_1->get_transformation());
    }
    */
    SECTION("TODO: GeometryMeshedItem set/get methods"){};

    SECTION("Check GeometryGroup constructor and set/get methods") {}
}
