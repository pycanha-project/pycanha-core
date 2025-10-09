// NOLINTBEGIN(misc-include-cleaner, bugprone-chained-comparison)
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <numbers>

#include "pycanha-core/gmm/transformations.hpp"
#include "pycanha-core/gmm/trimesh.hpp"
#include "pycanha-core/parameters.hpp"

using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::TransformOrder;
using pycanha::gmm::TriMesh;
using pycanha::gmm::VerticesList;
using std::numbers::pi;

TEST_CASE("CoordinateTransformation", "[gmm][coordinate_transformation]") {
    SECTION("Check constructor and set/get methods") {
        Vector3D translation(1.0, 2.0, 3.0);
        Vector3D const rotation(pi / 2.0, 0.0, 0.0);
        CoordinateTransformation transform(
            translation, rotation, TransformOrder::TRANSLATION_THEN_ROTATION);

        REQUIRE(transform.get_translation() == translation);
        // REQUIRE(transform.get_rotation() == rotation);
        REQUIRE(transform.get_order() ==
                TransformOrder::TRANSLATION_THEN_ROTATION);

        Vector3D new_translation(2.0, 3.0, 4.0);
        Vector3D const new_rotation(0.0, pi / 2.0, 0.0);
        transform.set_translation(new_translation);
        transform.set_rotation_angles(new_rotation);
        transform.set_order(TransformOrder::ROTATION_THEN_TRANSLATION);

        REQUIRE(transform.get_translation() == new_translation);
        // REQUIRE(transform.get_rotation() == new_rotation);
        REQUIRE(transform.get_order() ==
                TransformOrder::ROTATION_THEN_TRANSLATION);
    }

    SECTION("Check transform_point method") {
        Vector3D const point(1.0, 2.0, 3.0);
        Vector3D const translation(1.0, 2.0, 3.0);
        Vector3D const rotation(pi / 2.0, pi / 3.0, pi / 4.0);

        CoordinateTransformation const transform1(
            translation, rotation, TransformOrder::TRANSLATION_THEN_ROTATION);
        Vector3D const expected_point1(7.399237211089, -1.0860441631496,
                                       0.26794919243112);

        REQUIRE(transform1.transform_point(point).isApprox(
            expected_point1, pycanha::LENGTH_TOL));

        CoordinateTransformation const transform2(
            translation, rotation, TransformOrder::ROTATION_THEN_TRANSLATION);
        Vector3D const expected_point2(4.6996186055445, 1.45697791842522,
                                       3.13397459621556);
        REQUIRE(transform2.transform_point(point).isApprox(
            expected_point2, pycanha::LENGTH_TOL));
    }

    SECTION("Check transform_point_list_inplace method") {
        VerticesList points(2, 3);
        points.row(0) << 1.0, 2.0, 3.0;
        points.row(1) << 1.0, 2.0, 3.0;

        Vector3D const translation(1.0, 2.0, 3.0);
        Vector3D const rotation(pi / 2.0, pi / 3.0, pi / 4.0);

        CoordinateTransformation const transform1(
            translation, rotation, TransformOrder::TRANSLATION_THEN_ROTATION);
        Vector3D const expected_point1(7.399237211089, -1.0860441631496,
                                       0.26794919243112);

        transform1.transform_point_list_inplace(&points);
        Point3D transformed_point1 = points.row(0);
        // Point3D transformed_point2 = points.row(1);

        REQUIRE(
            transformed_point1.isApprox(expected_point1, pycanha::LENGTH_TOL));

        REQUIRE(
            transformed_point1.isApprox(expected_point1, pycanha::LENGTH_TOL));

        CoordinateTransformation const transform2(
            translation, rotation, TransformOrder::ROTATION_THEN_TRANSLATION);
        Vector3D const expected_point2(4.6996186055445, 1.45697791842522,
                                       3.13397459621556);
        points.row(0) << 1.0, 2.0, 3.0;
        points.row(1) << 1.0, 2.0, 3.0;

        transform2.transform_point_list_inplace(&points);
        transformed_point1 = points.row(0);
        REQUIRE(
            transformed_point1.isApprox(expected_point2, pycanha::LENGTH_TOL));

        REQUIRE(
            transformed_point1.isApprox(expected_point2, pycanha::LENGTH_TOL));
    }

    SECTION("Check transform_point_list method") {
        VerticesList points(2, 3);
        points.row(0) << 1.0, 2.0, 3.0;
        points.row(1) << 1.0, 2.0, 3.0;

        Vector3D const translation(1.0, 2.0, 3.0);
        Vector3D const rotation(pi / 2.0, pi / 3.0, pi / 4.0);

        CoordinateTransformation const transform1(
            translation, rotation, TransformOrder::TRANSLATION_THEN_ROTATION);

        auto transformed_points = transform1.transform_point_list(points);
        transform1.transform_point_list_inplace(&points);

        REQUIRE(transformed_points.isApprox(points, pycanha::LENGTH_TOL));
    }

    SECTION("Check transform_trimesh methods") {
        TriMesh mesh;
        VerticesList points(2, 3);
        points.row(0) << 1.0, 2.0, 3.0;
        points.row(1) << 1.0, 2.0, 3.0;
        mesh.set_vertices(points);

        Vector3D const translation(1.0, 2.0, 3.0);
        Vector3D const rotation(pi / 2.0, pi / 3.0, pi / 4.0);

        CoordinateTransformation const transform1(
            translation, rotation, TransformOrder::TRANSLATION_THEN_ROTATION);
        Vector3D const expected_point1(7.399237211089, -1.0860441631496,
                                       0.26794919243112);

        transform1.transform_trimesh_inplace(mesh);
        Point3D transformed_point1 = mesh.get_vertices().row(0);
        Point3D transformed_point2 = mesh.get_vertices().row(1);

        REQUIRE(
            transformed_point1.isApprox(expected_point1, pycanha::LENGTH_TOL));

        REQUIRE(
            transformed_point2.isApprox(expected_point1, pycanha::LENGTH_TOL));

        VerticesList points2(2, 3);
        points2.row(0) << 1.0, 2.0, 3.0;
        points2.row(1) << 1.0, 2.0, 3.0;

        // Check that the original points are not modified
        REQUIRE(points2.isApprox(points, pycanha::LENGTH_TOL));

        mesh.set_vertices(points);
        auto mesh2 = transform1.transform_trimesh(mesh);
        transformed_point1 = mesh2.get_vertices().row(0);
        transformed_point2 = mesh2.get_vertices().row(1);

        REQUIRE(
            transformed_point1.isApprox(expected_point1, pycanha::LENGTH_TOL));

        REQUIRE(
            transformed_point2.isApprox(expected_point1, pycanha::LENGTH_TOL));
    }

    SECTION("Check chained transformations") {
        Vector3D const point(1.0, 2.0, 3.0);
        // Vector3D translation(1.0, 2.0, 3.0);
        //  Vector3D rotation(pi / 2.0, pi / 3.0, pi / 4.0);
        Vector3D const translation(1.0, 2.0, 3.0);
        Vector3D const rotation(pi / 2.0, pi / 3.0, pi / 4.0);

        const CoordinateTransformation transform1(
            translation, rotation, TransformOrder::TRANSLATION_THEN_ROTATION);

        const CoordinateTransformation transform2(
            translation, rotation, TransformOrder::ROTATION_THEN_TRANSLATION);

        auto expected_point_1 =
            transform1.transform_point(transform1.transform_point(point));
        auto expected_point_2 =
            transform1.transform_point(transform2.transform_point(point));
        auto expected_point_3 =
            transform2.transform_point(transform1.transform_point(point));
        auto expected_point_4 =
            transform2.transform_point(transform2.transform_point(point));

        auto chain_transf_1 = transform1.chain(transform1);
        auto chain_transf_2 = transform1.chain(transform2);
        auto chain_transf_3 = transform2.chain(transform1);
        auto chain_transf_4 = transform2.chain(transform2);

        REQUIRE(chain_transf_1.transform_point(point).isApprox(
            expected_point_1, pycanha::LENGTH_TOL));
        REQUIRE(chain_transf_2.transform_point(point).isApprox(
            expected_point_2, pycanha::LENGTH_TOL));
        REQUIRE(chain_transf_3.transform_point(point).isApprox(
            expected_point_3, pycanha::LENGTH_TOL));
        REQUIRE(chain_transf_4.transform_point(point).isApprox(
            expected_point_4, pycanha::LENGTH_TOL));
    }
}

// NOLINTEND(misc-include-cleaner, bugprone-chained-comparison)
