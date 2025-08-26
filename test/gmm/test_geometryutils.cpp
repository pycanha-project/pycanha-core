// NOLINTBEGIN(misc-include-cleaner, bugprone-chained-comparison)
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "pycanha-core/gmm/geometryutils.hpp"
#include "pycanha-core/parameters.hpp"

using namespace pycanha;

TEST_CASE("Test the geometry utility functions", "[gmm][utils]") {
    using pycanha::LENGTH_TOL;
    using pycanha::Point3D;

    SECTION("Distance point to line segment") {
        using pycanha::gmm::dist_point_to_line_segment;
        const Point3D a(0.0, 0.0, 0.0);
        const Point3D b(1.0, 0.0, 0.0);

        const Point3D p0(0.5, 0.0, 0.0);
        REQUIRE_THAT(dist_point_to_line_segment(p0, a, b),
                     Catch::Matchers::WithinAbs(0.0, LENGTH_TOL));

        const Point3D p1(-0.5, 0.0, 0.0);
        REQUIRE_THAT(dist_point_to_line_segment(p1, a, b),
                     Catch::Matchers::WithinAbs(0.5, LENGTH_TOL));

        const Point3D p2(1.5, 0.0, 0.0);
        REQUIRE_THAT(dist_point_to_line_segment(p2, a, b),
                     Catch::Matchers::WithinAbs(0.5, LENGTH_TOL));

        const Point3D p3(0.5, 1.0, 1.0);
        REQUIRE_THAT(dist_point_to_line_segment(p3, a, b),
                     Catch::Matchers::WithinAbs(std::sqrt(2.0), LENGTH_TOL));
    }

    SECTION("Vector non-zero length") {
        using pycanha::gmm::are_vectors_nonzero_length;
        using pycanha::gmm::is_vector_nonzero_length;

        const Eigen::Vector3d v30(0.0, 0.0, 0.0);
        REQUIRE(is_vector_nonzero_length(v30) == false);

        const Eigen::Vector3d v31(0.0, 0.0, LENGTH_TOL * 0.5);
        REQUIRE(is_vector_nonzero_length(v31) == false);

        const Eigen::Vector3d v32(0.0, 0.0, LENGTH_TOL * 2.0);
        REQUIRE(is_vector_nonzero_length(v32) == true);

        const Eigen::Vector3d v33(0.5, -0.5, 0.1);
        REQUIRE(is_vector_nonzero_length(v33) == true);

        const Eigen::Vector2d v20(0.0, 0.0);
        REQUIRE(is_vector_nonzero_length(v20) == false);

        const Eigen::Vector2d v21(0.0, LENGTH_TOL * 0.5);
        REQUIRE(is_vector_nonzero_length(v21) == false);

        const Eigen::Vector2d v22(0.0, LENGTH_TOL * 2.0);
        REQUIRE(is_vector_nonzero_length(v22) == true);

        const Eigen::Vector2d v23(0.5, -0.5);
        REQUIRE(is_vector_nonzero_length(v23) == true);

        // Now the multiple vectors version
        REQUIRE(are_vectors_nonzero_length({&v30, &v31}) == false);
        REQUIRE(are_vectors_nonzero_length({&v32, &v33}) == true);
        REQUIRE(are_vectors_nonzero_length({&v30, &v31, &v32, &v33}) == false);
        REQUIRE(are_vectors_nonzero_length({&v20, &v21}) == false);
        REQUIRE(are_vectors_nonzero_length({&v22, &v23}) == true);
        REQUIRE(are_vectors_nonzero_length({&v20, &v21, &v22, &v23}) == false);
    }

    SECTION("Orthogonal vectors") {
        using pycanha::gmm::are_vectors_orthogonal;

        const Eigen::Vector3d v0(1.0, 0.0, 0.0);
        const Eigen::Vector3d v1(0.0, 1.0, 0.0);
        REQUIRE(are_vectors_orthogonal(v0, v1) == true);

        const Eigen::Vector3d v2(std::cos(ANGLE_TOL * 2.0),
                                 std::sin(ANGLE_TOL * 2.0), 0.0);
        const Eigen::Vector3d v3(0.0, 1.0, 0.0);
        REQUIRE(are_vectors_orthogonal(v2, v3) == false);

        const Eigen::Vector2d v4(1.0, 0.0);
        const Eigen::Vector2d v5(0.0, 1.0);
        REQUIRE(are_vectors_orthogonal(v4, v5) == true);

        const Eigen::Vector2d v6(1.0, 1.0);
        const Eigen::Vector2d v7(-1.0, 1.0);
        REQUIRE(are_vectors_orthogonal(v6, v7) == true);

        const Eigen::Vector2d v8(std::cos(ANGLE_TOL * 2.0),
                                 -std::sin(ANGLE_TOL * 2.0));
        const Eigen::Vector2d v9(0.0, 1.0);
        REQUIRE(are_vectors_orthogonal(v8, v9) == false);
    }

    SECTION("Parallel vectors") {
        using pycanha::gmm::are_vectors_parallel;

        const Eigen::Vector3d v0(1.0, 0.0, 0.0);
        const Eigen::Vector3d v1(2.3, 0.0, 0.0);
        REQUIRE(are_vectors_parallel(v0, v1) == true);

        const Eigen::Vector3d v2(std::cos(ANGLE_TOL * 2.0),
                                 std::sin(ANGLE_TOL * 2.0), 0.0);
        const Eigen::Vector3d v3(1.0, 0.0, 0.0);
        REQUIRE(are_vectors_parallel(v2, v3) == false);

        const Eigen::Vector3d v4(1.0, 0.0, 0.0);
        const Eigen::Vector3d v5(-2.1, 0.0, 0.0);
        REQUIRE(are_vectors_parallel(v4, v5) == true);

        const Eigen::Vector2d v6(1.0, 1.0);
        const Eigen::Vector2d v7(-0.5, -0.5);
        REQUIRE(are_vectors_parallel(v6, v7) == true);

        const Eigen::Vector2d v8(std::cos(ANGLE_TOL * 2.0),
                                 -std::sin(ANGLE_TOL * 2.0));
        const Eigen::Vector2d v9(2.0, 0.0);
        REQUIRE(are_vectors_parallel(v8, v9) == false);
    }
}
// NOLINTEND(misc-include-cleaner, bugprone-chained-comparison)
