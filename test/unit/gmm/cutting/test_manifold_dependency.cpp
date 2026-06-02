#include <manifold/manifold.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Manifold dependency is linked and usable", "[gmm][cutting]") {
    const manifold::Manifold sphere = manifold::Manifold::Sphere(1.0, 24);

    REQUIRE(sphere.Status() == manifold::Manifold::Error::NoError);
    REQUIRE(sphere.NumTri() > 0U);
    REQUIRE(sphere.GetMeshGL64().NumTri() > 0U);
}
