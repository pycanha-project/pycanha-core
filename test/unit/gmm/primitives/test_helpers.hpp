#pragma once

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm::tests {

template <typename Primitive, std::size_t N>
void require_round_trip(const Primitive& primitive,
                        const std::array<Point3D, N>& points,
                        double tolerance = LENGTH_TOL) {
    for (const Point3D& point : points) {
        const Point2D uv = primitive.to_uv(point);
        REQUIRE(primitive.to_cartesian(uv).isApprox(point, tolerance));
    }
}

inline void require_parallel(const Vector3D& actual, const Vector3D& expected,
                             double tolerance = LENGTH_TOL) {
    REQUIRE(actual.normalized().isApprox(expected.normalized(), tolerance));
}

}  // namespace pycanha::gmm::tests
