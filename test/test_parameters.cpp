
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/parameters.hpp"

// Unit tests for the LinearSystemSolver class
TEST_CASE("Test 3D point", "[Point3D]") {
  constexpr double p_coord_x = 1.0;
  constexpr double p_coord_y = 2.0;
  constexpr double p_coord_z = 3.0;
  pycanha::Point3D point(p_coord_x, p_coord_y, p_coord_z);
  REQUIRE((point.x() == p_coord_x && point.y() == p_coord_y &&
           point.z() == p_coord_z));
}
