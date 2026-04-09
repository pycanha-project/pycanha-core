#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/thermaldata/interpolation_utils.hpp"

TEST_CASE("Interpolation locate resolves interior and boundary queries",
          "[thermaldata][interp]") {
    constexpr std::array<double, 4> x = {0.0, 2.0, 4.0, 6.0};

    const auto interior = pycanha::detail::locate(x.data(), 4, 3.0);
    REQUIRE(interior.lower_index == 1);
    REQUIRE(interior.fraction == Catch::Approx(0.5));
    REQUIRE(!interior.below_range);
    REQUIRE(!interior.above_range);

    const auto exact = pycanha::detail::locate(x.data(), 4, 2.0);
    REQUIRE(exact.lower_index == 1);
    REQUIRE(exact.fraction == Catch::Approx(0.0));

    const auto upper = pycanha::detail::locate(x.data(), 4, 6.0);
    REQUIRE(upper.lower_index == 2);
    REQUIRE(upper.fraction == Catch::Approx(1.0));
}

TEST_CASE("Interpolation locate handles extrapolation and single values",
          "[thermaldata][interp]") {
    constexpr std::array<double, 2> x = {1.0, 3.0};

    const auto below = pycanha::detail::locate(x.data(), 2, 0.0);
    REQUIRE(below.lower_index == 0);
    REQUIRE(below.fraction == Catch::Approx(-0.5));
    REQUIRE(below.below_range);
    REQUIRE(!below.above_range);

    const auto above = pycanha::detail::locate(x.data(), 2, 5.0);
    REQUIRE(above.lower_index == 0);
    REQUIRE(above.fraction == Catch::Approx(2.0));
    REQUIRE(!above.below_range);
    REQUIRE(above.above_range);

    constexpr std::array<double, 1> single = {7.0};
    const auto single_location = pycanha::detail::locate(single.data(), 1, 6.0);
    REQUIRE(single_location.lower_index == 0);
    REQUIRE(single_location.fraction == Catch::Approx(0.0));
    REQUIRE(single_location.below_range);
}