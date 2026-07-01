#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <stdexcept>

#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/materials/color.hpp"
#include "pycanha-core/gmm/materials/optical_material.hpp"

namespace {

using pycanha::gmm::BulkMaterial;
using pycanha::gmm::Color;
using pycanha::gmm::OpticalMaterial;

}  // namespace

TEST_CASE("BulkMaterial validates and exposes properties", "[gmm][materials]") {
    const BulkMaterial aluminum("aluminum_6061", 2700.0, 167.0, 896.0);
    REQUIRE(aluminum.get_name() == "aluminum_6061");
    REQUIRE(aluminum.get_density() == 2700.0);
    REQUIRE(aluminum.get_conductivity() == 167.0);
    REQUIRE(aluminum.get_specific_heat() == 896.0);

    REQUIRE_THROWS_AS(BulkMaterial("bad", -1.0, 1.0, 1.0),
                      std::invalid_argument);
    BulkMaterial mat;  // default zeros are valid
    REQUIRE_THROWS_AS(mat.set_conductivity(-5.0), std::invalid_argument);
    REQUIRE(mat.get_conductivity() == 0.0);
}

TEST_CASE("OpticalMaterial keeps full 6-DOF with named accessors",
          "[gmm][materials]") {
    const OpticalMaterial white("white_paint", 0.88, 0.22);
    REQUIRE(white.emissivity_ir() == 0.88);
    REQUIRE(white.absorptivity_solar() == 0.22);
    REQUIRE(white.get_th_optical_properties()[0] == 0.88);
    REQUIRE(white.get_th_optical_properties()[3] == 0.22);

    const OpticalMaterial black;  // default black body {1,0,0,1,0,0}
    REQUIRE(black.emissivity_ir() == 1.0);
    REQUIRE(black.absorptivity_solar() == 1.0);

    const OpticalMaterial full(
        "full", std::array<double, 6>{0.9, 0.1, 0.0, 0.3, 0.2, 0.5});
    REQUIRE(full.get_th_optical_properties()[4] == 0.2);

    REQUIRE_THROWS_AS(OpticalMaterial("bad", 1.5, 0.2), std::invalid_argument);
    OpticalMaterial mat;
    REQUIRE_THROWS_AS(mat.set_absorptivity_solar(-0.1), std::invalid_argument);
}

TEST_CASE("Color supports channels and the named palette", "[gmm][materials]") {
    const Color blue(0, 0, 255);
    REQUIRE(blue.red() == 0);
    REQUIRE(blue.green() == 0);
    REQUIRE(blue.blue() == 255);

    const Color white("WHITE");
    REQUIRE(white.get_rgb() == std::array<std::uint8_t, 3>{255, 255, 255});

    REQUIRE_THROWS_AS(Color("NOT_A_COLOR"), std::invalid_argument);
}
