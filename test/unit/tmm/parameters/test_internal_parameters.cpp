#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <string>

#include "pycanha-core/parameters/parameters.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("Internal parameters are readable but protected from user edits",
          "[parameters][internal]") {
    Parameters params;

    params.add_parameter("external", 1.0);
    params.add_internal_parameter("time", 0.0);

    REQUIRE(params.contains("time"));
    REQUIRE(params.is_internal_parameter("time"));
    REQUIRE(params.get_idx("time").has_value());
    REQUIRE(params.size() == 2U);
    REQUIRE(std::get<double>(params.get_parameter("time")) ==
            Catch::Approx(0.0));

    params.set_parameter("time", 2.0);
    params.rename_parameter("time", "renamed_time");
    params.remove_parameter("time");

    REQUIRE(params.contains("time"));
    REQUIRE_FALSE(params.contains("renamed_time"));
    REQUIRE(std::get<double>(params.get_parameter("time")) ==
            Catch::Approx(0.0));

    params.set_internal_parameter("time", 3.5);
    REQUIRE(std::get<double>(params.get_parameter("time")) ==
            Catch::Approx(3.5));

    params.remove_internal_parameter("external");
    REQUIRE(params.contains("external"));

    params.remove_internal_parameter("time");
    REQUIRE_FALSE(params.contains("time"));
    REQUIRE(params.size() == 1U);
}

TEST_CASE("Internal parameters bypass add lock but keep removal guarded",
          "[parameters][internal]") {
    Parameters params;

    params.add_parameter("external", std::int64_t{1});
    params.lock_structure();

    const auto version_before = params.get_structure_version();
    params.add_internal_parameter("time", 0.0);

    REQUIRE(params.contains("time"));
    REQUIRE(params.is_internal_parameter("time"));
    REQUIRE(params.get_structure_version() == version_before + 1U);

    params.remove_internal_parameter("time");
    REQUIRE(params.contains("time"));

    params.unlock_structure();
    params.remove_internal_parameter("time");
    REQUIRE_FALSE(params.contains("time"));
}

TEST_CASE("Internal parameters keep type and shape compatibility",
          "[parameters][internal]") {
    Parameters params;

    params.add_parameter("external", 1.0);

    Parameters::MatrixRXd matrix(1, 2);
    matrix << 2.0, 3.0;
    params.add_internal_parameter("mat", matrix);

    Parameters::MatrixRXd wrong_shape(2, 1);
    wrong_shape << 4.0, 5.0;
    params.set_internal_parameter("mat", wrong_shape);
    params.set_internal_parameter("mat", true);
    params.set_internal_parameter("external", 2.0);
    params.set_internal_parameter("missing", 3.0);

    const auto stored_matrix =
        std::get<Parameters::MatrixRXd>(params.get_parameter("mat"));
    REQUIRE(stored_matrix.rows() == 1);
    REQUIRE(stored_matrix.cols() == 2);
    REQUIRE(stored_matrix(0, 0) == Catch::Approx(2.0));
    REQUIRE(stored_matrix(0, 1) == Catch::Approx(3.0));
    REQUIRE(std::get<double>(params.get_parameter("external")) ==
            Catch::Approx(1.0));
}
