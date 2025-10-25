#include <Eigen/Core>
#include <bit>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <string>
#include <variant>

#include "pycanha-core/parameters/parameters.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

TEST_CASE("Parameters add and retrieve scalars", "[parameters]") {
    Parameters params;

    params.add_parameter("temp", 295.0);
    params.add_parameter("enabled", true);

    REQUIRE(params.size() == 2);
    REQUIRE(params.contains("temp"));
    REQUIRE(params.contains("enabled"));

    const auto temp = params.get_parameter("temp");
    REQUIRE(std::holds_alternative<double>(temp));
    REQUIRE(std::get<double>(temp) == Catch::Approx(295.0));

    const auto enabled = params.get_parameter("enabled");
    REQUIRE(std::holds_alternative<bool>(enabled));
    REQUIRE(std::get<bool>(enabled));

    params.remove_parameter("enabled");
    REQUIRE_FALSE(params.contains("enabled"));
    REQUIRE(params.size() == 1);
}

TEST_CASE("Parameters update values when type and shape match",
          "[parameters]") {
    Parameters params;

    Parameters::MatrixRXd matrix(2, 2);
    matrix << 1.0, 2.0, 3.0, 4.0;
    params.add_parameter("mat", matrix);

    Parameters::MatrixRXd updated(2, 2);
    updated << 5.0, 6.0, 7.0, 8.0;
    params.set_parameter("mat", updated);

    const auto after_update = params.get_parameter("mat");
    REQUIRE(std::holds_alternative<Parameters::MatrixRXd>(after_update));
    const auto stored = std::get<Parameters::MatrixRXd>(after_update);
    REQUIRE(stored(0, 0) == Catch::Approx(5.0));
    REQUIRE(stored(1, 1) == Catch::Approx(8.0));

    Parameters::MatrixRXd wrong_shape(3, 1);
    wrong_shape << 1.0, 1.0, 1.0;
    params.set_parameter("mat", wrong_shape);

    const auto after_wrong = params.get_parameter("mat");
    const auto stored_after_wrong =
        std::get<Parameters::MatrixRXd>(after_wrong);
    REQUIRE(stored_after_wrong.rows() == 2);
    REQUIRE(stored_after_wrong.cols() == 2);
    REQUIRE(stored_after_wrong(0, 0) == Catch::Approx(5.0));

    params.set_parameter("mat", 42.0);
    const auto after_type_mismatch = params.get_parameter("mat");
    REQUIRE(std::holds_alternative<Parameters::MatrixRXd>(after_type_mismatch));
}

TEST_CASE("Parameters report missing entries with NaN", "[parameters]") {
    Parameters params;

    const auto missing = params.get_parameter("missing");
    REQUIRE(std::holds_alternative<double>(missing));
    REQUIRE(std::isnan(std::get<double>(missing)));

    REQUIRE(params.get_idx("missing") == -1);
    REQUIRE(params.get_size_of_parameter("missing") == 0U);
}

TEST_CASE("Parameters expose memory pointers and sizes", "[parameters]") {
    Parameters params;

    params.add_parameter("scalar", 10.0);
    params.add_parameter("label", std::string("alpha"));

    auto* scalar_ptr = static_cast<double*>(params.get_value_ptr("scalar"));
    REQUIRE(scalar_ptr != nullptr);
    REQUIRE(*scalar_ptr == Catch::Approx(10.0));

    const auto scalar_address = params.get_memory_address("scalar");
    REQUIRE(scalar_address != 0U);
    const auto expected_address =
        static_cast<std::uint64_t>(std::bit_cast<std::uintptr_t>(scalar_ptr));
    REQUIRE(scalar_address == expected_address);

    const auto scalar_size = params.get_size_of_parameter("scalar");
    REQUIRE(scalar_size == sizeof(double));

    const auto label_size = params.get_size_of_parameter("label");
    REQUIRE(label_size == (std::string("alpha").size() + 1U));

    REQUIRE(params.get_idx("scalar") >= 0);
    REQUIRE(params.get_idx("label") >= 0);
}
