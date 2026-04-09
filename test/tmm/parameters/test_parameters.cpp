#include <Eigen/Core>
#include <bit>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <string>
#include <variant>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/parameters.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

// NOLINTBEGIN(bugprone-chained-comparison)

TEST_CASE("Parameters add and retrieve scalars", "[parameters]") {
    // NOLINTNEXTLINE(misc-const-correctness)
    Parameters params;

    params.add_parameter("temp", 295.0);
    params.add_parameter("enabled", true);
    params.add_parameter("count", std::int64_t{42});

    REQUIRE(params.size() == 3);
    REQUIRE(params.contains("temp"));
    REQUIRE(params.contains("enabled"));
    REQUIRE(params.contains("count"));

    const auto temp = params.get_parameter("temp");
    REQUIRE(std::holds_alternative<double>(temp));
    REQUIRE(std::get<double>(temp) == Catch::Approx(295.0));

    const auto enabled = params.get_parameter("enabled");
    REQUIRE(std::holds_alternative<bool>(enabled));
    REQUIRE(std::get<bool>(enabled));

    const auto count = params.get_parameter("count");
    REQUIRE(std::holds_alternative<std::int64_t>(count));
    REQUIRE(std::get<std::int64_t>(count) == 42);

    params.set_parameter("count", std::int64_t{128});
    const auto updated_count = params.get_parameter("count");
    REQUIRE(std::get<std::int64_t>(updated_count) == 128);

    params.remove_parameter("enabled");
    REQUIRE(!params.contains("enabled"));
    REQUIRE(params.size() == 2);
}

TEST_CASE("Parameters update values when type and shape match",
          "[parameters]") {
    // NOLINTNEXTLINE(misc-const-correctness)
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
    const Parameters params;

    const auto missing = params.get_parameter("missing");
    REQUIRE(std::holds_alternative<double>(missing));
    REQUIRE(std::isnan(std::get<double>(missing)));

    REQUIRE_FALSE(params.get_parameter_optional("missing").has_value());

    REQUIRE_FALSE(params.get_idx("missing").has_value());
    REQUIRE(params.get_size_of_parameter("missing") == 0U);
}

TEST_CASE("Parameters keep stable slot indices across deletion",
          "[parameters]") {
    Parameters params;

    params.add_parameter("a", 1.0);
    params.add_parameter("b", 2.0);
    params.add_parameter("c", 3.0);

    const auto idx_a = params.get_idx("a");
    const auto idx_b = params.get_idx("b");
    const auto idx_c = params.get_idx("c");

    REQUIRE(idx_a.has_value());
    REQUIRE(idx_b.has_value());
    REQUIRE(idx_c.has_value());
    const auto idx_a_value = idx_a.value_or(Index{-1});
    const auto idx_b_value = idx_b.value_or(Index{-1});
    const auto idx_c_value = idx_c.value_or(Index{-1});
    REQUIRE(idx_a_value == 0);
    REQUIRE(idx_b_value == 1);
    REQUIRE(idx_c_value == 2);

    params.remove_parameter("b");
    params.add_parameter("d", 4.0);

    const auto idx_d = params.get_idx("d");
    REQUIRE(idx_d.has_value());
    const auto idx_a_after = params.get_idx("a");
    const auto idx_c_after = params.get_idx("c");
    REQUIRE(idx_a_after.has_value());
    REQUIRE(idx_c_after.has_value());
    REQUIRE(idx_a_after.value_or(Index{-1}) == idx_a_value);
    REQUIRE(idx_c_after.value_or(Index{-1}) == idx_c_value);
    REQUIRE(idx_d.value_or(Index{-1}) == 3);
    REQUIRE_FALSE(params.is_parameter_valid(idx_b_value));
}

TEST_CASE("Parameters rename entries and track structure version",
          "[parameters]") {
    Parameters params;

    params.add_parameter("old_name", 2.0);
    const auto idx = params.get_idx("old_name");
    REQUIRE(idx.has_value());
    const auto version_after_add = params.get_structure_version();

    params.rename_parameter("old_name", "new_name");

    REQUIRE_FALSE(params.contains("old_name"));
    REQUIRE(params.contains("new_name"));
    REQUIRE(params.get_idx("new_name") == idx);
    REQUIRE(params.get_structure_version() == version_after_add + 1U);
}

TEST_CASE("Parameters expose explicit handle getters and setters",
          "[parameters]") {
    Parameters params;

    params.add_parameter("value", 12.0);

    auto handle = params.get_parameter_handle("value");
    REQUIRE(handle.is_valid());
    REQUIRE(handle.get_idx().has_value());
    const auto initial_name = handle.get_name();
    REQUIRE(initial_name.has_value());
    REQUIRE(initial_name.value_or(std::string{}) == "value");

    const auto current_value = handle.get_value();
    REQUIRE(current_value.has_value());
    const auto current_value_variant =
        current_value.value_or(Parameters::ThermalValue{0.0});
    REQUIRE(std::get<double>(current_value_variant) == Catch::Approx(12.0));

    handle.set_value(18.0);
    REQUIRE(std::get<double>(params.get_parameter("value")) ==
            Catch::Approx(18.0));

    handle.rename("renamed_value");
    REQUIRE_FALSE(params.contains("value"));
    REQUIRE(params.contains("renamed_value"));
    const auto renamed_name = handle.get_name();
    REQUIRE(renamed_name.has_value());
    REQUIRE(renamed_name.value_or(std::string{}) == "renamed_value");

    handle.remove();
    REQUIRE_FALSE(handle.is_valid());
    REQUIRE_FALSE(params.contains("renamed_value"));
}

TEST_CASE("Parameters block structural edits while locked", "[parameters]") {
    Parameters params;

    params.add_parameter("scalar", 10.0);
    params.lock_structure();

    const auto locked_version = params.get_structure_version();
    params.add_parameter("new_scalar", 20.0);
    params.rename_parameter("scalar", "renamed");
    params.remove_parameter("scalar");

    REQUIRE_FALSE(params.contains("new_scalar"));
    REQUIRE_FALSE(params.contains("renamed"));
    REQUIRE(params.contains("scalar"));
    REQUIRE(params.get_structure_version() == locked_version);

    params.set_parameter("scalar", 30.0);
    REQUIRE(std::get<double>(params.get_parameter("scalar")) ==
            Catch::Approx(30.0));

    params.unlock_structure();
    REQUIRE_FALSE(params.is_structure_locked());
}

TEST_CASE("Parameters expose memory pointers and sizes", "[parameters]") {
    // NOLINTNEXTLINE(misc-const-correctness)
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
    const auto expected_label_size = std::string("alpha").size() + 1U;
    REQUIRE(label_size == expected_label_size);

    REQUIRE(params.get_idx("scalar").has_value());
    REQUIRE(params.get_idx("label").has_value());
}

// NOLINTEND(bugprone-chained-comparison)
