#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>  // NOLINT(misc-include-cleaner)
#include <unordered_map>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/io/esatan.hpp"
#include "pycanha-core/solvers/sslu.hpp"
#include "pycanha-core/thermaldata/dense_time_series.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace {

std::filesystem::path get_reference_tmd_path() {
    const std::filesystem::path this_file(__FILE__);
    const std::filesystem::path test_root =
        this_file.parent_path().parent_path();
    return test_root / "data" / "esatan" / "DISCTR_TRANSIENT.TMD";
}

std::unordered_map<int, pycanha::Index> build_column_lookup(
    const std::vector<int>& node_numbers) {
    std::unordered_map<int, pycanha::Index> lookup;
    lookup.reserve(node_numbers.size());

    for (std::size_t i = 0; i < node_numbers.size(); ++i) {
        lookup.emplace(node_numbers[i], static_cast<pycanha::Index>(i));
    }

    return lookup;
}

void require_monotonic_times(const pycanha::DenseTimeSeries& series) {
    for (pycanha::Index i = 1; i < series.num_timesteps(); ++i) {
        REQUIRE(series.times()(i) >= series.times()(i - 1));
    }
}

void require_default_transient_series(const pycanha::ThermalData& thermal_data,
                                      const std::vector<int>& node_numbers) {
    for (const auto* suffix : {"T", "C", "QA", "QE", "QI", "QR", "QS"}) {
        const std::string series_name = std::string("transient_") + suffix;
        REQUIRE(thermal_data.has_dense_time_series(series_name));

        const auto& series = thermal_data.get_dense_time_series(series_name);
        REQUIRE(series.num_timesteps() > 1);
        REQUIRE(series.num_columns() ==
                static_cast<pycanha::Index>(node_numbers.size()));
        require_monotonic_times(series);
    }
}
}  // namespace

TEST_CASE("ESATANReader can import real TMD and model is solvable",
          "[tmm][esatan]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    auto model = std::make_shared<pycanha::ThermalMathematicalModel>(
        "esatan-real-file-smoke-test");
    pycanha::ESATANReader reader(*model);
    REQUIRE_NOTHROW(reader.read_tmd(reference_tmd_path.string()));

    REQUIRE(model->nodes().get_num_nodes() > 0);

    pycanha::SSLU solver(model);
    solver.MAX_ITERS = 2;
    solver.abstol_temp = 1.0;

    REQUIRE_NOTHROW(solver.initialize());
    REQUIRE(solver.solver_initialized);
    REQUIRE_NOTHROW(solver.solve());
}

TEST_CASE("read_tmd_transient imports default transient node attributes",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "transient");

    REQUIRE(!node_numbers.empty());
    require_default_transient_series(thermal_data, node_numbers);
}

TEST_CASE(
    "read_tmd_transient temperature matches steady-state import at the "
    "first timestep",
    "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    auto model = std::make_shared<pycanha::ThermalMathematicalModel>(
        "esatan-transient-cross-check");
    pycanha::ESATANReader reader(*model);
    REQUIRE_NOTHROW(reader.read_tmd(reference_tmd_path.string()));

    pycanha::ThermalData thermal_data;
    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "transient");
    const auto column_lookup = build_column_lookup(node_numbers);
    const auto& temperature_series =
        thermal_data.get_dense_time_series("transient_T");

    for (pycanha::Index i = 0; i < model->nodes().get_num_nodes(); ++i) {
        pycanha::Node node = model->nodes().get_node_from_idx(i);
        const auto iterator = column_lookup.find(node.get_node_num());
        REQUIRE(iterator != column_lookup.end());
        REQUIRE(temperature_series.values()(0, iterator->second) ==
                Catch::Approx(node.get_T()));
    }
}

TEST_CASE("read_tmd_transient preserves inactive nodes in returned columns",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    auto model = std::make_shared<pycanha::ThermalMathematicalModel>(
        "esatan-inactive-node-check");
    pycanha::ESATANReader reader(*model);
    REQUIRE_NOTHROW(reader.read_tmd(reference_tmd_path.string()));

    pycanha::ThermalData thermal_data;
    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "transient");

    REQUIRE(node_numbers.size() == 103U);
    REQUIRE(model->nodes().get_num_nodes() == 102);

    const auto& temperature_series =
        thermal_data.get_dense_time_series("transient_T");
    REQUIRE(temperature_series.num_columns() ==
            static_cast<pycanha::Index>(node_numbers.size()));
}

TEST_CASE("read_tmd_transient can load only requested attributes",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "single",
        /*overwrite=*/false, {pycanha::TMDNodeAttribute::T});

    REQUIRE(!node_numbers.empty());
    REQUIRE(thermal_data.has_dense_time_series("single_T"));
    REQUIRE(!thermal_data.has_dense_time_series("single_C"));
    REQUIRE(!thermal_data.has_dense_time_series("single_QA"));
    REQUIRE(!thermal_data.has_dense_time_series("single_QE"));
    REQUIRE(!thermal_data.has_dense_time_series("single_QI"));
    REQUIRE(!thermal_data.has_dense_time_series("single_QR"));
    REQUIRE(!thermal_data.has_dense_time_series("single_QS"));
}

TEST_CASE("read_tmd_transient throws before writing when overwrite is disabled",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    auto& existing_series = thermal_data.add_dense_time_series("case_QS", 1, 1);
    existing_series.times()(0) = 42.0;
    existing_series.values()(0, 0) = 99.0;

    REQUIRE_THROWS_AS(pycanha::read_tmd_transient(reference_tmd_path.string(),
                                                  thermal_data, "case"),
                      std::runtime_error);

    REQUIRE_FALSE(thermal_data.has_dense_time_series("case_T"));
    REQUIRE_FALSE(thermal_data.has_dense_time_series("case_C"));
    REQUIRE(thermal_data.get_dense_time_series("case_QS").num_timesteps() == 1);
    REQUIRE(thermal_data.get_dense_time_series("case_QS").num_columns() == 1);
    REQUIRE(thermal_data.get_dense_time_series("case_QS").times()(0) ==
            Catch::Approx(42.0));
    REQUIRE(thermal_data.get_dense_time_series("case_QS").values()(0, 0) ==
            Catch::Approx(99.0));
}

TEST_CASE("read_tmd_transient overwrites existing series when requested",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    thermal_data.add_dense_time_series("case_QS", 1, 1);

    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "case",
        /*overwrite=*/true);

    REQUIRE(thermal_data.has_dense_time_series("case_T"));
    REQUIRE(thermal_data.has_dense_time_series("case_QS"));

    const auto& overwritten_series =
        thermal_data.get_dense_time_series("case_QS");
    REQUIRE(overwritten_series.num_timesteps() > 1);
    REQUIRE(overwritten_series.num_columns() ==
            static_cast<pycanha::Index>(node_numbers.size()));
}
