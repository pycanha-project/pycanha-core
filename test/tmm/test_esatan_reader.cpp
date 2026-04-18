#include <H5Cpp.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>  // NOLINT(misc-include-cleaner)
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/io/esatan.hpp"
#include "pycanha-core/solvers/sslu.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
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

std::unordered_map<pycanha::Index, pycanha::Index> build_column_lookup(
    const std::vector<pycanha::Index>& node_numbers) {
    std::unordered_map<pycanha::Index, pycanha::Index> lookup;
    lookup.reserve(node_numbers.size());

    for (std::size_t i = 0; i < node_numbers.size(); ++i) {
        lookup.emplace(node_numbers[i], static_cast<pycanha::Index>(i));
    }

    return lookup;
}

double read_tabs_attribute(const std::filesystem::path& filepath) {
    const H5::H5File tmd_file(filepath.string(), H5F_ACC_RDONLY);
    const H5::Group analysis_group = tmd_file.openGroup("AnalysisSet1");
    H5::Attribute tabs_attribute = analysis_group.openAttribute("TAbs");

    double tabs = 0.0;
    tabs_attribute.read(H5::PredType::NATIVE_DOUBLE, &tabs);
    return tabs;
}

double read_first_raw_temperature(const std::filesystem::path& filepath,
                                  pycanha::Index column_index) {
    const H5::H5File tmd_file(filepath.string(), H5F_ACC_RDONLY);
    const H5::Group data_group =
        tmd_file.openGroup("AnalysisSet1").openGroup("DataGroup1");
    H5::DataSet node_real_data_dataset =
        data_group.openDataSet("thermalNodesRealData");

    const std::array<hsize_t, 3> count{1, 1, 1};
    const std::array<hsize_t, 3> offset{0, static_cast<hsize_t>(column_index),
                                        0};

    const H5::DataSpace file_space = node_real_data_dataset.getSpace();
    file_space.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());

    const hsize_t num_values = 1;
    const H5::DataSpace mem_space(1, &num_values);

    double value = 0.0;
    node_real_data_dataset.read(&value, H5::PredType::NATIVE_DOUBLE, mem_space,
                                file_space);
    return value;
}

void require_monotonic_times(const pycanha::DenseTimeSeries& series) {
    for (pycanha::Index i = 1; i < series.num_timesteps(); ++i) {
        REQUIRE(series.times()(i) >= series.times()(i - 1));
    }
}

void require_default_transient_series(
    const pycanha::DataModel& model,
    const std::vector<pycanha::Index>& node_numbers) {
    for (const auto attribute :
         {pycanha::DataModelAttribute::T, pycanha::DataModelAttribute::C,
          pycanha::DataModelAttribute::QA, pycanha::DataModelAttribute::QE,
          pycanha::DataModelAttribute::QI, pycanha::DataModelAttribute::QR,
          pycanha::DataModelAttribute::QS}) {
        const auto& series = model.get_dense_attribute(attribute);
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
    REQUIRE(thermal_data.models().has_model("transient"));
    require_default_transient_series(
        thermal_data.models().get_model("transient"), node_numbers);
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
        thermal_data.models().get_model("transient").T();

    for (pycanha::Index i = 0; i < model->nodes().get_num_nodes(); ++i) {
        pycanha::Node node = model->nodes().get_node_from_idx(i);
        const auto iterator = column_lookup.find(node.get_node_num());
        REQUIRE(iterator != column_lookup.end());
        REQUIRE(temperature_series.values()(0, iterator->second) ==
                Catch::Approx(node.get_T()));
    }
}

TEST_CASE("read_tmd_transient uses TAbs for temperature conversion",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "transient");
    const auto& temperature_series =
        thermal_data.models().get_model("transient").T();

    REQUIRE_FALSE(node_numbers.empty());

    const double tabs = read_tabs_attribute(reference_tmd_path);
    const double raw_temperature =
        read_first_raw_temperature(reference_tmd_path, 0);
    REQUIRE(temperature_series.values()(0, 0) ==
            Catch::Approx(raw_temperature + tabs));
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
        thermal_data.models().get_model("transient").T();
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
        /*overwrite=*/false, {pycanha::DataModelAttribute::T});

    REQUIRE(!node_numbers.empty());
    const auto& model_data = thermal_data.models().get_model("single");
    REQUIRE(model_data.T().num_timesteps() > 0);
    REQUIRE(model_data.C().num_timesteps() == 0);
    REQUIRE(model_data.QA().num_timesteps() == 0);
    REQUIRE(model_data.QE().num_timesteps() == 0);
    REQUIRE(model_data.QI().num_timesteps() == 0);
    REQUIRE(model_data.QR().num_timesteps() == 0);
    REQUIRE(model_data.QS().num_timesteps() == 0);
}

TEST_CASE("read_tmd_transient rejects unsupported JAC attribute",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    REQUIRE_THROWS_AS(
        pycanha::read_tmd_transient(
            reference_tmd_path.string(), thermal_data, "jac",
            /*overwrite=*/false, {pycanha::DataModelAttribute::JAC}),
        std::invalid_argument);
}

TEST_CASE("read_tmd_transient throws before writing when overwrite is disabled",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    pycanha::DataModel existing_model({1});
    existing_model.QS().resize(1, 1);
    existing_model.QS().times()(0) = 42.0;
    existing_model.QS().values()(0, 0) = 99.0;
    thermal_data.models().add_model("case", std::move(existing_model));

    REQUIRE_THROWS_AS(pycanha::read_tmd_transient(reference_tmd_path.string(),
                                                  thermal_data, "case"),
                      std::runtime_error);

    const auto& model_data = thermal_data.models().get_model("case");
    REQUIRE(model_data.T().num_timesteps() == 0);
    REQUIRE(model_data.C().num_timesteps() == 0);
    REQUIRE(model_data.QS().num_timesteps() == 1);
    REQUIRE(model_data.QS().num_columns() == 1);
    REQUIRE(model_data.QS().times()(0) == Catch::Approx(42.0));
    REQUIRE(model_data.QS().values()(0, 0) == Catch::Approx(99.0));
}

TEST_CASE("read_tmd_transient overwrites existing series when requested",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    pycanha::DataModel existing_model({1});
    existing_model.QS().resize(1, 1);
    thermal_data.models().add_model("case", std::move(existing_model));

    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "case",
        /*overwrite=*/true);

    REQUIRE(thermal_data.models().has_model("case"));

    const auto& overwritten_series =
        thermal_data.models().get_model("case").QS();
    REQUIRE(overwritten_series.num_timesteps() > 1);
    REQUIRE(overwritten_series.num_columns() ==
            static_cast<pycanha::Index>(node_numbers.size()));
}
