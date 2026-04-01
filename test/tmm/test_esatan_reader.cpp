#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <memory>
#include <string>  // NOLINT(misc-include-cleaner)

#include "pycanha-core/io/esatan.hpp"
#include "pycanha-core/solvers/sslu.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace {

std::filesystem::path get_reference_tmd_path() {
    const std::filesystem::path this_file(__FILE__);
    const std::filesystem::path test_root =
        this_file.parent_path().parent_path();
    return test_root / "data" / "esatan" / "DISCTR_TRANSIENT.TMD";
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
