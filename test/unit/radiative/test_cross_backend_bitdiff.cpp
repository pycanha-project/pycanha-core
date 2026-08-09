// Cross-backend bit-diff harness. HIDDEN (the [.] tag): it asserts nothing on
// its own and is not part of a normal ctest run — it dumps results so the SAME
// fixed-seed case can be compared between two machines running different
// backends:
//
//     ./tests "[bitdiff]"        # on each machine, from the same directory
//     diff bitdiff_vf.txt bitdiff_exchange.txt   # between the two dumps
//
// The files must match BYTE FOR BYTE. That is not a floating-point tolerance
// claim: both backends run the same kernel sources, the deposits are integer
// atomics (order-independent by construction), and the host normalization is
// the same code, so any difference at all is a real bug in a backend — a
// residency mistake, a wrong buffer index, a broken split-counter carry — not
// numerical noise. Values are written as the raw bit pattern of the double for
// exactly that reason; decimal formatting would hide the last few bits.
//
// Nothing about the machine (device name, backend, timings) goes into the
// files, so a matching pair is a genuine comparison of results only.

#include <array>
#include <bit>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <string>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "scene_fixtures.hpp"

namespace rad = pycanha::radiative;
using radiative_fixtures::gray_row;
using radiative_fixtures::make_box_enclosure;
using radiative_fixtures::make_materials;

namespace {

// One line per stored entry: row, column, and the exact bits of the value.
void dump_csr(const std::string& path, const rad::SparseMatrix& matrix) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "rows " << matrix.rows() << " cols " << matrix.cols() << "\n";
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (rad::SparseMatrix::InnerIterator entry(matrix, row); entry;
             ++entry) {
            out << row << ' ' << entry.col() << ' ' << std::hex << std::setw(16)
                << std::setfill('0')
                << std::bit_cast<std::uint64_t>(entry.value()) << std::dec
                << '\n';
        }
    }
}

}  // namespace

TEST_CASE("radiative cross-backend: fixed-seed result dump",
          "[.][bitdiff][radiative][gpu]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: nothing to dump");
        return;
    }
    rad::Device device = rad::Device::create();
    WARN("dumping from device: " << device.info().name);

    const auto model = make_box_enclosure();
    const std::array<int, 6> pair_rows{0, 0, 0, 0, 0, 0};

    // View factors: pure geometry, so a difference here is traversal or
    // emission sampling.
    {
        const std::array<std::array<float, 6>, 1> rows{gray_row(1.0F)};
        rad::RadiativeScene scene(device, model->mesh_parts(),
                                  make_materials(rows, pair_rows));
        rad::TraceSettings settings;
        settings.rays_per_face = 5'000;
        settings.seed = 20250731;
        rad::VfAccumulator acc(scene);
        scene.accumulate_vf(acc, settings);
        dump_csr("bitdiff_vf.txt", acc.result().vf);
    }

    // Exchange with a partly reflective surface: multi-bounce paths, Russian
    // roulette and the fixed-point deposit — the path that exercises the
    // 64-bit accumulator cells (a lo/hi split pair on Metal).
    {
        const std::array<std::array<float, 6>, 1> rows{gray_row(0.4F)};
        rad::RadiativeScene scene(device, model->mesh_parts(),
                                  make_materials(rows, pair_rows));
        rad::TraceSettings settings;
        settings.rays_per_face = 5'000;
        settings.seed = 20250731;
        rad::ExchangeAccumulator acc(scene, rad::Band::IR);
        scene.accumulate_exchange(acc, settings);
        dump_csr("bitdiff_exchange.txt", acc.result().factors);
        // Zero here on both machines is the other half of the check: the row
        // balances close exactly, so no deposit was lost on either backend.
        WARN("exchange conservation error: " << acc.conservation_error());
    }
}
