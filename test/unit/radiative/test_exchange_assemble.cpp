// White-box tests of the exchange assembly. They drive assemble_exchange
// with hand-built fixed-point cell matrices instead of a traced scene, so
// every semantic the result depends on — what a stored factor means, which
// columns carry the closure accounting, when an entry is dropped — is
// checked exactly and without a GPU.
//
// Ray counts and the fixed-point scale are powers of two throughout, which
// makes every expected value exactly representable and lets these be
// equality checks rather than tolerance checks.

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "radiative/exchange_assemble.hpp"
#include "radiative/vf_assemble.hpp"

namespace rad = pycanha::radiative;
namespace detail = pycanha::radiative::detail;

namespace {

// The power of two an accumulator would have fixed on its first batch. One
// deposit unit is 1/fp_scale of a ray's energy.
constexpr double fp_scale = 1024.0;

// A fixed-point cell matrix in both layouts at once: the dense cell block
// the GPU buffer would hold, and the per-row maps the tiled path accumulates
// into. Writing through set() keeps the two in step, which is what makes the
// Dense/Tiled comparison meaningful.
class Cells {
  public:
    explicit Cells(std::size_t slots)
        : _slots(slots),
          _cols(slots + static_cast<std::size_t>(rad::num_virtual_columns)),
          _rays(slots, 0),
          _cells(slots * _cols, 0),
          _rows(slots) {}

    void set(std::size_t row, std::size_t column, std::uint64_t cell) {
        _cells[(row * _cols) + column] = cell;
        if (cell == 0) {
            _rows[row].erase(static_cast<std::uint32_t>(column));
        } else {
            _rows[row][static_cast<std::uint32_t>(column)] = cell;
        }
    }

    // Adds a map entry the dense block does not have. This is how a cell
    // that accumulated back to zero reaches the tiled path: absorb_block
    // only ever inserts nonzero cells, but the wrapping += that follows can
    // land on zero and the entry stays.
    void set_tiled_only(std::size_t row, std::size_t column,
                        std::uint64_t cell) {
        _rows[row][static_cast<std::uint32_t>(column)] = cell;
    }

    void set_rays(std::size_t slot, std::uint64_t rays) { _rays[slot] = rays; }

    [[nodiscard]] std::size_t slots() const { return _slots; }
    [[nodiscard]] std::size_t space_column() const { return _slots; }
    [[nodiscard]] std::size_t inactive_column() const {
        return _slots + static_cast<std::size_t>(rad::inactive_column_offset);
    }
    [[nodiscard]] std::size_t lost_column() const {
        return _slots + static_cast<std::size_t>(rad::lost_column_offset);
    }

    [[nodiscard]] rad::ExchangeResult dense(
        const rad::AccumConfig& config) const {
        return detail::assemble_exchange(std::span<const std::uint64_t>(_cells),
                                         _rays, fp_scale, rad::Band::IR,
                                         config);
    }

    [[nodiscard]] rad::ExchangeResult tiled(
        const rad::AccumConfig& config) const {
        return detail::assemble_exchange(
            std::span<const detail::HostCountRow>(_rows), _rays, fp_scale,
            rad::Band::IR, config);
    }

    [[nodiscard]] std::uint64_t dense_error() const {
        return detail::exchange_conservation_error(
            std::span<const std::uint64_t>(_cells), _rays, fp_scale);
    }

    [[nodiscard]] std::uint64_t tiled_error() const {
        return detail::exchange_conservation_error(
            std::span<const detail::HostCountRow>(_rows), _rays, fp_scale);
    }

  private:
    std::size_t _slots;
    std::size_t _cols;
    std::vector<std::uint64_t> _rays;
    std::vector<std::uint64_t> _cells;
    std::vector<detail::HostCountRow> _rows;
};

[[nodiscard]] double value_at(const rad::SparseMatrix& matrix, Eigen::Index row,
                              Eigen::Index column) {
    for (rad::SparseMatrix::InnerIterator entry(matrix, row); entry; ++entry) {
        if (entry.col() == column) {
            return entry.value();
        }
    }
    return 0.0;
}

[[nodiscard]] bool same_entries(const rad::SparseMatrix& lhs,
                                const rad::SparseMatrix& rhs) {
    using Iterator = rad::SparseMatrix::InnerIterator;
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols() ||
        lhs.nonZeros() != rhs.nonZeros()) {
        return false;
    }
    for (Eigen::Index row = 0; row < lhs.rows(); ++row) {
        Iterator left(lhs, row);
        Iterator right(rhs, row);
        for (; left && right; ++left, ++right) {
            if (left.col() != right.col() || left.value() != right.value()) {
                return false;
            }
        }
        if (left || right) {
            return false;
        }
    }
    return true;
}

// Two emitting rows and one that never emits. Every emitting row balances
// exactly — its cells, buckets included, sum to the energy it emitted — and
// row 1 carries a NEGATIVE lost cell, the Russian-roulette withdrawal that
// makes that column signed.
[[nodiscard]] Cells make_scene() {
    Cells cells(3);
    cells.set_rays(0, 1024);
    cells.set_rays(1, 1024);
    cells.set_rays(2, 0);  // never emits

    cells.set(0, 1, 262144);                     // factor 0.25
    cells.set(0, 2, 131072);                     // factor 0.125
    cells.set(0, cells.space_column(), 655360);  // factor 0.625

    cells.set(1, 0, 524288);                        // factor 0.5
    cells.set(1, cells.inactive_column(), 262144);  // factor 0.25
    cells.set(1, cells.lost_column(),
              std::uint64_t{0} - 131072);        // factor -0.125
    cells.set(1, cells.space_column(), 393216);  // factor 0.375
    return cells;
}

}  // namespace

TEST_CASE("radiative exchange assemble: a stored factor is the deposit share",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    const rad::ExchangeResult result = cells.dense(rad::AccumConfig{});

    REQUIRE(result.band == rad::Band::IR);
    REQUIRE(result.factors.rows() == static_cast<Eigen::Index>(cells.slots()));
    REQUIRE(result.factors.cols() == static_cast<Eigen::Index>(cells.slots()) +
                                         rad::num_virtual_columns);

    // Deposit / (fp_scale * rays), exactly, over BOTH triangles: the
    // exchange factor is intensive and asymmetric, so nothing is folded.
    REQUIRE(value_at(result.factors, 0, 1) == 0.25);
    REQUIRE(value_at(result.factors, 0, 2) == 0.125);
    REQUIRE(value_at(result.factors, 1, 0) == 0.5);
    REQUIRE(value_at(result.factors, 0,
                     static_cast<Eigen::Index>(cells.space_column())) == 0.625);
    REQUIRE(value_at(result.factors, 1,
                     static_cast<Eigen::Index>(cells.inactive_column())) ==
            0.25);

    // A row that emitted nothing stores nothing: without an emitted energy
    // to divide by there is no factor to report.
    REQUIRE_FALSE(rad::SparseMatrix::InnerIterator(result.factors, 2));
}

TEST_CASE("radiative exchange assemble: the lost column is signed",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    const rad::ExchangeResult result = cells.dense(rad::AccumConfig{});

    // Russian-roulette withdrawals wrap the cell, so it must be read back as
    // a signed integer; read as unsigned it would come out astronomically
    // large instead of slightly negative.
    REQUIRE(value_at(result.factors, 1,
                     static_cast<Eigen::Index>(cells.lost_column())) == -0.125);
    // -128 deposit units lost against 2048 rays emitted across the scene.
    REQUIRE(result.stats.lost_energy_fraction == -0.0625);
}

TEST_CASE("radiative exchange assemble: the layouts are bit-identical",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    const rad::ExchangeResult dense = cells.dense(rad::AccumConfig{});
    const rad::ExchangeResult tiled = cells.tiled(rad::AccumConfig{});

    REQUIRE(same_entries(tiled.factors, dense.factors));
    REQUIRE(tiled.stats.mean_stderr == dense.stats.mean_stderr);
    REQUIRE(tiled.stats.max_stderr == dense.stats.max_stderr);
    REQUIRE(tiled.stats.lost_energy_fraction ==
            dense.stats.lost_energy_fraction);
}

TEST_CASE("radiative exchange assemble: a cell that wrapped to zero is skipped",
          "[radiative][exchange][assemble]") {
    Cells cells = make_scene();
    // Row 0 never deposited on itself, so the dense block holds a plain zero
    // there and skips it. The tiled map still holds the key, so the sparse
    // walk has to skip it too or the two layouts stop agreeing.
    cells.set_tiled_only(0, 0, 0);

    REQUIRE(same_entries(cells.tiled(rad::AccumConfig{}).factors,
                         cells.dense(rad::AccumConfig{}).factors));
    REQUIRE(cells.tiled_error() == cells.dense_error());
}

TEST_CASE("radiative exchange assemble: thresholding leaves the statistics",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    const rad::ExchangeResult kept = cells.dense(rad::AccumConfig{});
    rad::AccumConfig pruned_config;
    pruned_config.sparse_threshold = 0.2;
    const rad::ExchangeResult pruned = cells.dense(pruned_config);

    // The threshold prunes only what is stored. Everything the row reports —
    // the standard errors and the lost-energy accounting — is computed
    // before it, so the accounting stays exact however much is dropped.
    REQUIRE(pruned.factors.nonZeros() < kept.factors.nonZeros());
    REQUIRE(pruned.stats.mean_stderr == kept.stats.mean_stderr);
    REQUIRE(pruned.stats.max_stderr == kept.stats.max_stderr);
    REQUIRE(pruned.stats.lost_energy_fraction ==
            kept.stats.lost_energy_fraction);
    REQUIRE(value_at(pruned.factors, 0, 1) == 0.25);
    REQUIRE(value_at(pruned.factors, 0, 2) == 0.0);

    // The comparison is on the magnitude, which is the only reading that
    // makes sense for a column that can legitimately go negative: -0.125
    // goes at a threshold of 0.2 and stays at one of 0.05.
    const auto lost = static_cast<Eigen::Index>(cells.lost_column());
    REQUIRE(value_at(pruned.factors, 1, lost) == 0.0);
    rad::AccumConfig fine_config;
    fine_config.sparse_threshold = 0.05;
    REQUIRE(value_at(cells.dense(fine_config).factors, 1, lost) == -0.125);
}

TEST_CASE("radiative exchange assemble: energy conservation is exact",
          "[radiative][exchange][assemble]") {
    Cells cells = make_scene();
    REQUIRE(cells.dense_error() == 0);
    REQUIRE(cells.tiled_error() == 0);

    // A row whose cells no longer add up to the energy it emitted is what
    // the check exists to catch, in either direction of the wrap.
    cells.set(0, 1, 262144 + 64);
    REQUIRE(cells.dense_error() == 64);
    REQUIRE(cells.tiled_error() == 64);
    cells.set(0, 1, 262144 - 64);
    REQUIRE(cells.dense_error() == 64);
    REQUIRE(cells.tiled_error() == 64);
}

TEST_CASE("radiative exchange assemble: nothing traced yields an empty result",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    const std::vector<std::uint64_t> rays(cells.slots(), 0);
    const std::vector<std::uint64_t> flat(
        cells.slots() * (cells.slots() +
                         static_cast<std::size_t>(rad::num_virtual_columns)),
        0);
    // fp_scale 0 is an accumulator that never ran a batch: no scale was ever
    // chosen, so there is nothing to divide by and nothing to report.
    const rad::ExchangeResult result =
        detail::assemble_exchange(std::span<const std::uint64_t>(flat), rays,
                                  0.0, rad::Band::Solar, rad::AccumConfig{});
    REQUIRE(result.band == rad::Band::Solar);
    REQUIRE(result.factors.nonZeros() == 0);
    REQUIRE(result.stats.lost_energy_fraction == 0.0);
    REQUIRE(detail::exchange_conservation_error(
                std::span<const std::uint64_t>(flat), rays, 0.0) == 0);
}

TEST_CASE("radiative exchange assemble: a mismatched cell buffer is rejected",
          "[radiative][exchange][assemble]") {
    const std::vector<std::uint64_t> rays(3, 1024);
    const std::vector<std::uint64_t> too_small(3 * 3, 0);
    REQUIRE_THROWS_AS(detail::assemble_exchange(
                          std::span<const std::uint64_t>(too_small), rays,
                          fp_scale, rad::Band::IR, rad::AccumConfig{}),
                      std::invalid_argument);

    const std::vector<detail::HostCountRow> too_few(2);
    REQUIRE_THROWS_AS(detail::assemble_exchange(
                          std::span<const detail::HostCountRow>(too_few), rays,
                          fp_scale, rad::Band::IR, rad::AccumConfig{}),
                      std::invalid_argument);
}
