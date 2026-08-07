// Backend-agnostic exchange assembly (pure CPU; see exchange_assemble.hpp
// for why it does not live in either backend).
//
// One sequential pass over the accumulated cells: every stored deposit is
// divided by the energy its row emitted, which is what makes the result the
// intensive factor B_ij. The pass also carries the closure accounting — the
// signed lost column and the per-entry standard errors — all of which are
// computed BEFORE thresholding, so pruning tiny entries never corrupts them.

#include "exchange_assemble.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "csr_assembly.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vf_assemble.hpp"

namespace pycanha::radiative::detail {

namespace {

// Matrix row stride: the real face columns plus the virtual
// space/inactive/lost bucket columns.
[[nodiscard]] std::size_t matrix_columns(std::size_t slots) {
    return slots + static_cast<std::size_t>(num_virtual_columns);
}

// Magnitude of a wrapping-u64 difference (conservation residuals are exact
// zeros when the kernel is right; a broken kernel may miss in either
// direction).
[[nodiscard]] std::uint64_t wrap_magnitude(std::uint64_t difference) {
    return std::min(difference, std::uint64_t{0} - difference);
}

// Running per-entry statistics while scanning rows into a CSR.
struct EntryStats {
    double stderr_sum = 0.0;
    double stderr_max = 0.0;
    std::size_t entries = 0;
    double lost_energy = 0.0;  // signed: Russian-roulette adjustments
};

// The mapped GPU cell block, read in place. Unlike the view-factor assembly
// this pass touches every cell exactly once and in order, so there is
// nothing a bulk copy out of mapped memory would buy back.
struct DenseCells {
    std::span<const std::uint64_t> cells;
    std::size_t cols = 0;
};

// Row-major CSR of the tiled layout's per-row maps. Sorting each row once is
// linear in the stored cells; the alternative those hash maps invite — one
// lookup per column of every row — is quadratic in the slot count.
struct SparseCells {
    std::vector<std::size_t> row_start;  // slots + 1
    std::vector<std::uint32_t> column;
    std::vector<std::uint64_t> value;
};

[[nodiscard]] DenseCells dense_cells(std::span<const std::uint64_t> mapped,
                                     std::size_t slots) {
    DenseCells cells;
    cells.cols = matrix_columns(slots);
    if (mapped.size() != slots * cells.cols) {
        throw std::invalid_argument(
            "pycanha::radiative: the dense cell buffer does not match the "
            "face-slot count");
    }
    cells.cells = mapped;
    return cells;
}

[[nodiscard]] SparseCells sparse_cells(std::span<const HostCountRow> host_rows,
                                       std::size_t slots) {
    if (host_rows.size() != slots) {
        throw std::invalid_argument(
            "pycanha::radiative: the tiled cell rows do not match the "
            "face-slot count");
    }
    SparseCells cells;
    cells.row_start.assign(slots + 1, 0);
    for (std::size_t row = 0; row < slots; ++row) {
        cells.row_start[row + 1] = cells.row_start[row] + host_rows[row].size();
    }
    cells.column.resize(cells.row_start[slots]);
    cells.value.resize(cells.row_start[slots]);
    for (std::size_t row = 0; row < slots; ++row) {
        using Entry = std::pair<std::uint32_t, std::uint64_t>;
        std::vector<Entry> entries(host_rows[row].begin(),
                                   host_rows[row].end());
        std::ranges::sort(entries, {}, &Entry::first);
        std::size_t at = cells.row_start[row];
        for (const auto& [column, cell] : entries) {
            cells.column[at] = column;
            cells.value[at] = cell;
            ++at;
        }
    }
    return cells;
}

// Visits the nonzero cells of one row in ascending column order. Both
// layouts present the same order, which is what makes the CSR and every
// derived statistic bit-identical between them.
template <typename Visit>
void visit_row(const DenseCells& cells, std::size_t row, const Visit& visit) {
    const std::span<const std::uint64_t> line =
        cells.cells.subspan(row * cells.cols, cells.cols);
    for (std::size_t column = 0; column < line.size(); ++column) {
        if (line[column] != 0) {
            visit(column, line[column]);
        }
    }
}

template <typename Visit>
void visit_row(const SparseCells& cells, std::size_t row, const Visit& visit) {
    for (std::size_t at = cells.row_start[row]; at < cells.row_start[row + 1];
         ++at) {
        // A cell can accumulate back to zero (the lost column wraps), and the
        // dense walk skips those, so the sparse one must too.
        if (cells.value[at] != 0) {
            visit(static_cast<std::size_t>(cells.column[at]), cells.value[at]);
        }
    }
}

struct ScanInputs {
    std::size_t slots = 0;
    std::size_t lost_column = 0;
    double inv_scale = 0.0;
    double threshold = 0.0;
};

// The CSR being built, plus the statistics that accumulate alongside it.
// Rows are appended in order, so the arrays are a valid CSR image as they
// stand.
struct RowSink {
    std::vector<SparseIndex> indices;
    std::vector<double> values;
    EntryStats stats;
};

template <typename Cells>
void scan_row(const Cells& cells, std::size_t row, std::uint64_t rays_row,
              const ScanInputs& in, RowSink& out) {
    const auto rays = static_cast<double>(rays_row);
    visit_row(cells, row, [&](std::size_t column, std::uint64_t cell) {
        // The lost column is signed: Russian-roulette boost withdrawals may
        // push it (slightly) negative. The fixed-point value converts FIRST
        // (exact for full deposits), THEN divides by the rays — the same
        // rounding path as the vf counts, so blackbody exchange factors are
        // bit-identical to view factors.
        const double energy =
            column == in.lost_column
                ? static_cast<double>(static_cast<std::int64_t>(cell)) *
                      in.inv_scale
                : static_cast<double>(cell) * in.inv_scale;
        const double factor = energy / rays;
        if (column == in.lost_column) {
            out.stats.lost_energy += energy;
        }
        if (column < in.slots) {
            // Conservative per-entry standard error: a ray's deposit into
            // one cell is in [0, 1], so the Bernoulli bound dominates the
            // true variance.
            const double entry_stderr =
                std::sqrt(factor * std::max(1.0 - factor, 0.0) / rays);
            out.stats.stderr_sum += entry_stderr;
            out.stats.stderr_max = std::max(out.stats.stderr_max, entry_stderr);
            ++out.stats.entries;
        }
        // Row statistics come before thresholding; the threshold only prunes
        // what is stored.
        if (std::abs(factor) > in.threshold) {
            out.indices.push_back(static_cast<SparseIndex>(column));
            out.values.push_back(factor);
        }
    });
}

template <typename Cells>
[[nodiscard]] ExchangeResult build_result(
    const Cells& cells, std::span<const std::uint64_t> rays_per_row,
    double fp_scale, Band band, const AccumConfig& config) {
    const std::size_t slots = rays_per_row.size();
    const ScanInputs scan_in{
        .slots = slots,
        .lost_column = slots + static_cast<std::size_t>(lost_column_offset),
        .inv_scale = fp_scale > 0.0 ? 1.0 / fp_scale : 0.0,
        .threshold = config.sparse_threshold};

    std::vector<SparseIndex> row_starts;
    row_starts.reserve(slots + 1);
    RowSink sink;
    double emitted_energy = 0.0;

    row_starts.push_back(0);
    for (std::size_t row = 0; row < slots; ++row) {
        const std::uint64_t rays_row = rays_per_row[row];
        if (rays_row > 0 && fp_scale > 0.0) {
            scan_row(cells, row, rays_row, scan_in, sink);
            emitted_energy += static_cast<double>(rays_row);
        }
        check_sparse_capacity(sink.values.size());
        row_starts.push_back(static_cast<SparseIndex>(sink.values.size()));
    }

    ExchangeResult result;
    result.band = band;
    result.factors = make_csr(static_cast<Eigen::Index>(slots),
                              static_cast<Eigen::Index>(matrix_columns(slots)),
                              row_starts, sink.indices, sink.values);
    result.stats.mean_stderr =
        sink.stats.entries > 0
            ? sink.stats.stderr_sum / static_cast<double>(sink.stats.entries)
            : 0.0;
    result.stats.max_stderr = sink.stats.stderr_max;
    result.stats.lost_energy_fraction =
        emitted_energy > 0.0 ? sink.stats.lost_energy / emitted_energy : 0.0;
    return result;
}

template <typename Cells>
[[nodiscard]] std::uint64_t row_balance_error(
    const Cells& cells, std::span<const std::uint64_t> rays_per_row,
    std::uint64_t scale) {
    std::uint64_t max_error = 0;
    for (std::size_t row = 0; row < rays_per_row.size(); ++row) {
        if (rays_per_row[row] == 0) {
            continue;
        }
        // Everything wraps mod 2^64 — the identity the kernel maintains. The
        // virtual columns are part of the balance, so the full row accounts
        // for every parcel of emitted energy.
        std::uint64_t balance = 0;
        visit_row(cells, row,
                  [&balance](std::size_t /*column*/, std::uint64_t cell) {
                      balance += cell;
                  });
        const std::uint64_t expected = rays_per_row[row] * scale;
        max_error = std::max(max_error, wrap_magnitude(balance - expected));
    }
    return max_error;
}

}  // namespace

ExchangeResult assemble_exchange(ExchangeCellSource cells,
                                 std::span<const std::uint64_t> rays_per_row,
                                 double fp_scale, Band band,
                                 const AccumConfig& config) {
    const std::size_t slots = rays_per_row.size();
    if (const auto* mapped =
            std::get_if<std::span<const std::uint64_t>>(&cells)) {
        return build_result(dense_cells(*mapped, slots), rays_per_row, fp_scale,
                            band, config);
    }
    return build_result(
        sparse_cells(std::get<std::span<const HostCountRow>>(cells), slots),
        rays_per_row, fp_scale, band, config);
}

std::uint64_t exchange_conservation_error(
    ExchangeCellSource cells, std::span<const std::uint64_t> rays_per_row,
    double fp_scale) {
    if (fp_scale <= 0.0) {
        return 0;
    }
    const std::size_t slots = rays_per_row.size();
    const auto scale = static_cast<std::uint64_t>(fp_scale);
    if (const auto* mapped =
            std::get_if<std::span<const std::uint64_t>>(&cells)) {
        return row_balance_error(dense_cells(*mapped, slots), rays_per_row,
                                 scale);
    }
    return row_balance_error(
        sparse_cells(std::get<std::span<const HostCountRow>>(cells), slots),
        rays_per_row, scale);
}

}  // namespace pycanha::radiative::detail
