// Backend-agnostic VF assembly (pure CPU; see vf_assemble.hpp for why it
// does not live in either backend).
//
// The pass is O(Nf^2) over a matrix that reaches ~1e5 slots on a side, so it
// is written to stay memory-bound: the per-pair arithmetic is two loads, a
// table lookup and a pair of multiply-adds; the transposed reads go through
// a cache-blocked tile-pair walk instead of striding a full row per cell;
// and the sparse layout merges two sorted rows instead of hashing every
// column of every row.
//
// Everything that accumulates a double does so in a fixed order — one whole
// row or one whole row-tile per worker, each writing its own output range —
// so the worker count and the schedule cannot change a bit of the result.
// The only reductions that cross workers are maxima, which are
// order-independent anyway.

#include "vf_assemble.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "csr_assembly.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative::detail {

namespace {

// Nodes across m in [1, 2] for the tabulated m^n.
constexpr std::size_t mantissa_nodes = 1U << 12U;
// Smallest binary exponent a positive double can carry (the smallest
// subnormal is 2^-1074).
constexpr int min_binary_exponent = -1074;
// Significand bits kept in the tables (see the WeightTable comment).
constexpr int table_significand_bits = 24;

// Rounds to table_significand_bits using only exactly-representable
// operations, so the rounded value cannot depend on the platform.
[[nodiscard]] double quantize(double value) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        return value;
    }
    int exponent = 0;
    const double fraction = std::frexp(value, &exponent);
    const double scaled = std::ldexp(fraction, table_significand_bits);
    return std::ldexp(std::round(scaled), exponent - table_significand_bits);
}

}  // namespace

WeightTable::WeightTable(double exponent) : _exponent(exponent) {
    if (!std::isfinite(exponent) || exponent <= 0.0) {
        throw std::invalid_argument(
            "pycanha::radiative: the triangulation exponent must be finite "
            "and greater than zero");
    }
    _mantissa.resize(mantissa_nodes + 1);
    for (std::size_t node = 0; node <= mantissa_nodes; ++node) {
        const double mantissa = 1.0 + (static_cast<double>(node) /
                                       static_cast<double>(mantissa_nodes));
        _mantissa[node] = quantize(std::pow(mantissa, exponent));
    }
    _scale.resize(static_cast<std::size_t>(-min_binary_exponent) + 1);
    for (int binary = min_binary_exponent; binary <= 0; ++binary) {
        _scale[static_cast<std::size_t>(binary - min_binary_exponent)] =
            quantize(std::pow(2.0, exponent * static_cast<double>(binary)));
    }
}

double WeightTable::unit_pow(double t) const noexcept {
    if (!(t > 0.0)) {
        return 0.0;
    }
    if (t >= 1.0) {
        return 1.0;
    }
    if (_exponent == 1.0) {
        // Exact, and it is what makes exponent 1 reduce to inverse-variance
        // weighting itself rather than to an interpolation of it.
        return t;
    }
    int exponent = 0;
    const double fraction = std::frexp(t, &exponent);  // in [0.5, 1)
    const double mantissa = fraction * 2.0;            // in [1, 2)
    const double position =
        (mantissa - 1.0) * static_cast<double>(mantissa_nodes);
    const auto node = static_cast<std::size_t>(position);
    const double offset = position - static_cast<double>(node);
    const double lower = _mantissa[node];
    const double step = _mantissa[node + 1] - lower;
    const double mantissa_pow = lower + (offset * step);
    // t = mantissa * 2^(exponent - 1), so the binary factor is 2^(n(e - 1)).
    return mantissa_pow *
           _scale[static_cast<std::size_t>(exponent - 1 - min_binary_exponent)];
}

double WeightTable::forward_weight(double u, double v) const noexcept {
    const double sum = u + v;
    if (!(sum > 0.0)) {
        return 0.5;  // no sampling either way: nothing to prefer
    }
    const double ratio = (v - u) / sum;
    const double magnitude = unit_pow(std::abs(ratio));
    return 0.5 * (1.0 + (ratio < 0.0 ? -magnitude : magnitude));
}

double WeightTable::naive_forward_weight(double u, double v,
                                         double exponent) noexcept {
    const double sum = u + v;
    if (!(sum > 0.0)) {
        return 0.5;
    }
    const double ratio = (v - u) / sum;
    const double magnitude = std::pow(std::abs(ratio), exponent);
    return 0.5 * (1.0 + (ratio < 0.0 ? -magnitude : magnitude));
}

namespace {

// Matrix row stride: the real face columns plus the virtual
// space/inactive/lost bucket columns.
[[nodiscard]] std::size_t matrix_columns(std::size_t slots) {
    return slots + static_cast<std::size_t>(num_virtual_columns);
}

// The chosen combination rule, resolved once so the pair loop only ever
// sees a weight lookup.
class Weighting {
  public:
    Weighting(const TriangulationConfig& config, const AssemblyTuning& tuning)
        : _mode(config.mode),
          _exponent(config.exponent),
          _tabulated(tuning.tabulated_weight),
          _table(config.exponent) {}

    [[nodiscard]] TriangulationMode mode() const noexcept { return _mode; }

    [[nodiscard]] double forward_weight(double u, double v) const noexcept {
        if (_tabulated) {
            return _table.forward_weight(u, v);
        }
        return WeightTable::naive_forward_weight(u, v, _exponent);
    }

  private:
    TriangulationMode _mode;
    double _exponent;
    bool _tabulated;
    // Built unconditionally: a few thousand pow calls cost microseconds, and
    // constructing it up front also validates the exponent for every mode
    // rather than only for the ones that go on to use it.
    WeightTable _table;
};

// Per-slot precomputation. A_i/N_i is at once the variance proxy the weight
// is built from and the factor that turns a raw count into the extensive
// A_i * count / N_i, so hoisting it out of the pair loop removes a multiply
// and a divide from every cell and leaves the weight a function of two
// scalars that do not vary along the inner loop.
struct SlotWeights {
    // A_i / N_i, and 0 for a row that emitted nothing — which stands in for
    // the infinite ratio that limit implies, since every use of it either
    // multiplies a count that is itself zero or is guarded explicitly.
    std::vector<double> ratio;
};

[[nodiscard]] SlotWeights slot_weights(
    std::span<const double> areas,
    std::span<const std::uint64_t> rays_per_row) {
    SlotWeights weights;
    weights.ratio.assign(areas.size(), 0.0);
    for (std::size_t slot = 0; slot < areas.size(); ++slot) {
        if (rays_per_row[slot] > 0) {
            weights.ratio[slot] =
                areas[slot] / static_cast<double>(rays_per_row[slot]);
        }
    }
    return weights;
}

// One stored matrix row, built in ascending column order.
struct RowEntries {
    std::vector<SparseIndex> columns;
    std::vector<double> values;

    void push(std::size_t column, double value) {
        columns.push_back(static_cast<SparseIndex>(column));
        values.push_back(value);
    }
};

// Per-row scan products, reduced afterwards in row order.
struct RowStats {
    double stderr_sum = 0.0;
    double stderr_max = 0.0;
    std::size_t entries = 0;
};

// The combined value of one face pair, with the disagreement of the two raw
// estimates measured BEFORE they are combined.
struct Combined {
    double value = 0.0;
    double residual = 0.0;
};

[[nodiscard]] Combined combine_pair(double forward_ratio, double backward_ratio,
                                    std::uint64_t forward_count,
                                    std::uint64_t backward_count,
                                    const Weighting& weighting) {
    const double forward = forward_ratio * static_cast<double>(forward_count);
    const double backward =
        backward_ratio * static_cast<double>(backward_count);
    const bool has_forward = forward_ratio > 0.0;
    const bool has_backward = backward_ratio > 0.0;
    // The residual is measured on the raw estimates, before they are
    // combined, and only where both directions actually carry samples. It
    // is the winding/parity check: taken after combining it would be zero
    // by construction and would detect nothing.
    const double larger = std::max(forward, backward);
    const double residual = (has_forward && has_backward && larger > 0.0)
                                ? std::abs(forward - backward) / larger
                                : 0.0;
    if (weighting.mode() == TriangulationMode::None) {
        return {.value = forward, .residual = residual};
    }
    // A row that emitted nothing has an infinite A/N, so Y -> -1 and the
    // weight passes entirely to the direction that does have data. The
    // degenerate limit is the right answer; only the arithmetic needs
    // guarding, which is why the ratio is stored as zero there.
    if (!has_forward) {
        return {.value = backward, .residual = residual};
    }
    if (!has_backward) {
        return {.value = forward, .residual = residual};
    }
    const double weight =
        weighting.forward_weight(forward_ratio, backward_ratio);
    return {.value = (weight * forward) + ((1.0 - weight) * backward),
            .residual = residual};
}

// Thresholding compares the INTENSIVE max(F_ij, F_ji) = G / min(A_i, A_j),
// so the knob keeps its units and a pair goes only when both directions are
// negligible. A slot with no area carries no geometry and cannot be hit, so
// there the guard keeps whatever arrived rather than dividing by zero.
[[nodiscard]] bool keep_pair(double value, double area_i, double area_j,
                             double threshold) {
    if (value == 0.0) {
        return false;
    }
    const double smaller = std::min(area_i, area_j);
    return !(smaller > 0.0) || (value / smaller) > threshold;
}

// Runs `body(index)` for every index in [0, count) across `threads` workers,
// handing indices out dynamically because a row-tile's work shrinks as its
// index grows. Every work item writes only its own output range, so which
// worker takes which index cannot change the result.
template <typename Body>
void parallel_for_index(std::size_t count, unsigned threads, const Body& body) {
    if (count == 0) {
        return;
    }
    if (threads <= 1) {
        for (std::size_t index = 0; index < count; ++index) {
            body(index);
        }
        return;
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::jthread> workers;
    workers.reserve(threads);
    for (unsigned worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&] {
            for (std::size_t index = next.fetch_add(1); index < count;
                 index = next.fetch_add(1)) {
                body(index);
            }
        });
    }
}

// Threading a small matrix costs more than the pass itself, so the
// automatic choice stays serial until the cell count justifies it. An
// explicit request is always honoured — that is what lets a test pin the
// worker count and show the result does not depend on it.
[[nodiscard]] unsigned worker_count(const AssemblyTuning& tuning,
                                    std::size_t work_items, std::size_t cells) {
    constexpr std::size_t parallel_cell_threshold = 1U << 18U;
    unsigned threads = tuning.threads;
    if (threads == 0) {
        if (cells < parallel_cell_threshold) {
            return 1;
        }
        threads = std::thread::hardware_concurrency();
    }
    threads = std::max(threads, 1U);
    return static_cast<unsigned>(
        std::min<std::size_t>(threads, std::max<std::size_t>(work_items, 1)));
}

// Concatenates the per-row entries into one compressed matrix. The row
// offsets are known before anything is copied, so each row lands at a fixed
// offset and the copy parallelises without affecting the result. Building
// the CSR arrays directly avoids setFromTriplets, whose sort and per-entry
// allocation would dominate everything above it.
[[nodiscard]] SparseMatrix pack_rows(std::span<const RowEntries> rows,
                                     Eigen::Index cols, unsigned threads) {
    std::vector<SparseIndex> row_starts;
    row_starts.reserve(rows.size() + 1);
    row_starts.push_back(0);
    std::size_t total = 0;
    for (const RowEntries& row : rows) {
        total += row.columns.size();
        check_sparse_capacity(total);
        row_starts.push_back(static_cast<SparseIndex>(total));
    }
    std::vector<SparseIndex> columns(total);
    std::vector<double> values(total);
    parallel_for_index(rows.size(), threads, [&](std::size_t row) {
        const auto at = static_cast<std::ptrdiff_t>(row_starts[row]);
        std::ranges::copy(rows[row].columns, columns.begin() + at);
        std::ranges::copy(rows[row].values, values.begin() + at);
    });
    return make_csr(static_cast<Eigen::Index>(rows.size()), cols, row_starts,
                    columns, values);
}

// --- count normalization -------------------------------------------------

// Dense counts, copied out of the mapped GPU allocation. Reads from mapped
// device memory can traverse PCIe and may be uncached or write-combined,
// which is pathological for the transposed access triangulation needs; one
// bulk sequential copy — the pattern that hardware is good at — buys back
// every later read.
struct DenseCounts {
    std::vector<std::uint32_t> cells;
    std::size_t cols = 0;

    [[nodiscard]] std::span<const std::uint32_t> row(std::size_t index) const {
        return std::span<const std::uint32_t>(cells).subspan(index * cols,
                                                             cols);
    }
};

// Row-major CSR of the tiled layout's per-row maps, plus the transpose of
// its real-column block. Building the transpose once as a counting sort is
// linear in the stored entries; the alternative those hash maps invite —
// one lookup per column of every row — is quadratic in the slot count and
// was the largest single cost in the assembly this replaces.
struct SparseCounts {
    std::vector<std::size_t> row_start;  // slots + 1
    std::vector<std::uint32_t> column;
    std::vector<std::uint64_t> value;
    // Transpose of the real-column block only; bucket columns have no
    // transpose partner and are never triangulated. `transposed_row` holds
    // the SOURCE row of each entry, so transposed row i lists every j with
    // a nonzero count from j to i.
    std::vector<std::size_t> transposed_start;
    std::vector<std::uint32_t> transposed_row;
    std::vector<std::uint64_t> transposed_value;
};

[[nodiscard]] DenseCounts copy_dense(std::span<const std::uint32_t> mapped,
                                     std::size_t slots) {
    DenseCounts counts;
    counts.cols = matrix_columns(slots);
    if (mapped.size() != slots * counts.cols) {
        throw std::invalid_argument(
            "pycanha::radiative: the dense count buffer does not match the "
            "face-slot count");
    }
    counts.cells.assign(mapped.begin(), mapped.end());
    return counts;
}

void sort_sparse_rows(SparseCounts& counts,
                      std::span<const HostCountRow> host_rows,
                      unsigned threads) {
    const std::size_t slots = host_rows.size();
    counts.row_start.assign(slots + 1, 0);
    for (std::size_t row = 0; row < slots; ++row) {
        counts.row_start[row + 1] =
            counts.row_start[row] + host_rows[row].size();
    }
    counts.column.resize(counts.row_start[slots]);
    counts.value.resize(counts.row_start[slots]);
    parallel_for_index(slots, threads, [&](std::size_t row) {
        using Entry = std::pair<std::uint32_t, std::uint64_t>;
        std::vector<Entry> entries(host_rows[row].begin(),
                                   host_rows[row].end());
        std::ranges::sort(entries, {}, &Entry::first);
        std::size_t at = counts.row_start[row];
        for (const auto& [column, cell] : entries) {
            counts.column[at] = column;
            counts.value[at] = cell;
            ++at;
        }
    });
}

void transpose_sparse_rows(SparseCounts& counts, std::size_t slots) {
    std::vector<std::size_t> per_column(slots + 1, 0);
    for (const std::uint32_t column : counts.column) {
        if (column < slots) {
            ++per_column[static_cast<std::size_t>(column) + 1];
        }
    }
    counts.transposed_start.assign(slots + 1, 0);
    std::inclusive_scan(per_column.begin() + 1, per_column.end(),
                        counts.transposed_start.begin() + 1);
    counts.transposed_row.resize(counts.transposed_start[slots]);
    counts.transposed_value.resize(counts.transposed_start[slots]);
    std::vector<std::size_t> cursor(counts.transposed_start.begin(),
                                    counts.transposed_start.end() - 1);
    // Rows are visited in ascending order, so every transposed row comes out
    // sorted by source row without a second sort.
    for (std::size_t row = 0; row < slots; ++row) {
        for (std::size_t at = counts.row_start[row];
             at < counts.row_start[row + 1]; ++at) {
            const std::uint32_t column = counts.column[at];
            if (column >= slots) {
                continue;
            }
            const std::size_t target = cursor[column]++;
            counts.transposed_row[target] = static_cast<std::uint32_t>(row);
            counts.transposed_value[target] = counts.value[at];
        }
    }
}

[[nodiscard]] SparseCounts build_sparse(std::span<const HostCountRow> host_rows,
                                        std::size_t slots, unsigned threads) {
    if (host_rows.size() != slots) {
        throw std::invalid_argument(
            "pycanha::radiative: the tiled count rows do not match the "
            "face-slot count");
    }
    SparseCounts counts;
    sort_sparse_rows(counts, host_rows, threads);
    transpose_sparse_rows(counts, slots);
    return counts;
}

// --- pass A: closure, standard errors, optional full matrix ---------------

// Visits the nonzero cells of one row in ascending column order. Both
// layouts present the same order, which is what makes every derived
// statistic bit-identical between them.
template <typename Visit>
void visit_row(const DenseCounts& counts, std::size_t row, const Visit& visit) {
    const std::span<const std::uint32_t> cells = counts.row(row);
    for (std::size_t column = 0; column < cells.size(); ++column) {
        if (cells[column] != 0) {
            visit(column, static_cast<std::uint64_t>(cells[column]));
        }
    }
}

template <typename Visit>
void visit_row(const SparseCounts& counts, std::size_t row,
               const Visit& visit) {
    for (std::size_t at = counts.row_start[row]; at < counts.row_start[row + 1];
         ++at) {
        visit(static_cast<std::size_t>(counts.column[at]), counts.value[at]);
    }
}

struct ScanInputs {
    std::size_t slots = 0;
    std::span<const std::uint64_t> rays_per_row;
    std::span<const double> ratio;
    double threshold = 0.0;
    bool keep_full_matrix = false;
};

struct ScanOutputs {
    std::span<double> row_sums;
    std::span<RowStats> stats;
    std::span<RowEntries> full_rows;  // empty unless keep_full_matrix
};

template <typename Counts>
void scan_rows(const Counts& counts, const ScanInputs& in,
               const ScanOutputs& out, unsigned threads) {
    parallel_for_index(in.slots, threads, [&](std::size_t row) {
        const std::uint64_t rays = in.rays_per_row[row];
        if (rays == 0) {
            return;
        }
        const auto rays_as_double = static_cast<double>(rays);
        const double ratio = in.ratio[row];
        double row_sum = 0.0;
        RowStats stats;
        visit_row(counts, row, [&](std::size_t column, std::uint64_t count) {
            const double vf = static_cast<double>(count) / rays_as_double;
            // Closure accounting runs over every column and BEFORE any
            // thresholding, so pruning tiny entries never corrupts it.
            row_sum += vf;
            if (column < in.slots) {
                // Binomial standard error of the per-entry estimate (real
                // face columns only).
                const double entry_stderr =
                    std::sqrt(vf * std::max(1.0 - vf, 0.0) / rays_as_double);
                stats.stderr_sum += entry_stderr;
                stats.stderr_max = std::max(stats.stderr_max, entry_stderr);
                ++stats.entries;
            }
            if (in.keep_full_matrix && vf > in.threshold) {
                out.full_rows[row].push(column,
                                        ratio * static_cast<double>(count));
            }
        });
        out.row_sums[row] = row_sum;
        out.stats[row] = stats;
    });
}

// --- pass B: triangulation into the upper triangle ------------------------

struct PairInputs {
    std::size_t slots = 0;
    std::span<const double> areas;
    std::span<const double> ratio;
    const Weighting* weighting = nullptr;
    double threshold = 0.0;
};

void emit_pair(const PairInputs& in, std::size_t row, std::size_t column,
               std::uint64_t forward_count, std::uint64_t backward_count,
               RowEntries& entries, double& residual) {
    const Combined combined =
        combine_pair(in.ratio[row], in.ratio[column], forward_count,
                     backward_count, *in.weighting);
    residual = std::max(residual, combined.residual);
    if (keep_pair(combined.value, in.areas[row], in.areas[column],
                  in.threshold)) {
        entries.push(column, combined.value);
    }
}

// The space/inactive/lost columns are the closure accounting: they have no
// transpose partner to be combined with and no partner to be negligible
// against, so they pass through untriangulated and unthresholded.
void emit_buckets(const PairInputs& in,
                  std::span<const std::uint32_t> bucket_cells, std::size_t row,
                  RowEntries& entries) {
    for (std::size_t offset = 0; offset < bucket_cells.size(); ++offset) {
        if (bucket_cells[offset] != 0) {
            entries.push(
                in.slots + offset,
                in.ratio[row] * static_cast<double>(bucket_cells[offset]));
        }
    }
}

// Copies the transposed block (rows [col_begin, col_end), columns
// [row_begin, row_end)) into a small [row][column] scratch buffer. Read in
// place it would stride a full matrix row per useful cell; here the source
// runs are contiguous and the scatter stays inside a buffer that fits in L2
// alongside the forward block.
void gather_transposed(const DenseCounts& counts, std::size_t row_begin,
                       std::size_t height, std::size_t col_begin,
                       std::size_t width, std::span<std::uint32_t> scratch) {
    for (std::size_t column = 0; column < width; ++column) {
        const std::span<const std::uint32_t> source =
            counts.row(col_begin + column).subspan(row_begin, height);
        for (std::size_t row = 0; row < height; ++row) {
            scratch[(row * width) + column] = source[row];
        }
    }
}

void triangulate_diagonal_block(const PairInputs& in, const DenseCounts& counts,
                                std::size_t begin, std::size_t end,
                                std::span<RowEntries> rows, double& residual) {
    for (std::size_t row = begin; row < end; ++row) {
        const std::span<const std::uint32_t> forward = counts.row(row);
        for (std::size_t column = row; column < end; ++column) {
            const std::uint32_t forward_count = forward[column];
            const std::uint32_t backward_count = counts.row(column)[row];
            if (forward_count == 0 && backward_count == 0) {
                continue;
            }
            emit_pair(in, row, column, forward_count, backward_count, rows[row],
                      residual);
        }
    }
}

void triangulate_block(const PairInputs& in, const DenseCounts& counts,
                       std::size_t row_begin, std::size_t height,
                       std::size_t col_begin, std::size_t width,
                       std::span<const std::uint32_t> scratch,
                       std::span<RowEntries> rows, double& residual) {
    for (std::size_t row = 0; row < height; ++row) {
        const std::span<const std::uint32_t> forward =
            counts.row(row_begin + row).subspan(col_begin, width);
        const std::span<const std::uint32_t> backward =
            scratch.subspan(row * width, width);
        for (std::size_t column = 0; column < width; ++column) {
            if (forward[column] == 0 && backward[column] == 0) {
                continue;
            }
            emit_pair(in, row_begin + row, col_begin + column, forward[column],
                      backward[column], rows[row_begin + row], residual);
        }
    }
}

// Triangulates one row-tile against every tile to its right, then appends
// that tile's bucket columns. All output from a tile pair (I, J >= I) lands
// in row block I, so the row-tile index is a private, contiguous output
// range and the walk parallelises with no contention and no atomics.
void triangulate_row_tile(const PairInputs& in, const DenseCounts& counts,
                          std::size_t tile, std::size_t row_tile,
                          std::span<RowEntries> rows,
                          std::span<std::uint32_t> scratch, double& residual) {
    const std::size_t row_begin = row_tile * tile;
    const std::size_t row_end = std::min(row_begin + tile, in.slots);
    const std::size_t height = row_end - row_begin;
    for (std::size_t col_begin = row_begin; col_begin < in.slots;
         col_begin += tile) {
        const std::size_t width =
            std::min(col_begin + tile, in.slots) - col_begin;
        if (col_begin == row_begin) {
            triangulate_diagonal_block(in, counts, row_begin, row_end, rows,
                                       residual);
            continue;
        }
        gather_transposed(counts, row_begin, height, col_begin, width, scratch);
        triangulate_block(in, counts, row_begin, height, col_begin, width,
                          scratch.first(height * width), rows, residual);
    }
    for (std::size_t row = row_begin; row < row_end; ++row) {
        emit_buckets(in, counts.row(row).subspan(in.slots), row, rows[row]);
    }
}

// Merges the sorted row i of the counts with the sorted row i of their
// transpose, which is column i: linear in the stored entries, sequential in
// both streams, and a pair present in only one direction merges against an
// implicit zero — exactly the degenerate case the weight already handles.
void triangulate_sparse_row(const PairInputs& in, const SparseCounts& counts,
                            std::size_t row, RowEntries& entries,
                            double& residual) {
    std::size_t at = counts.row_start[row];
    const std::size_t row_end = counts.row_start[row + 1];
    while (at < row_end && counts.column[at] < row) {
        ++at;
    }
    std::size_t transposed = counts.transposed_start[row];
    const std::size_t transposed_end = counts.transposed_start[row + 1];
    while (transposed < transposed_end &&
           counts.transposed_row[transposed] < row) {
        ++transposed;
    }
    while (true) {
        // The slot count doubles as the end sentinel: past it lie only the
        // bucket columns, which never take part in a merge.
        const std::size_t forward_column =
            (at < row_end && counts.column[at] < in.slots) ? counts.column[at]
                                                           : in.slots;
        const std::size_t backward_column =
            transposed < transposed_end ? counts.transposed_row[transposed]
                                        : in.slots;
        const std::size_t column = std::min(forward_column, backward_column);
        if (column >= in.slots) {
            break;
        }
        std::uint64_t forward_count = 0;
        std::uint64_t backward_count = 0;
        if (forward_column == column) {
            forward_count = counts.value[at++];
        }
        if (backward_column == column) {
            backward_count = counts.transposed_value[transposed++];
        }
        emit_pair(in, row, column, forward_count, backward_count, entries,
                  residual);
    }
    for (; at < row_end; ++at) {
        entries.push(counts.column[at],
                     in.ratio[row] * static_cast<double>(counts.value[at]));
    }
}

// --- drivers -------------------------------------------------------------

[[nodiscard]] double triangulate_dense(const PairInputs& in,
                                       const DenseCounts& counts,
                                       std::span<RowEntries> rows,
                                       const AssemblyTuning& tuning) {
    // Tile 0 asks for the plain row-by-row walk: one tile spanning the
    // whole matrix degenerates to exactly that, transposed reads and all.
    const std::size_t tile =
        tuning.tile > 0 ? tuning.tile : std::max<std::size_t>(in.slots, 1);
    const std::size_t row_tiles = (in.slots + tile - 1) / tile;
    const unsigned threads =
        worker_count(tuning, row_tiles, in.slots * counts.cols);
    // A single row-tile means only the diagonal block runs, which reads the
    // transpose in place and never touches the scratch buffer.
    const std::size_t scratch_size = row_tiles > 1 ? tile * tile : 0;
    std::vector<double> tile_residual(row_tiles, 0.0);
    parallel_for_index(row_tiles, threads, [&](std::size_t row_tile) {
        std::vector<std::uint32_t> scratch(scratch_size, 0);
        triangulate_row_tile(in, counts, tile, row_tile, rows, scratch,
                             tile_residual[row_tile]);
    });
    return tile_residual.empty() ? 0.0
                                 : *std::ranges::max_element(tile_residual);
}

[[nodiscard]] double triangulate_sparse(const PairInputs& in,
                                        const SparseCounts& counts,
                                        std::span<RowEntries> rows,
                                        const AssemblyTuning& tuning) {
    const unsigned threads =
        worker_count(tuning, in.slots, in.slots * matrix_columns(in.slots));
    std::vector<double> row_residual(in.slots, 0.0);
    parallel_for_index(in.slots, threads, [&](std::size_t row) {
        triangulate_sparse_row(in, counts, row, rows[row], row_residual[row]);
    });
    return row_residual.empty() ? 0.0 : *std::ranges::max_element(row_residual);
}

// Reduces the per-row scan products in row order, so the double sums do not
// depend on how the rows were distributed across workers.
void reduce_stats(std::span<const RowStats> rows, TraceStats& stats) {
    double stderr_sum = 0.0;
    std::size_t entries = 0;
    double stderr_max = 0.0;
    for (const RowStats& row : rows) {
        stderr_sum += row.stderr_sum;
        entries += row.entries;
        stderr_max = std::max(stderr_max, row.stderr_max);
    }
    stats.mean_stderr =
        entries > 0 ? stderr_sum / static_cast<double>(entries) : 0.0;
    stats.max_stderr = stderr_max;
}

}  // namespace

VfResult assemble_vf(CountSource counts, std::span<const double> areas,
                     std::span<const std::uint64_t> rays_per_row,
                     const AccumConfig& config, const AssemblyTuning& tuning) {
    const std::size_t slots = areas.size();
    if (rays_per_row.size() != slots) {
        throw std::invalid_argument(
            "pycanha::radiative: rays_per_row must have one entry per face "
            "slot");
    }
    const std::size_t cols = matrix_columns(slots);
    const SlotWeights weights = slot_weights(areas, rays_per_row);
    const Weighting weighting(config.triangulation, tuning);
    const bool keep_full = config.triangulation.keep_full_matrix;

    VfResult result;
    result.row_sums = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(slots));
    std::vector<RowStats> row_stats(slots);
    std::vector<RowEntries> rows(slots);
    std::vector<RowEntries> full_rows(keep_full ? slots : 0);

    const ScanInputs scan_in{.slots = slots,
                             .rays_per_row = rays_per_row,
                             .ratio = weights.ratio,
                             .threshold = config.sparse_threshold,
                             .keep_full_matrix = keep_full};
    const ScanOutputs scan_out{
        .row_sums = std::span<double>(result.row_sums.data(), slots),
        .stats = row_stats,
        .full_rows = full_rows};
    const PairInputs pair_in{.slots = slots,
                             .areas = areas,
                             .ratio = weights.ratio,
                             .weighting = &weighting,
                             .threshold = config.sparse_threshold};
    const unsigned scan_threads = worker_count(tuning, slots, slots * cols);

    double residual = 0.0;
    if (const auto* mapped =
            std::get_if<std::span<const std::uint32_t>>(&counts)) {
        const DenseCounts dense = copy_dense(*mapped, slots);
        scan_rows(dense, scan_in, scan_out, scan_threads);
        residual = triangulate_dense(pair_in, dense, rows, tuning);
    } else {
        const SparseCounts sparse =
            build_sparse(std::get<std::span<const HostCountRow>>(counts), slots,
                         scan_threads);
        scan_rows(sparse, scan_in, scan_out, scan_threads);
        residual = triangulate_sparse(pair_in, sparse, rows, tuning);
    }

    result.vf = pack_rows(rows, static_cast<Eigen::Index>(cols), scan_threads);
    if (keep_full) {
        result.full_vf =
            pack_rows(full_rows, static_cast<Eigen::Index>(cols), scan_threads);
    }
    reduce_stats(row_stats, result.stats);
    result.stats.reciprocity_residual = residual;
    return result;
}

}  // namespace pycanha::radiative::detail
