// Backend-agnostic VF assembly (pure CPU; see vf_assemble.hpp for why it
// does not live in either backend, and pair_walk.hpp for the traversal it
// shares with the exchange assembly).
//
// Two passes over the accumulated counts: a streaming row scan for closure
// and the per-entry statistics, then the pair walk that combines each face
// pair with its transpose into the upper triangle.
//
// Everything that accumulates a double does so in a fixed order — one whole
// row or one whole row-tile per worker, each writing its own output range —
// so the worker count and the schedule cannot change a bit of the result.
// The only reductions that cross workers are maxima, which are
// order-independent anyway.

#include "vf_assemble.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <variant>
#include <vector>

#include "pair_walk.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative::detail {

namespace {

// Per-slot precomputation. A_i/N_i is at once the variance proxy the weight
// is built from and the factor that turns a raw count into the extensive
// A_i * count / N_i, so hoisting it out of the pair loop removes a multiply
// and a divide from every cell and leaves the weight a function of two
// scalars that do not vary along the inner loop.
//
// A row that emitted nothing gets 0, which stands in for the infinite ratio
// that limit implies: every use of it either multiplies a count that is
// itself zero or is guarded explicitly.
[[nodiscard]] std::vector<double> slot_ratios(
    std::span<const double> areas,
    std::span<const std::uint64_t> rays_per_row) {
    std::vector<double> ratio(areas.size(), 0.0);
    for (std::size_t slot = 0; slot < areas.size(); ++slot) {
        if (rays_per_row[slot] > 0) {
            ratio[slot] = areas[slot] / static_cast<double>(rays_per_row[slot]);
        }
    }
    return ratio;
}

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

// --- pass A: closure, standard errors, optional full matrix ---------------

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

template <typename Cells>
void scan_rows(const Cells& cells, const ScanInputs& in, const ScanOutputs& out,
               unsigned threads) {
    parallel_for_index(in.slots, threads, [&](std::size_t row) {
        const std::uint64_t rays = in.rays_per_row[row];
        if (rays == 0) {
            return;
        }
        const auto rays_as_double = static_cast<double>(rays);
        const double ratio = in.ratio[row];
        double row_sum = 0.0;
        RowStats stats;
        visit_row(cells, row, [&](std::size_t column, std::uint64_t count) {
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

// The Emit the shared pair walk drives. Every member writes only into the
// row it is handed, so the walk can distribute rows however it likes.
class VfEmit {
  public:
    VfEmit(std::span<const double> areas, std::span<const double> ratio,
           const Weighting& weighting, double threshold,
           std::span<RowEntries> rows, std::span<double> residual)
        : _areas(areas),
          _ratio(ratio),
          _weighting(&weighting),
          _threshold(threshold),
          _rows(rows),
          _residual(residual) {}

    void pair(std::size_t row, std::size_t column, std::uint64_t forward,
              std::uint64_t backward) const {
        const Combined combined = combine_pair(_ratio[row], _ratio[column],
                                               forward, backward, *_weighting);
        _residual[row] = std::max(_residual[row], combined.residual);
        if (keep_pair(combined.value, _areas[row], _areas[column],
                      _threshold)) {
            _rows[row].push(column, combined.value);
        }
    }

    // The space/inactive/lost columns are the closure accounting: they have
    // no transpose partner to be combined with and no partner to be
    // negligible against, so they pass through untriangulated and
    // unthresholded.
    void bucket(std::size_t row, std::size_t column, std::uint64_t cell) const {
        _rows[row].push(column, _ratio[row] * static_cast<double>(cell));
    }

  private:
    std::span<const double> _areas;
    std::span<const double> _ratio;
    const Weighting* _weighting;
    double _threshold;
    std::span<RowEntries> _rows;
    std::span<double> _residual;
};

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
    const std::vector<double> ratio = slot_ratios(areas, rays_per_row);
    const Weighting weighting(config.triangulation, tuning);
    const bool keep_full = config.triangulation.keep_full_matrix;

    VfResult result;
    result.row_sums = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(slots));
    std::vector<RowStats> row_stats(slots);
    std::vector<RowEntries> rows(slots);
    std::vector<RowEntries> full_rows(keep_full ? slots : 0);
    std::vector<double> residual(slots, 0.0);

    const ScanInputs scan_in{.slots = slots,
                             .rays_per_row = rays_per_row,
                             .ratio = ratio,
                             .threshold = config.sparse_threshold,
                             .keep_full_matrix = keep_full};
    const ScanOutputs scan_out{
        .row_sums = std::span<double>(result.row_sums.data(), slots),
        .stats = row_stats,
        .full_rows = full_rows};
    const VfEmit emit(areas, ratio, weighting, config.sparse_threshold, rows,
                      residual);
    const unsigned scan_threads = worker_count(tuning, slots, slots * cols);

    if (const auto* mapped =
            std::get_if<std::span<const std::uint32_t>>(&counts)) {
        const DenseCells<std::uint32_t> dense = copy_dense(*mapped, slots);
        scan_rows(dense, scan_in, scan_out, scan_threads);
        walk_dense_pairs(dense, slots, tuning, emit);
    } else {
        const SparseCells sparse =
            build_sparse(std::get<std::span<const HostCountRow>>(counts), slots,
                         scan_threads);
        scan_rows(sparse, scan_in, scan_out, scan_threads);
        walk_sparse_pairs(sparse, slots, tuning, emit);
    }

    result.vf = pack_rows(rows, static_cast<Eigen::Index>(cols), scan_threads);
    if (keep_full) {
        result.full_vf =
            pack_rows(full_rows, static_cast<Eigen::Index>(cols), scan_threads);
    }
    reduce_stats(row_stats, result.stats);
    result.stats.reciprocity_residual =
        residual.empty() ? 0.0 : *std::ranges::max_element(residual);
    return result;
}

}  // namespace pycanha::radiative::detail
