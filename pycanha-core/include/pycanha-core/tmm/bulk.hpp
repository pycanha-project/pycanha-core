#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"

namespace pycanha {

/**
 * @brief Outcome of one bulk call on the nodes or the couplings.
 *
 * A bulk call never fails half-way: every entry is either taken into the model
 * or rejected, and rejected entries leave the model untouched. The first few
 * rejections are kept, in words, for the single WARN line the call logs.
 */
struct BulkReport {
    /// Entries written into the model.
    std::size_t accepted = 0;
    /// Couplings only: accepted entries that met an existing coupling, or an
    /// earlier entry of the same call, and were resolved by the merge mode.
    std::size_t merged = 0;
    /// Entries dropped: unknown or duplicated node, bad value, ...
    std::size_t rejected = 0;
    /// Up to max_reported_rejections of them, described.
    std::vector<std::string> first_rejections;

    static constexpr std::size_t max_reported_rejections = 5;

    /// Counts one rejection and keeps its description while there is room.
    void reject(std::string description) {
        ++rejected;
        if (first_rejections.size() < max_reported_rejections) {
            first_rejections.push_back(std::move(description));
        }
    }
};

/**
 * @brief A column-oriented batch of nodes of one type, for Nodes::add_nodes.
 *
 * Every non-empty span must have the length of @c numbers; an attribute left
 * empty takes its default (zero, or a zero temperature). Nothing is copied
 * until the values are written into the model's own storage.
 *
 * A batch sorted by increasing number whose numbers are all above the ones
 * already stored in its block is appended at the end of the storage, which
 * is the fastest case. Any other batch is sorted and merged in, which costs a
 * pass over every stored node.
 */
struct NodeBatch {
    /// 'D' (diffusive) or 'B' (boundary), for the whole batch.
    char type = 'D';
    std::span<const NodeNum> numbers{};
    /// T [K]
    std::span<const double> temperature{};
    /// C [J/K]
    std::span<const double> capacity{};
    std::span<const double> qs{};
    std::span<const double> qa{};
    std::span<const double> qe{};
    std::span<const double> qi{};
    std::span<const double> qr{};
    std::span<const double> a{};
    std::span<const double> fx{};
    std::span<const double> fy{};
    std::span<const double> fz{};
    std::span<const double> eps{};
    std::span<const double> aph{};
};

/// Node attribute selector for the bulk getters and setters.
enum class NodeAttribute : std::uint8_t {
    T,
    C,
    QS,
    QA,
    QE,
    QI,
    QR,
    A,
    FX,
    FY,
    FZ,
    EPS,
    APH,
};

/// What a bulk coupling insertion does with a coupling that already exists,
/// or that appears more than once in the same call. Matches the per-element
/// add_ovw_coupling / add_sum_coupling / add_new_coupling.
enum class CouplingMerge : std::uint8_t {
    /// The later value replaces the earlier one.
    OVERWRITE,
    /// The values are added, in call order.
    SUM,
    /// The first value is kept, later ones are dropped.
    NEW,
};

/**
 * @brief One run of couplings given by internal node index, for
 * CouplingMatrices::append_couplings.
 *
 * Entry i joins internal nodes @c offset + idx_1[i] and @c offset + idx_2[i]
 * (either order). The offset lets a caller that built a network part with its
 * own local node positions hand it over without rewriting its indices.
 */
struct CouplingChunk {
    std::span<const std::int32_t> idx_1{};
    std::span<const std::int32_t> idx_2{};
    std::span<const double> values{};
    std::int32_t offset = 0;
};

}  // namespace pycanha
