#include "pycanha-core/parameters/table.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "pycanha-core/parameters/origin.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"

namespace pycanha {

namespace {

constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();

[[nodiscard]] std::span<const double> x_of(const detail::TableView& table) {
    // A named span: some analyzers read `return {pointer, size}` as a list
    // of two elements.
    const std::span<const double> values(table.x,
                                         static_cast<std::size_t>(table.size));
    return values;
}

[[nodiscard]] std::span<const double> y_of(const detail::TableView& table) {
    // A named span: some analyzers read `return {pointer, size}` as a list
    // of two elements.
    const std::span<const double> values(table.y,
                                         static_cast<std::size_t>(table.size));
    return values;
}

// The segment [i, i + 1] that holds `argument`, which is inside the range,
// and the position in it. A breakpoint starts the segment above it, except
// the last one.
struct Segment {
    std::size_t lower{0U};
    double fraction{0.0};
};

[[nodiscard]] Segment find_segment(std::span<const double> x, double argument) {
    const auto above = std::ranges::upper_bound(x, argument);
    const auto bound = static_cast<std::size_t>(above - x.begin());
    const std::size_t lower = std::min(bound, x.size() - 1U) - 1U;
    return {.lower = lower,
            .fraction = (argument - x[lower]) / (x[lower + 1U] - x[lower])};
}

[[nodiscard]] double segment_slope(std::span<const double> x,
                                   std::span<const double> y,
                                   std::size_t lower) {
    return (y[lower + 1U] - y[lower]) / (x[lower + 1U] - x[lower]);
}

[[noreturn]] void throw_out_of_range(const detail::TableView& table,
                                     double argument) {
    const auto x = x_of(table);
    throw std::out_of_range(
        std::format("Table '{}' was queried at {}, outside its range [{}, {}]",
                    table.name == nullptr ? std::string("?") : *table.name,
                    argument, x.front(), x.back()));
}

// The out-of-range branch of table_value, kept apart from the lookup.
[[nodiscard]] double value_outside(const detail::TableView& table,
                                   double argument) {
    const auto x = x_of(table);
    const auto y = y_of(table);
    if (std::isnan(argument)) {
        if (table.extrap == ExtrapolationMethod::Throw) {
            throw_out_of_range(table, argument);
        }
        if (table.range != nullptr) {
            table.range->note(argument, table.time);
        }
        return nan_value;
    }
    switch (table.extrap) {
        case ExtrapolationMethod::Linear: {
            if (x.size() == 1U) {
                return y.front();
            }
            const std::size_t lower = argument < x.front() ? 0U : x.size() - 2U;
            return y[lower] +
                   ((argument - x[lower]) * segment_slope(x, y, lower));
        }
        case ExtrapolationMethod::Constant:
            if (table.range != nullptr) {
                table.range->note(argument, table.time);
            }
            return argument < x.front() ? y.front() : y.back();
        case ExtrapolationMethod::Throw:
            throw_out_of_range(table, argument);
    }
    return nan_value;
}

[[nodiscard]] double slope_outside(const detail::TableView& table,
                                   double argument) {
    const auto x = x_of(table);
    if (std::isnan(argument)) {
        return value_outside(table, argument);
    }
    switch (table.extrap) {
        case ExtrapolationMethod::Linear:
            if ((x.size() == 1U) ||
                (table.interp != InterpolationMethod::Linear)) {
                return 0.0;
            }
            return segment_slope(x, y_of(table),
                                 argument < x.front() ? 0U : x.size() - 2U);
        case ExtrapolationMethod::Constant:
            if (table.range != nullptr) {
                table.range->note(argument, table.time);
            }
            return 0.0;
        case ExtrapolationMethod::Throw:
            throw_out_of_range(table, argument);
    }
    return nan_value;
}

}  // namespace

namespace detail {

void TableRangeRecord::note(double argument, const double* time) noexcept {
    if (!out_of_range) {
        out_of_range = true;
        first_time = time == nullptr ? nan_value : *time;
        first_argument = argument;
        min_argument = argument;
        max_argument = argument;
        return;
    }
    // fmin / fmax ignore a NaN operand, so a NaN query never hides the range.
    min_argument = std::fmin(min_argument, argument);
    max_argument = std::fmax(max_argument, argument);
}

double table_value(const TableView& table, double argument) {
    const auto x = x_of(table);
    // False for a NaN argument too, which then takes the out-of-range branch.
    const bool inside = (argument >= x.front()) && (argument <= x.back());
    if (!inside) {
        return value_outside(table, argument);
    }
    const auto y = y_of(table);
    if (x.size() == 1U) {
        return y.front();
    }

    const auto [lower, fraction] = find_segment(x, argument);
    switch (table.interp) {
        case InterpolationMethod::Linear:
            return y[lower] + (fraction * (y[lower + 1U] - y[lower]));
        case InterpolationMethod::NearestLower:
        case InterpolationMethod::Step:
            return fraction >= 1.0 ? y[lower + 1U] : y[lower];
        case InterpolationMethod::NearestUpper:
            return fraction <= 0.0 ? y[lower] : y[lower + 1U];
    }
    return nan_value;
}

double table_slope(const TableView& table, double argument) {
    const auto x = x_of(table);
    const bool inside = (argument >= x.front()) && (argument <= x.back());
    if (!inside) {
        return slope_outside(table, argument);
    }
    if ((x.size() == 1U) || (table.interp != InterpolationMethod::Linear)) {
        return 0.0;
    }
    return segment_slope(x, y_of(table), find_segment(x, argument).lower);
}

}  // namespace detail

Table::Table(std::string name, LookupTable1D lookup_table, Origin origin,
             std::string evaluated_at)
    : _name(std::move(name)),
      _lookup_table(std::move(lookup_table)),
      _evaluated_at(std::move(evaluated_at)),
      _origin(origin) {
    if (_lookup_table.size() == 0) {
        throw std::invalid_argument("Table '" + _name + "' has no data");
    }
}

double Table::evaluate(double argument) const {
    auto table = view(nullptr);
    table.range = nullptr;
    return detail::table_value(table, argument);
}

double Table::slope(double argument) const {
    auto table = view(nullptr);
    table.range = nullptr;
    return detail::table_slope(table, argument);
}

detail::TableView Table::view(const double* time) const noexcept {
    return {.x = _lookup_table.x().data(),
            .y = _lookup_table.y().data(),
            .size = _lookup_table.size(),
            .interp = _lookup_table.interpolation_method(),
            .extrap = _lookup_table.extrapolation_method(),
            .range = &_range,
            .time = time,
            .name = &_name};
}

std::string_view to_string(Origin origin) noexcept {
    switch (origin) {
        case Origin::user:
            return "user";
        case Origin::conduction_builder:
            return "conduction_builder";
        case Origin::etms_reader:
            return "etms_reader";
        case Origin::esatan_tmm_reader:
            return "esatan_tmm_reader";
        case Origin::steptas_gmm_reader:
            return "steptas_gmm_reader";
    }
    return "unknown";
}

}  // namespace pycanha
