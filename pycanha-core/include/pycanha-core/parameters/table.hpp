#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/origin.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"

namespace pycanha {

// A table whose arguments went outside its x range, as reported after a
// solve: when it happened first and how far the arguments went.
struct TableRange {
    std::string table;
    // The model time of the first query outside the range; NaN when the
    // parameters have no "time".
    double first_time{std::numeric_limits<double>::quiet_NaN()};
    double first_argument{std::numeric_limits<double>::quiet_NaN()};
    // The smallest and largest argument seen outside the range.
    double min_argument{std::numeric_limits<double>::quiet_NaN()};
    double max_argument{std::numeric_limits<double>::quiet_NaN()};
    // The table's own range.
    double x_min{std::numeric_limits<double>::quiet_NaN()};
    double x_max{std::numeric_limits<double>::quiet_NaN()};
};

// What a generator's table write did, see Parameters::add_generated_table.
enum class TableGeneration : std::uint8_t {
    added,
    // A generated table of the same name had its data replaced.
    overwritten,
    // A user table of the same name exists; it stays and nothing was written.
    kept_user,
};

namespace detail {

class ParameterAccess;

// The queries of one table that fell outside its range since the last reset.
// Written only on the out-of-range branch of a lookup. Formulas may be
// evaluated on several threads: note() reads the flag and the range through
// atomic references and writes under a lock, and "first" is the first query
// recorded.
struct TableRangeRecord {
    alignas(std::atomic_ref<bool>::required_alignment) bool out_of_range{false};
    double first_time{std::numeric_limits<double>::quiet_NaN()};
    double first_argument{std::numeric_limits<double>::quiet_NaN()};
    alignas(std::atomic_ref<double>::required_alignment) double min_argument{
        std::numeric_limits<double>::quiet_NaN()};
    alignas(std::atomic_ref<double>::required_alignment) double max_argument{
        std::numeric_limits<double>::quiet_NaN()};

    void note(double argument, const double* time) noexcept;
    void reset() noexcept { *this = TableRangeRecord{}; }
};

// Everything a lookup reads, taken from a table when formulas are compiled.
// The pointers stay valid until the table is resized, removed or its
// methods change, which all change the parameters' structure version.
struct TableView {
    const double* x{nullptr};
    const double* y{nullptr};
    // slopes[i] is the slope of the segment [x[i], x[i + 1]].
    const double* slopes{nullptr};
    Index size{0};
    InterpolationMethod interp{InterpolationMethod::Linear};
    ExtrapolationMethod extrap{ExtrapolationMethod::Constant};
    // Null when out-of-range queries are not recorded.
    TableRangeRecord* range{nullptr};
    // The model time, for the out-of-range record; may be null.
    const double* time{nullptr};
    // For the message of a Throw table; read only when it throws.
    const std::string* name{nullptr};
};

// table_value for every case: the stepped interpolations, one-point tables
// and the queries outside the range.
[[nodiscard]] double table_value_general(const TableView& table,
                                         double argument);

// The interpolated value at `argument`. Outside the range: `Linear`
// extrapolates silently, `Constant` holds the end value and records the
// query, `Throw` throws std::out_of_range. A NaN argument is recorded and
// gives NaN.
//
// Inline for the formula evaluator: linear interpolation inside the range
// runs here with no call, everything else in table_value_general.
[[nodiscard]] inline double table_value(const TableView& table,
                                        double argument) {
    const std::span<const double> x(table.x,
                                    static_cast<std::size_t>(table.size));
    // False for a NaN argument too, which takes the general path.
    if ((table.interp != InterpolationMethod::Linear) || (x.size() < 2U) ||
        !(argument >= x.front()) || !(argument <= x.back())) {
        return table_value_general(table, argument);
    }
    // The segment [lower, lower + 1] holding the argument: the last point
    // not above it, at most the second to last. Conditional moves rather
    // than branches, since the arguments of consecutive lookups differ.
    std::size_t lower = 0U;
    std::size_t count = x.size() - 1U;
    while (count > 1U) {
        const std::size_t half = count / 2U;
        lower = (x[lower + half] <= argument) ? lower + half : lower;
        count -= half;
    }
    const std::span<const double> y(table.y, x.size());
    const std::span<const double> slopes(table.slopes, x.size() - 1U);
    return y[lower] + ((argument - x[lower]) * slopes[lower]);
}

// The derivative of table_value with respect to `argument`: the segment
// slope for `Linear` interpolation, zero for the stepped methods and for
// `Constant` extrapolation. At a breakpoint it is the slope of the segment
// above it.
[[nodiscard]] double table_slope(const TableView& table, double argument);

}  // namespace detail

// A named 1D table that formulas call as `name(argument)`. A table may also
// be evaluated at a parameter, usually "time": it is looked up once per
// callback and its value is published as a parameter with the table's name,
// so formulas can read `name` as a parameter too.
class Table {
  public:
    Table(std::string name, LookupTable1D lookup_table, Origin origin,
          std::string evaluated_at);

    [[nodiscard]] const std::string& name() const noexcept { return _name; }
    [[nodiscard]] const LookupTable1D& lookup_table() const noexcept {
        return _lookup_table;
    }
    [[nodiscard]] Origin origin() const noexcept { return _origin; }
    // The parameter the table is evaluated at, or empty for a table that is
    // only called from formulas.
    [[nodiscard]] const std::string& evaluated_at() const noexcept {
        return _evaluated_at;
    }

    // The value and slope at `argument`, without recording out-of-range
    // queries.
    [[nodiscard]] double evaluate(double argument) const;
    [[nodiscard]] double slope(double argument) const;

    // The view a compiled formula keeps; out-of-range queries through it are
    // recorded on this table.
    [[nodiscard]] detail::TableView view(const double* time) const noexcept;

  private:
    friend class Parameters;
    friend class detail::ParameterAccess;

    // Computes _slopes from the data; called whenever the data is written.
    void update_slopes();

    std::string _name;
    LookupTable1D _lookup_table;
    // The slope of every segment, so a lookup multiplies instead of
    // dividing. Written in place, so views keep pointing at it.
    std::vector<double> _slopes;
    std::string _evaluated_at;
    mutable detail::TableRangeRecord _range;
    // Resolved by Parameters for a table evaluated at a parameter.
    Index _argument_idx{-1};
    Index _published_idx{-1};
    // When the data was last written in place (set_table_values,
    // set_table_x), on the parameters' change counter.
    std::uint64_t _changed{0U};
    Origin _origin{Origin::user};
};

}  // namespace pycanha
