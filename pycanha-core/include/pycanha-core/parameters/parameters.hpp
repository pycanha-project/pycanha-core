#pragma once

#include <Eigen/Sparse>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <pycanha-core/config.hpp>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/origin.hpp"
#include "pycanha-core/parameters/table.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"

namespace pycanha {

class Parameters;

namespace detail {
// Pointer access to parameter values for the formula evaluator; not part
// of the public interface.
class ParameterAccess {
  public:
    [[nodiscard]] static const double* value(const Parameters& parameters,
                                             Index idx);
    // Every value change, of any parameter or of a table's data, takes the
    // next number of one counter. These point at the counter and at the
    // number of the last change of one parameter or one table.
    [[nodiscard]] static const std::uint64_t* changes(
        const Parameters& parameters) noexcept;
    [[nodiscard]] static const std::uint64_t* changed(
        const Parameters& parameters, Index idx);
    [[nodiscard]] static const std::uint64_t* changed(
        const Table& table) noexcept;
};
}  // namespace detail

class Parameters {
  public:
    using MatrixRXb =
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixRXi =
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixRXd =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ThermalValue = std::variant<bool, std::int64_t, double, std::string,
                                      MatrixRXb, MatrixRXi, MatrixRXd>;
    using ParametersDict = std::unordered_map<std::string, ThermalValue>;

    class Parameter {
      public:
        Parameter() = default;

        [[nodiscard]] bool is_valid() const noexcept;
        [[nodiscard]] std::optional<Index> get_idx() const noexcept;
        [[nodiscard]] std::optional<std::string> get_name() const;
        [[nodiscard]] std::optional<ThermalValue> get_value() const;
        void set_value(ThermalValue value);
        void rename(std::string new_name);
        void remove();

      private:
        friend class Parameters;

        Parameter(Parameters* parameters, Index idx) noexcept
            : _parameters(parameters), _idx(idx) {}

        Parameters* _parameters{nullptr};
        Index _idx{-1};
    };

    Parameters() = default;
    ~Parameters() = default;

    Parameters(const Parameters&) = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&) noexcept = default;
    Parameters& operator=(Parameters&&) noexcept = default;

    void add_parameter(std::string name, ThermalValue value);
    void add_internal_parameter(std::string name, ThermalValue value);
    void remove_parameter(const std::string& name);
    void remove_internal_parameter(const std::string& name);
    void rename_parameter(const std::string& current_name,
                          std::string new_name);
    [[nodiscard]] Parameter get_parameter_handle(
        const std::string& name) noexcept;

    [[nodiscard]] ThermalValue get_parameter(const std::string& name) const;
    [[nodiscard]] std::optional<ThermalValue> get_parameter_optional(
        const std::string& name) const;
    [[nodiscard]] std::optional<ThermalValue> get_parameter_optional(
        Index idx) const;
    void set_parameter(const std::string& name, ThermalValue value);
    void set_internal_parameter(const std::string& name, ThermalValue value);

    void print_parameter(const std::string& name) const;

    [[nodiscard]] std::optional<Index> get_idx(const std::string& name) const;
    [[nodiscard]] bool is_internal_parameter(
        const std::string& name) const noexcept;
    [[nodiscard]] bool is_parameter_valid(Index idx) const noexcept;
    [[nodiscard]] std::optional<std::string> get_parameter_name(
        Index idx) const;
    [[nodiscard]] std::size_t get_size_of_parameter(
        const std::string& name) const;

    void lock_structure() noexcept;
    void unlock_structure() noexcept;
    [[nodiscard]] bool is_structure_locked() const noexcept;
    [[nodiscard]] std::uint64_t get_structure_version() const noexcept;

    [[nodiscard]] bool contains(const std::string& name) const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const ParametersDict& data() const;

    // Tables share the parameters' name space (case-insensitive) and
    // structure version: adding, removing or resizing a table is a
    // structural change, refused while the structure is locked. A table
    // name cannot be an entity token (T, C, QS, QA, QE, QI, QR, GL, GR),
    // "time" or a built-in function or constant of the expressions.
    //
    // With a non-empty `evaluated_at`, the table is looked up at that
    // parameter on every update_evaluated_tables() call and the value is
    // published as an internal parameter with the table's name.
    // Throws std::invalid_argument for a refused name or argument and
    // std::runtime_error while the structure is locked.
    void add_table(std::string name, LookupTable1D lookup_table,
                   std::string evaluated_at = {});
    // A generator writes a table with its origin (anything but user, which
    // throws std::invalid_argument). A user table of the same name stays and
    // the call logs a WARN; a generated table of the same name, of any
    // origin, has its data and origin replaced, with a WARN when the origin
    // differs. Every write is a structural change.
    TableGeneration add_generated_table(Origin origin, std::string name,
                                        LookupTable1D lookup_table,
                                        std::string evaluated_at = {});
    // Removing a table that formulas call is allowed: the next formula
    // validation lists them. Returns false when there is no such table.
    bool remove_table(const std::string& name);
    [[nodiscard]] bool has_table(const std::string& name) const;
    // Null when there is no such table.
    [[nodiscard]] const Table* find_table(const std::string& name) const;
    // Throws std::out_of_range when there is no such table.
    [[nodiscard]] const Table& table(const std::string& name) const;
    [[nodiscard]] std::vector<std::string> table_names() const;
    // Overwrite the y or x values in place. The size must not change and x
    // must stay strictly increasing; nothing is written when a check fails.
    // Not a structural change, so they are safe in a callback during a
    // solve: the next lookup reads the new values.
    void set_table_values(const std::string& name,
                          std::span<const double> values);
    void set_table_x(const std::string& name, std::span<const double> x);

    // Looks up every table evaluated at a parameter and publishes its value.
    void update_evaluated_tables();
    // The tables queried outside their range since the last reset, by name.
    [[nodiscard]] std::vector<TableRange> table_ranges() const;
    void reset_table_ranges() const noexcept;
    // The value of the "time" parameter, or null without one. Out-of-range
    // records take the time from here.
    [[nodiscard]] const double* time_ptr() const;

  private:
    struct ParameterSlot {
        std::string name;
        ThermalValue value;
        bool active{true};
        bool is_internal{false};
        // The change counter at the last change of the value.
        std::uint64_t changed{0U};
    };

    struct ParameterSize;

    friend class detail::ParameterAccess;
    // The value of a double parameter, or null. Only the formulas read
    // values through a pointer; writes go through the setters.
    [[nodiscard]] double* get_double_ptr(Index idx);
    [[nodiscard]] const double* get_double_ptr(Index idx) const;

    [[nodiscard]] static ThermalValue missing_parameter_value();
    [[nodiscard]] std::optional<Index> find_idx(
        const std::string& name) const noexcept;
    [[nodiscard]] ParameterSlot* find_slot(const std::string& name) noexcept;
    [[nodiscard]] const ParameterSlot* find_slot(
        const std::string& name) const noexcept;
    [[nodiscard]] ParameterSlot* find_slot(Index idx) noexcept;
    [[nodiscard]] const ParameterSlot* find_slot(Index idx) const noexcept;
    void invalidate_data_cache() const noexcept;
    void note_change(ParameterSlot& slot) noexcept;
    void mark_structural_change() noexcept;
    void insert_slot(std::string name, ThermalValue value, bool is_internal);
    void deactivate_slot(ParameterSlot& slot);
    [[nodiscard]] bool is_table_name(std::string_view name) const;
    void check_table_change(std::string_view name) const;
    void publish_table(Table& table);
    void unpublish_table(Table& table);
    [[nodiscard]] Table& existing_table(const std::string& name);

    std::unordered_map<std::string, std::size_t> _name_to_slot;
    // Tables by lower-case name. Node-based, so a compiled formula's view
    // of one table survives adding and removing others.
    std::unordered_map<std::string, Table> _tables;
    std::deque<ParameterSlot> _slots;
    std::size_t _active_size{0U};
    mutable ParametersDict _data_cache;
    mutable bool _data_cache_dirty{true};
    bool _structure_locked{false};
    std::uint64_t _structure_version{0U};
    // Counts value changes (see detail::ParameterAccess::changes).
    std::uint64_t _changes{0U};
};

}  // namespace pycanha
