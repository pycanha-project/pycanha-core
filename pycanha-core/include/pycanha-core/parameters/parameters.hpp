#pragma once

#include <Eigen/Sparse>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <pycanha-core/config.hpp>
#include <string>
#include <unordered_map>
#include <variant>

#include "pycanha-core/globals.hpp"

namespace pycanha {

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

    void print_memory_address(const std::string& name) const;
    void print_parameter(const std::string& name) const;

    [[nodiscard]] void* get_value_ptr(const std::string& name);
    [[nodiscard]] void* get_value_ptr(Index idx);
    [[nodiscard]] const void* get_value_ptr(const std::string& name) const;
    [[nodiscard]] const void* get_value_ptr(Index idx) const;
    [[nodiscard]] double* get_double_ptr(Index idx);
    [[nodiscard]] const double* get_double_ptr(Index idx) const;
    [[nodiscard]] std::uint64_t get_memory_address(
        const std::string& name) const;

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

  private:
    struct ParameterSlot {
        std::string name;
        ThermalValue value;
        bool active{true};
        bool is_internal{false};
    };

    struct DataMemoryAddress;
    struct ConstDataMemoryAddress;
    struct ParameterSize;

    [[nodiscard]] static ThermalValue missing_parameter_value();
    [[nodiscard]] ParameterSlot* find_slot(const std::string& name) noexcept;
    [[nodiscard]] const ParameterSlot* find_slot(
        const std::string& name) const noexcept;
    [[nodiscard]] ParameterSlot* find_slot(Index idx) noexcept;
    [[nodiscard]] const ParameterSlot* find_slot(Index idx) const noexcept;
    void invalidate_data_cache() const noexcept;
    void mark_structural_change() noexcept;

    std::unordered_map<std::string, std::size_t> _name_to_slot;
    std::deque<ParameterSlot> _slots;
    std::size_t _active_size{0U};
    mutable ParametersDict _data_cache;
    mutable bool _data_cache_dirty{true};
    bool _structure_locked{false};
    std::uint64_t _structure_version{0U};
};

}  // namespace pycanha
