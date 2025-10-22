#pragma once

#include <Eigen/Sparse>
#include <cstddef>
#include <cstdint>
#include <pycanha-core/config.hpp>
#include <string>
#include <unordered_map>
#include <variant>

namespace pycanha {

class Parameters {
  public:
    using MatrixRXb =
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixRXi =
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixRXd =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ThermalValue = std::variant<bool, int, double, std::string, MatrixRXb,
                                      MatrixRXi, MatrixRXd>;
    using ParametersDict = std::unordered_map<std::string, ThermalValue>;

    Parameters() = default;
    ~Parameters() = default;

    Parameters(const Parameters&) = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&) noexcept = default;
    Parameters& operator=(Parameters&&) noexcept = default;

    void add_parameter(std::string name, ThermalValue value);
    void remove_parameter(const std::string& name);

    [[nodiscard]] ThermalValue get_parameter(const std::string& name) const;
    void set_parameter(const std::string& name, ThermalValue value);

    void print_memory_address(const std::string& name) const;
    void print_parameter(const std::string& name) const;

    [[nodiscard]] void* get_value_ptr(const std::string& name);
    [[nodiscard]] const void* get_value_ptr(const std::string& name) const;
    [[nodiscard]] std::uint64_t get_memory_address(
        const std::string& name) const;

    [[nodiscard]] int get_idx(const std::string& name) const;
    [[nodiscard]] std::size_t get_size_of_parameter(
        const std::string& name) const;

    [[nodiscard]] bool contains(const std::string& name) const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const ParametersDict& data() const noexcept;

  private:
    struct DataMemoryAddress;
    struct ConstDataMemoryAddress;
    struct ParameterSize;

    ParametersDict _parameters;
};

}  // namespace pycanha
