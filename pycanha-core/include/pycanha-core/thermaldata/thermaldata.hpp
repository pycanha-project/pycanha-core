#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <iostream>
#include <memory>
#include <pycanha-core/config.hpp>
#include <pycanha-core/tmm/thermalnetwork.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace pycanha {

class ThermalData {
  public:
    using MatrixDataType =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using TableDict = std::unordered_map<std::string, MatrixDataType>;

    ThermalData() = default;
    explicit ThermalData(std::shared_ptr<ThermalNetwork> network) noexcept
        : _network(std::move(network)) {
        PYCANHA_ASSERT(_network != nullptr,
                       "ThermalData requires a valid ThermalNetwork");
    }

    void associate(std::shared_ptr<ThermalNetwork> network) noexcept {
        PYCANHA_ASSERT(network != nullptr,
                       "ThermalData requires a valid ThermalNetwork");
        _network = std::move(network);
    }

    [[nodiscard]] std::shared_ptr<ThermalNetwork> network_ptr() const noexcept {
        return _network;
    }

    [[nodiscard]] ThermalNetwork* network() noexcept {
        PYCANHA_ASSERT(_network != nullptr,
                       "ThermalData requires a valid ThermalNetwork");
        return _network.get();
    }

    [[nodiscard]] const ThermalNetwork* network() const noexcept {
        PYCANHA_ASSERT(_network != nullptr,
                       "ThermalData requires a valid ThermalNetwork");
        return _network.get();
    }

    void create_new_table(const std::string& name, Eigen::Index rows,
                          Eigen::Index cols) {
        auto [iterator, inserted] =
            _tables.try_emplace(name, MatrixDataType::Zero(rows, cols));

        if constexpr (DEBUG) {
            if (inserted) {
                std::cout << "Table '" << iterator->first << "' added\n";
            } else {
                std::cout << "Table '" << iterator->first
                          << "' already exists\n";
            }
        }
    }

    void create_reset_table(const std::string& name, Eigen::Index rows,
                            Eigen::Index cols) {
        auto iterator = _tables.find(name);
        if (iterator == _tables.end()) {
            create_new_table(name, rows, cols);
            return;
        }

        if ((iterator->second.rows() == rows) &&
            (iterator->second.cols() == cols)) {
            iterator->second.setZero();
            return;
        }

        iterator->second = MatrixDataType::Zero(rows, cols);

        if constexpr (DEBUG) {
            std::cout << "Table '" << name << "' resized\n";
        }
    }

    void remove_table(const std::string& name) {
        const auto removed = _tables.erase(name);
        if constexpr (DEBUG) {
            if (removed == 0U) {
                std::cout << "Table '" << name << "' doesn't exist\n";
            } else {
                std::cout << "Table '" << name << "' removed\n";
            }
        }
    }

    [[nodiscard]] MatrixDataType& get_table(const std::string& name) {
        auto iterator = _tables.find(name);
        if (iterator == _tables.end()) {
            throw std::out_of_range("Table doesn't exist");
        }

        return iterator->second;
    }

    [[nodiscard]] const MatrixDataType& get_table(
        const std::string& name) const {
        auto iterator = _tables.find(name);
        if (iterator == _tables.end()) {
            throw std::out_of_range("Table doesn't exist");
        }

        return iterator->second;
    }

    [[nodiscard]] bool has_table(const std::string& name) const noexcept {
        return _tables.find(name) != _tables.end();
    }

    [[nodiscard]] std::size_t size() const noexcept { return _tables.size(); }

  private:
    std::shared_ptr<ThermalNetwork> _network;
    TableDict _tables;
};

}  // namespace pycanha
