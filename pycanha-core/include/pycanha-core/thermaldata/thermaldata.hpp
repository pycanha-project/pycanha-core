#pragma once

#include <cstddef>
#include <memory>
#include <pycanha-core/config.hpp>
#include <pycanha-core/thermaldata/dense_time_series.hpp>
#include <pycanha-core/thermaldata/lookup_table.hpp>
#include <pycanha-core/thermaldata/sparse_time_series.hpp>
#include <pycanha-core/tmm/thermalnetwork.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace pycanha {

class ThermalData {
  public:
    using DenseSeriesDict = std::unordered_map<std::string, DenseTimeSeries>;
    using SparseSeriesDict = std::unordered_map<std::string, SparseTimeSeries>;
    using LookupTableDict = std::unordered_map<std::string, LookupTable1D>;
    using LookupTableVecDict =
        std::unordered_map<std::string, LookupTableVec1D>;

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

    DenseTimeSeries& add_dense_time_series(const std::string& name,
                                           DenseTimeSeries series) {
        auto [iterator, inserted] =
            _dense_series.insert_or_assign(name, std::move(series));
        (void)inserted;
        return iterator->second;
    }

    DenseTimeSeries& add_dense_time_series(const std::string& name,
                                           Index num_timesteps,
                                           Index num_columns) {
        return add_dense_time_series(
            name, DenseTimeSeries(num_timesteps, num_columns));
    }

    [[nodiscard]] DenseTimeSeries& get_dense_time_series(
        const std::string& name) {
        auto iterator = _dense_series.find(name);
        if (iterator == _dense_series.end()) {
            throw std::out_of_range("Dense time series does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] const DenseTimeSeries& get_dense_time_series(
        const std::string& name) const {
        auto iterator = _dense_series.find(name);
        if (iterator == _dense_series.end()) {
            throw std::out_of_range("Dense time series does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] bool has_dense_time_series(
        const std::string& name) const noexcept {
        return _dense_series.find(name) != _dense_series.end();
    }

    void remove_dense_time_series(const std::string& name) {
        _dense_series.erase(name);
    }

    SparseTimeSeries& add_sparse_time_series(const std::string& name,
                                             SparseTimeSeries series) {
        auto [iterator, inserted] =
            _sparse_series.insert_or_assign(name, std::move(series));
        (void)inserted;
        return iterator->second;
    }

    SparseTimeSeries& add_sparse_time_series(const std::string& name) {
        return add_sparse_time_series(name, SparseTimeSeries());
    }

    [[nodiscard]] SparseTimeSeries& get_sparse_time_series(
        const std::string& name) {
        auto iterator = _sparse_series.find(name);
        if (iterator == _sparse_series.end()) {
            throw std::out_of_range("Sparse time series does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] const SparseTimeSeries& get_sparse_time_series(
        const std::string& name) const {
        auto iterator = _sparse_series.find(name);
        if (iterator == _sparse_series.end()) {
            throw std::out_of_range("Sparse time series does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] bool has_sparse_time_series(
        const std::string& name) const noexcept {
        return _sparse_series.find(name) != _sparse_series.end();
    }

    void remove_sparse_time_series(const std::string& name) {
        _sparse_series.erase(name);
    }

    LookupTable1D& add_lookup_table(const std::string& name,
                                    LookupTable1D table) {
        auto [iterator, inserted] =
            _lookup_tables.insert_or_assign(name, std::move(table));
        (void)inserted;
        return iterator->second;
    }

    [[nodiscard]] LookupTable1D& get_lookup_table(const std::string& name) {
        auto iterator = _lookup_tables.find(name);
        if (iterator == _lookup_tables.end()) {
            throw std::out_of_range("Lookup table does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] const LookupTable1D& get_lookup_table(
        const std::string& name) const {
        auto iterator = _lookup_tables.find(name);
        if (iterator == _lookup_tables.end()) {
            throw std::out_of_range("Lookup table does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] bool has_lookup_table(
        const std::string& name) const noexcept {
        return _lookup_tables.find(name) != _lookup_tables.end();
    }

    void remove_lookup_table(const std::string& name) {
        _lookup_tables.erase(name);
    }

    LookupTableVec1D& add_lookup_table_vec(const std::string& name,
                                           LookupTableVec1D table) {
        auto [iterator, inserted] =
            _lookup_tables_vec.insert_or_assign(name, std::move(table));
        (void)inserted;
        return iterator->second;
    }

    [[nodiscard]] LookupTableVec1D& get_lookup_table_vec(
        const std::string& name) {
        auto iterator = _lookup_tables_vec.find(name);
        if (iterator == _lookup_tables_vec.end()) {
            throw std::out_of_range("Vector lookup table does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] const LookupTableVec1D& get_lookup_table_vec(
        const std::string& name) const {
        auto iterator = _lookup_tables_vec.find(name);
        if (iterator == _lookup_tables_vec.end()) {
            throw std::out_of_range("Vector lookup table does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] bool has_lookup_table_vec(
        const std::string& name) const noexcept {
        return _lookup_tables_vec.find(name) != _lookup_tables_vec.end();
    }

    void remove_lookup_table_vec(const std::string& name) {
        _lookup_tables_vec.erase(name);
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return _dense_series.size() + _sparse_series.size() +
               _lookup_tables.size() + _lookup_tables_vec.size();
    }

  private:
    std::shared_ptr<ThermalNetwork> _network;
    DenseSeriesDict _dense_series;
    SparseSeriesDict _sparse_series;
    LookupTableDict _lookup_tables;
    LookupTableVecDict _lookup_tables_vec;
};

}  // namespace pycanha
