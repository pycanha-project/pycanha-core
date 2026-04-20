#pragma once

#include <cstddef>
#include <memory>
#include <pycanha-core/config.hpp>
#include <pycanha-core/thermaldata/data_model_store.hpp>
#include <pycanha-core/thermaldata/data_table_store.hpp>
#include <pycanha-core/tmm/thermalnetwork.hpp>
#include <utility>

namespace pycanha {

class ThermalData {
  public:
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

    [[nodiscard]] DataTableStore& tables() noexcept { return _tables; }

    [[nodiscard]] const DataTableStore& tables() const noexcept {
        return _tables;
    }

    [[nodiscard]] DataModelStore& models() noexcept { return _models; }

    [[nodiscard]] const DataModelStore& models() const noexcept {
        return _models;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return _tables.size() + _models.size();
    }

  private:
    std::shared_ptr<ThermalNetwork> _network;
    DataTableStore _tables;
    DataModelStore _models;
};

}  // namespace pycanha
