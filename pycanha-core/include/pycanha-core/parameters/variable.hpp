#pragma once

#include <string>

#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"

namespace pycanha {

class ThermalData;

class TimeVariable {
  public:
    TimeVariable(std::string name, LookupTable1D lookup_table,
                 Parameters& parameters, ThermalData& thermal_data,
                 const double* time_ptr);
    ~TimeVariable();

    TimeVariable(const TimeVariable&) = delete;
    TimeVariable& operator=(const TimeVariable&) = delete;
    TimeVariable(TimeVariable&& other) noexcept;
    TimeVariable& operator=(TimeVariable&& other) noexcept;

    void update();

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] const LookupTable1D& lookup_table() const;
    [[nodiscard]] double current_value() const noexcept;

  private:
    void cleanup() noexcept;

    std::string _name;
    std::string _lookup_table_key;
    Parameters* _parameters{nullptr};
    ThermalData* _thermal_data{nullptr};
    const double* _time_ptr{nullptr};
    const LookupTable1D* _lookup_table_ptr{nullptr};
    double* _parameter_data_ptr{nullptr};
    double _current_value{0.0};
};

class TemperatureVariable {
  public:
    TemperatureVariable(std::string name, LookupTable1D lookup_table);

    [[nodiscard]] double evaluate(double temperature) const;
    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] const LookupTable1D& lookup_table() const noexcept;

  private:
    std::string _name;
    LookupTable1D _lookup_table;
};

}  // namespace pycanha
