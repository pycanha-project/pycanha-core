#include "pycanha-core/parameters/variable.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"

namespace pycanha {

TimeVariable::TimeVariable(std::string name, LookupTable1D lookup_table,
                           Parameters& parameters, const double* time_ptr)
    : _name(std::move(name)),
      _parameters(&parameters),
      _time_ptr(time_ptr),
      _lookup_table(std::move(lookup_table)) {
    if (_name.empty()) {
        throw std::invalid_argument("TimeVariable requires a non-empty name");
    }
    if (_time_ptr == nullptr) {
        throw std::invalid_argument(
            "TimeVariable requires a valid time pointer");
    }
    if (_parameters->contains(_name)) {
        throw std::invalid_argument("TimeVariable parameter '" + _name +
                                    "' already exists");
    }

    _current_value = _lookup_table.evaluate(*_time_ptr);
    _parameters->add_internal_parameter(_name, _current_value);

    const auto parameter_idx = _parameters->get_idx(_name);
    if (!parameter_idx.has_value()) {
        cleanup();
        throw std::runtime_error("TimeVariable could not resolve parameter '" +
                                 _name + "'");
    }

    _parameter_data_ptr = _parameters->get_double_ptr(*parameter_idx);
    if (_parameter_data_ptr == nullptr) {
        cleanup();
        throw std::runtime_error(
            "TimeVariable requires a double-backed parameter slot");
    }

    *_parameter_data_ptr = _current_value;
}

TimeVariable::~TimeVariable() { cleanup(); }

TimeVariable::TimeVariable(TimeVariable&& other) noexcept
    : _name(std::move(other._name)),
      _parameters(other._parameters),
      _time_ptr(other._time_ptr),
      _lookup_table(std::move(other._lookup_table)),
      _parameter_data_ptr(other._parameter_data_ptr),
      _current_value(other._current_value) {
    other._parameters = nullptr;
    other._time_ptr = nullptr;
    other._parameter_data_ptr = nullptr;
    other._current_value = 0.0;
}

TimeVariable& TimeVariable::operator=(TimeVariable&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    cleanup();

    _name = std::move(other._name);
    _parameters = other._parameters;
    _time_ptr = other._time_ptr;
    _lookup_table = std::move(other._lookup_table);
    _parameter_data_ptr = other._parameter_data_ptr;
    _current_value = other._current_value;

    other._parameters = nullptr;
    other._time_ptr = nullptr;
    other._parameter_data_ptr = nullptr;
    other._current_value = 0.0;

    return *this;
}

void TimeVariable::update() {
    if ((_time_ptr == nullptr) || (_parameter_data_ptr == nullptr)) {
        throw std::runtime_error("TimeVariable is not fully initialized");
    }

    _current_value = _lookup_table.evaluate(*_time_ptr);
    *_parameter_data_ptr = _current_value;
}

const std::string& TimeVariable::name() const noexcept { return _name; }

const LookupTable1D& TimeVariable::lookup_table() const {
    return _lookup_table;
}

double TimeVariable::current_value() const noexcept { return _current_value; }

void TimeVariable::cleanup() noexcept {
    if ((_parameters != nullptr) && !_name.empty() &&
        _parameters->is_internal_parameter(_name)) {
        _parameters->remove_internal_parameter(_name);
    }
    _parameter_data_ptr = nullptr;
}

TemperatureVariable::TemperatureVariable(std::string name,
                                         LookupTable1D lookup_table)
    : _name(std::move(name)), _lookup_table(std::move(lookup_table)) {
    if (_name.empty()) {
        throw std::invalid_argument(
            "TemperatureVariable requires a non-empty name");
    }
}

double TemperatureVariable::evaluate(double temperature) const {
    return _lookup_table.evaluate(temperature);
}

const std::string& TemperatureVariable::name() const noexcept { return _name; }

const LookupTable1D& TemperatureVariable::lookup_table() const noexcept {
    return _lookup_table;
}

}  // namespace pycanha
