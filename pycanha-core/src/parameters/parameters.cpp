#include "pycanha-core/parameters/parameters.hpp"

#include <spdlog/spdlog.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {

namespace {

template <typename T>
struct IsMatrixType : std::false_type {};

template <typename Scalar>
struct IsMatrixType<
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    : std::true_type {};

template <typename T>
constexpr bool is_matrix_type_v = IsMatrixType<T>::value;

}  // namespace

struct Parameters::DataMemoryAddress {
    template <typename T>
    void* operator()(T& data) const noexcept {
        if constexpr (std::is_same_v<T, std::string> || is_matrix_type_v<T>) {
            return static_cast<void*>(data.data());
        }

        return static_cast<void*>(std::addressof(data));
    }
};

struct Parameters::ConstDataMemoryAddress {
    template <typename T>
    const void* operator()(const T& data) const noexcept {
        if constexpr (std::is_same_v<T, std::string> || is_matrix_type_v<T>) {
            return static_cast<const void*>(data.data());
        }

        return static_cast<const void*>(std::addressof(data));
    }
};

struct Parameters::ParameterSize {
    template <typename T>
    std::size_t operator()(const T& data) const noexcept {
        if constexpr (std::is_same_v<T, std::string>) {
            return data.size() + 1U;
        } else if constexpr (is_matrix_type_v<T>) {
            return static_cast<std::size_t>(data.size()) *
                   sizeof(typename T::Scalar);
        }

        return sizeof(T);
    }
};

Parameters::ThermalValue Parameters::missing_parameter_value() {
    return ThermalValue{std::numeric_limits<double>::quiet_NaN()};
}

Parameters::ParameterSlot* Parameters::find_slot(
    const std::string& name) noexcept {
    const auto iterator = _name_to_slot.find(name);
    if (iterator == _name_to_slot.end()) {
        return nullptr;
    }

    return find_slot(to_idx(iterator->second));
}

const Parameters::ParameterSlot* Parameters::find_slot(
    const std::string& name) const noexcept {
    const auto iterator = _name_to_slot.find(name);
    if (iterator == _name_to_slot.end()) {
        return nullptr;
    }

    return find_slot(to_idx(iterator->second));
}

Parameters::ParameterSlot* Parameters::find_slot(Index idx) noexcept {
    if (idx < 0) {
        return nullptr;
    }

    const auto position = to_sizet(idx);
    if (position >= _slots.size()) {
        return nullptr;
    }

    auto* slot = std::addressof(_slots[position]);
    if (!slot->active) {
        return nullptr;
    }

    return slot;
}

const Parameters::ParameterSlot* Parameters::find_slot(
    Index idx) const noexcept {
    if (idx < 0) {
        return nullptr;
    }

    const auto position = to_sizet(idx);
    if (position >= _slots.size()) {
        return nullptr;
    }

    const auto* slot = std::addressof(_slots[position]);
    if (!slot->active) {
        return nullptr;
    }

    return slot;
}

void Parameters::invalidate_data_cache() const noexcept {
    _data_cache_dirty = true;
}

void Parameters::mark_structural_change() noexcept {
    ++_structure_version;
    invalidate_data_cache();
}

bool Parameters::Parameter::is_valid() const noexcept {
    return (_parameters != nullptr) && _parameters->is_parameter_valid(_idx);
}

std::optional<Index> Parameters::Parameter::get_idx() const noexcept {
    if (!is_valid()) {
        return std::nullopt;
    }

    return _idx;
}

std::optional<std::string> Parameters::Parameter::get_name() const {
    if (_parameters == nullptr) {
        return std::nullopt;
    }

    return _parameters->get_parameter_name(_idx);
}

std::optional<Parameters::ThermalValue> Parameters::Parameter::get_value()
    const {
    if (_parameters == nullptr) {
        return std::nullopt;
    }

    return _parameters->get_parameter_optional(_idx);
}

void Parameters::Parameter::set_value(ThermalValue value) {
    if (_parameters == nullptr) {
        return;
    }

    const auto name = get_name();
    if (!name.has_value()) {
        return;
    }

    _parameters->set_parameter(*name, std::move(value));
}

void Parameters::Parameter::rename(std::string new_name) {
    if (_parameters == nullptr) {
        return;
    }

    const auto current_name = get_name();
    if (!current_name.has_value()) {
        return;
    }

    _parameters->rename_parameter(*current_name, std::move(new_name));
}

void Parameters::Parameter::remove() {
    if (_parameters == nullptr) {
        return;
    }

    const auto name = get_name();
    if (!name.has_value()) {
        return;
    }

    _parameters->remove_parameter(*name);
}

void Parameters::add_parameter(std::string name, ThermalValue value) {
    if (_structure_locked) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' cannot be added while parameters "
                           "are structurally locked",
                           name);
        return;
    }

    if (_name_to_slot.contains(name)) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' already exists", name);
        return;
    }

    _name_to_slot.emplace(name, _slots.size());
    _slots.push_back(
        ParameterSlot{std::move(name), std::move(value), true, false});
    ++_active_size;
    mark_structural_change();

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Parameter '{}' added",
                       _slots.back().name);
}

void Parameters::add_internal_parameter(std::string name, ThermalValue value) {
    if (_name_to_slot.contains(name)) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Internal parameter '{}' already exists", name);
        return;
    }

    _name_to_slot.emplace(name, _slots.size());
    _slots.push_back(
        ParameterSlot{std::move(name), std::move(value), true, true});
    ++_active_size;
    mark_structural_change();

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Internal parameter '{}' added",
                       _slots.back().name);
}

void Parameters::remove_parameter(const std::string& name) {
    if (_structure_locked) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' cannot be removed while parameters "
                           "are structurally locked",
                           name);
        return;
    }

    auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return;
    }

    if (slot->is_internal) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Internal parameter '{}' cannot be removed with "
                           "remove_parameter",
                           name);
        return;
    }

    _name_to_slot.erase(name);
    slot->active = false;
    slot->value = missing_parameter_value();
    --_active_size;
    mark_structural_change();

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Parameter '{}' removed", name);
}

void Parameters::remove_internal_parameter(const std::string& name) {
    if (_structure_locked) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Internal parameter '{}' cannot be removed while "
                           "parameters are structurally locked",
                           name);
        return;
    }

    auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Internal parameter '{}' doesn't exist", name);
        return;
    }

    if (!slot->is_internal) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' is not internal", name);
        return;
    }

    _name_to_slot.erase(name);
    slot->active = false;
    slot->value = missing_parameter_value();
    slot->is_internal = false;
    --_active_size;
    mark_structural_change();

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Internal parameter '{}' removed",
                       name);
}

void Parameters::rename_parameter(const std::string& current_name,
                                  std::string new_name) {
    if (_structure_locked) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' cannot be renamed while parameters "
                           "are structurally locked",
                           current_name);
        return;
    }

    auto* slot = find_slot(current_name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", current_name);
        return;
    }

    if (slot->is_internal) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Internal parameter '{}' cannot be renamed",
                           current_name);
        return;
    }

    if (_name_to_slot.contains(new_name)) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' already exists", new_name);
        return;
    }

    const auto slot_index = _name_to_slot.at(current_name);
    _name_to_slot.erase(current_name);
    _name_to_slot.emplace(new_name, slot_index);
    slot->name = std::move(new_name);
    mark_structural_change();

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Parameter '{}' renamed",
                       slot->name);
}

Parameters::Parameter Parameters::get_parameter_handle(
    const std::string& name) noexcept {
    const auto index = get_idx(name);
    if (!index.has_value()) {
        return {};
    }

    return {this, *index};
}

Parameters::ThermalValue Parameters::get_parameter(
    const std::string& name) const {
    const auto parameter = get_parameter_optional(name);
    if (parameter.has_value()) {
        return *parameter;
    }

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Parameter '{}' doesn't exist",
                       name);

    return missing_parameter_value();
}

std::optional<Parameters::ThermalValue> Parameters::get_parameter_optional(
    const std::string& name) const {
    const auto* slot = find_slot(name);
    if (slot == nullptr) {
        return std::nullopt;
    }

    return slot->value;
}

std::optional<Parameters::ThermalValue> Parameters::get_parameter_optional(
    Index idx) const {
    const auto* slot = find_slot(idx);
    if (slot == nullptr) {
        return std::nullopt;
    }

    return slot->value;
}

void Parameters::set_parameter(const std::string& name, ThermalValue value) {
    auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return;
    }

    if (slot->is_internal) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Internal parameter '{}' cannot be modified with "
                           "set_parameter",
                           name);
        return;
    }

    if (slot->value.index() != value.index()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' type mismatch", name);
        return;
    }

    std::visit(
        [&](auto& existing) {
            using ExistingType = std::decay_t<decltype(existing)>;
            auto* incoming = std::get_if<ExistingType>(&value);
            if (incoming == nullptr) {
                return;
            }

            if constexpr (is_matrix_type_v<ExistingType>) {
                if ((existing.rows() != incoming->rows()) ||
                    (existing.cols() != incoming->cols())) {
                    SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                                       "Parameter '{}' shape mismatch", name);
                    return;
                }
            }

            existing = std::move(*incoming);
        },
        slot->value);

    invalidate_data_cache();
}

void Parameters::set_internal_parameter(const std::string& name,
                                        ThermalValue value) {
    auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Internal parameter '{}' doesn't exist", name);
        return;
    }

    if (!slot->is_internal) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' is not internal", name);
        return;
    }

    if (slot->value.index() != value.index()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Internal parameter '{}' type mismatch", name);
        return;
    }

    std::visit(
        [&](auto& existing) {
            using ExistingType = std::decay_t<decltype(existing)>;
            auto* incoming = std::get_if<ExistingType>(&value);
            if (incoming == nullptr) {
                return;
            }

            if constexpr (is_matrix_type_v<ExistingType>) {
                if ((existing.rows() != incoming->rows()) ||
                    (existing.cols() != incoming->cols())) {
                    SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                                       "Internal parameter '{}' shape "
                                       "mismatch",
                                       name);
                    return;
                }
            }

            existing = std::move(*incoming);
        },
        slot->value);

    invalidate_data_cache();
}

void Parameters::print_memory_address(const std::string& name) const {
    const auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return;
    }

    const void* const address =
        std::visit(ConstDataMemoryAddress{}, slot->value);

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Mem. addr: {}", address);
}

void Parameters::print_parameter(const std::string& name) const {
    const auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return;
    }

    std::ostringstream oss;
    oss << name << " = ";
    std::visit(
        [&](const auto& value) {
            if constexpr (is_matrix_type_v<std::decay_t<decltype(value)>>) {
                oss << '\n' << value;
            } else {
                oss << value;
            }
        },
        slot->value);

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "{}", oss.str());
}

void* Parameters::get_value_ptr(const std::string& name) {
    const auto index = get_idx(name);
    if (!index.has_value()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return nullptr;
    }

    return get_value_ptr(*index);
}

void* Parameters::get_value_ptr(Index idx) {
    auto* slot = find_slot(idx);
    if (slot == nullptr) {
        return nullptr;
    }

    return std::visit(DataMemoryAddress{}, slot->value);
}

const void* Parameters::get_value_ptr(const std::string& name) const {
    const auto index = get_idx(name);
    if (!index.has_value()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return nullptr;
    }

    return get_value_ptr(*index);
}

const void* Parameters::get_value_ptr(Index idx) const {
    const auto* slot = find_slot(idx);
    if (slot == nullptr) {
        return nullptr;
    }

    return std::visit(ConstDataMemoryAddress{}, slot->value);
}

double* Parameters::get_double_ptr(Index idx) {
    auto* slot = find_slot(idx);
    if (slot == nullptr) {
        return nullptr;
    }

    return std::get_if<double>(&slot->value);
}

const double* Parameters::get_double_ptr(Index idx) const {
    const auto* slot = find_slot(idx);
    if (slot == nullptr) {
        return nullptr;
    }

    return std::get_if<double>(&slot->value);
}

std::uint64_t Parameters::get_memory_address(const std::string& name) const {
    const void* const address = get_value_ptr(name);
    if (address == nullptr) {
        return 0U;
    }

    return static_cast<std::uint64_t>(std::bit_cast<std::uintptr_t>(address));
}

std::optional<Index> Parameters::get_idx(const std::string& name) const {
    const auto iterator = _name_to_slot.find(name);
    if (iterator == _name_to_slot.end()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return std::nullopt;
    }

    const auto* slot = find_slot(to_idx(iterator->second));
    if (slot == nullptr) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return std::nullopt;
    }

    return to_idx(iterator->second);
}

bool Parameters::is_internal_parameter(const std::string& name) const noexcept {
    const auto* slot = find_slot(name);
    return (slot != nullptr) && slot->is_internal;
}

bool Parameters::is_parameter_valid(Index idx) const noexcept {
    return find_slot(idx) != nullptr;
}

std::optional<std::string> Parameters::get_parameter_name(Index idx) const {
    const auto* slot = find_slot(idx);
    if (slot == nullptr) {
        return std::nullopt;
    }

    return slot->name;
}

std::size_t Parameters::get_size_of_parameter(const std::string& name) const {
    const auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return 0U;
    }

    return std::visit(ParameterSize{}, slot->value);
}

void Parameters::lock_structure() noexcept { _structure_locked = true; }

void Parameters::unlock_structure() noexcept { _structure_locked = false; }

bool Parameters::is_structure_locked() const noexcept {
    return _structure_locked;
}

std::uint64_t Parameters::get_structure_version() const noexcept {
    return _structure_version;
}

bool Parameters::contains(const std::string& name) const noexcept {
    return _name_to_slot.find(name) != _name_to_slot.end();
}

std::size_t Parameters::size() const noexcept { return _active_size; }

const Parameters::ParametersDict& Parameters::data() const {
    if (_data_cache_dirty) {
        _data_cache.clear();
        for (const auto& slot : _slots) {
            if (!slot.active) {
                continue;
            }

            _data_cache.emplace(slot.name, slot.value);
        }

        _data_cache_dirty = false;
    }

    return _data_cache;
}

}  // namespace pycanha
