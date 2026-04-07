#include "pycanha-core/parameters/parameters.hpp"

#include <spdlog/spdlog.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
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

void Parameters::add_parameter(std::string name, ThermalValue value) {
    auto [iterator, inserted] =
        _parameters.emplace(std::move(name), std::move(value));

    if (inserted) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Parameter '{}' added",
                           iterator->first);
    } else {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' already exists", iterator->first);
    }
}

void Parameters::remove_parameter(const std::string& name) {
    const auto removed = _parameters.erase(name);

    if (removed == 0U) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
    } else {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Parameter '{}' removed",
                           name);
    }
}

Parameters::ThermalValue Parameters::get_parameter(
    const std::string& name) const {
    const auto iterator = _parameters.find(name);
    if (iterator != _parameters.end()) {
        return iterator->second;
    }

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Parameter '{}' doesn't exist",
                       name);

    return ThermalValue{std::numeric_limits<double>::quiet_NaN()};
}

void Parameters::set_parameter(const std::string& name, ThermalValue value) {
    auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return;
    }

    if (iterator->second.index() != value.index()) {
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
        iterator->second);
}

void Parameters::print_memory_address(const std::string& name) const {
    const auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return;
    }

    const void* const address =
        std::visit(ConstDataMemoryAddress{}, iterator->second);

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Mem. addr: {}", address);
}

void Parameters::print_parameter(const std::string& name) const {
    const auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
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
        iterator->second);

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "{}", oss.str());
}

void* Parameters::get_value_ptr(const std::string& name) {
    auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return nullptr;
    }

    return std::visit(DataMemoryAddress{}, iterator->second);
}

const void* Parameters::get_value_ptr(const std::string& name) const {
    auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return nullptr;
    }

    return std::visit(ConstDataMemoryAddress{}, iterator->second);
}

std::uint64_t Parameters::get_memory_address(const std::string& name) const {
    const void* const address = get_value_ptr(name);
    if (address == nullptr) {
        return 0U;
    }

    return static_cast<std::uint64_t>(std::bit_cast<std::uintptr_t>(address));
}

std::optional<Index> Parameters::get_idx(const std::string& name) const {
    const auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return std::nullopt;
    }

    return to_idx(std::distance(_parameters.begin(), iterator));
}

std::size_t Parameters::get_size_of_parameter(const std::string& name) const {
    const auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return 0U;
    }

    return std::visit(ParameterSize{}, iterator->second);
}

bool Parameters::contains(const std::string& name) const noexcept {
    return _parameters.find(name) != _parameters.end();
}

std::size_t Parameters::size() const noexcept { return _parameters.size(); }

const Parameters::ParametersDict& Parameters::data() const noexcept {
    return _parameters;
}

}  // namespace pycanha
