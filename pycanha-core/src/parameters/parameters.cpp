#include "pycanha-core/parameters/parameters.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "pycanha-core/config.hpp"

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

    if constexpr (DEBUG) {
        if (inserted) {
            std::cout << "Parameter '" << iterator->first << "' added\n";
        } else {
            std::cout << "Parameter '" << iterator->first
                      << "' already exists\n";
        }
    } else {
        (void)iterator;
        (void)inserted;
    }
}

void Parameters::remove_parameter(const std::string& name) {
    const auto removed = _parameters.erase(name);

    if constexpr (DEBUG) {
        if (removed == 0U) {
            std::cout << "Parameter '" << name << "' doesn't exist\n";
        } else {
            std::cout << "Parameter '" << name << "' removed\n";
        }
    } else {
        (void)removed;
    }
}

Parameters::ThermalValue Parameters::get_parameter(
    const std::string& name) const {
    const auto iterator = _parameters.find(name);
    if (iterator != _parameters.end()) {
        return iterator->second;
    }

    if constexpr (DEBUG) {
        std::cout << "Parameter '" << name << "' doesn't exist\n";
    }

    return ThermalValue{std::numeric_limits<double>::quiet_NaN()};
}

void Parameters::set_parameter(const std::string& name, ThermalValue value) {
    auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        if constexpr (DEBUG) {
            std::cout << "Parameter '" << name << "' doesn't exist\n";
        }
        return;
    }

    if (iterator->second.index() != value.index()) {
        if constexpr (DEBUG) {
            std::cout << "Parameter '" << name << "' type mismatch\n";
        }
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
                    if constexpr (DEBUG) {
                        std::cout << "Parameter '" << name
                                  << "' shape mismatch\n";
                    }
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
        if constexpr (DEBUG) {
            std::cout << "Parameter '" << name << "' doesn't exist\n";
        }
        return;
    }

    const void* const address =
        std::visit(ConstDataMemoryAddress{}, iterator->second);

    if constexpr (DEBUG) {
        std::cout << "Mem. addr: " << address << '\n';
    } else {
        (void)address;
    }
}

void Parameters::print_parameter(const std::string& name) const {
    const auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        if constexpr (DEBUG) {
            std::cout << "Parameter '" << name << "' doesn't exist\n";
        }
        return;
    }

    if constexpr (DEBUG) {
        std::cout << name << " = ";
        std::visit(
            [&](const auto& value) {
                if constexpr (is_matrix_type_v<std::decay_t<decltype(value)>>) {
                    std::cout << '\n' << value << '\n';
                } else {
                    std::cout << value;
                }
            },
            iterator->second);

        std::cout << '\n';
    }
}

void* Parameters::get_value_ptr(const std::string& name) {
    auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        if constexpr (DEBUG) {
            std::cout << "Parameter '" << name << "' doesn't exist\n";
        }
        return nullptr;
    }

    return std::visit(DataMemoryAddress{}, iterator->second);
}

const void* Parameters::get_value_ptr(const std::string& name) const {
    auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        if constexpr (DEBUG) {
            std::cout << "Parameter '" << name << "' doesn't exist\n";
        }
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

int Parameters::get_idx(const std::string& name) const {
    const auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        if constexpr (DEBUG) {
            std::cout << "Parameter '" << name << "' doesn't exist\n";
        }
        return -1;
    }

    return static_cast<int>(std::distance(_parameters.begin(), iterator));
}

std::size_t Parameters::get_size_of_parameter(const std::string& name) const {
    const auto iterator = _parameters.find(name);
    if (iterator == _parameters.end()) {
        if constexpr (DEBUG) {
            std::cout << "Parameter '" << name << "' doesn't exist\n";
        }
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
