#include "pycanha-core/thermaldata/named_constants.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"

namespace pycanha {

Eigen::VectorXd& NamedConstants::times() noexcept { return _times; }

const Eigen::VectorXd& NamedConstants::times() const noexcept { return _times; }

std::vector<std::string>& NamedConstants::real_names() noexcept {
    return _real_names;
}

const std::vector<std::string>& NamedConstants::real_names() const noexcept {
    return _real_names;
}

NamedConstants::RealMatrix& NamedConstants::real_values() noexcept {
    return _real_values;
}

const NamedConstants::RealMatrix& NamedConstants::real_values() const noexcept {
    return _real_values;
}

std::vector<std::string>& NamedConstants::int_names() noexcept {
    return _int_names;
}

const std::vector<std::string>& NamedConstants::int_names() const noexcept {
    return _int_names;
}

NamedConstants::IntMatrix& NamedConstants::int_values() noexcept {
    return _int_values;
}

const NamedConstants::IntMatrix& NamedConstants::int_values() const noexcept {
    return _int_values;
}

std::vector<std::string>& NamedConstants::char_names() noexcept {
    return _char_names;
}

const std::vector<std::string>& NamedConstants::char_names() const noexcept {
    return _char_names;
}

std::size_t NamedConstants::char_width() const noexcept { return _char_width; }

void NamedConstants::set_char_storage(std::size_t width,
                                      std::vector<char> buffer) {
    _char_width = width;
    _char_values = std::move(buffer);
}

std::string_view NamedConstants::char_value(Index t, Index c) const {
    const auto num_char = static_cast<Index>(_char_names.size());
    if ((t < 0) || (t >= num_timesteps()) || (c < 0) || (c >= num_char)) {
        throw std::out_of_range("NamedConstants character index out of range");
    }

    const std::size_t offset =
        ((to_sizet(t) * to_sizet(num_char)) + to_sizet(c)) * _char_width;
    return {&_char_values[offset], _char_width};
}

Index NamedConstants::num_timesteps() const noexcept { return _times.size(); }

}  // namespace pycanha
