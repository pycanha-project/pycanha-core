
/**
 * Custom string class that can be used as a template type for the SparseMatrix
 * and SparseVector classes of Eigen. This custom class is needed to represent
 * literal attributes which are typically empty. The custom class is needed
 * because when accesing a literal zero Eigen will return 'type_class(0)' and
 * std::string(0) is not an empty string, while LiteralString(0) will
 * initiallize its internal storage (based on std::string) with an empty string.
 *
 */

#pragma once

#include <iostream>
#include <string>
#include <utility>

namespace pycanha {

class LiteralString {
    friend class Nodes;

  public:
    LiteralString() = default;  // Default constructor
    explicit LiteralString([[maybe_unused]] int number) {}
    explicit LiteralString(const std::string& str) : _string(str) {}

    // Implicit conversion constructor from std::string
    explicit LiteralString(std::string&& str) : _string(std::move(str)) {}

    // Copy constructor
    LiteralString(const LiteralString& other) = default;

    // Move constructor
    LiteralString(LiteralString&& other) noexcept
        : _string(std::move(other._string)) {}

    // Copy assignment operator
    LiteralString& operator=(const LiteralString& other) {
        if (this != &other) {
            _string = other._string;
        }
        return *this;
    }

    // Move assignment operator
    LiteralString& operator=(LiteralString&& other) noexcept {
        if (this != &other) {
            _string = std::move(other._string);
        }
        return *this;
    }

    // Assignment operator from std::string
    LiteralString& operator=(const std::string& str) {
        _string = str;
        return *this;
    }

    void print_string() const { std::cout << _string; }
    [[nodiscard]] const std::string& get_literal() const { return _string; }
    [[nodiscard]] bool is_empty() const { return _string.empty(); }

    // Destructor
    ~LiteralString() = default;

  private:
    std::string _string;
};

}  // namespace pycanha
