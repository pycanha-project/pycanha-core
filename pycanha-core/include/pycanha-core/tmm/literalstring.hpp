
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

class LiteralString {
    friend class Nodes;

  public:
    LiteralString() : m_string(std::string()) {}
    explicit LiteralString(int) : m_string(std::string()) {}
    explicit LiteralString(const std::string& str) : m_string(str) {}

    // Implicit conversion constructor from std::string
    explicit LiteralString(std::string&& str) : m_string(std::move(str)) {}

    // Copy constructor
    LiteralString(const LiteralString& other) : m_string(other.m_string) {}

    // Move constructor
    LiteralString(LiteralString&& other) noexcept
        : m_string(std::move(other.m_string)) {}

    // Copy assignment operator
    LiteralString& operator=(const LiteralString& other) {
        if (this != &other) {
            m_string = other.m_string;
        }
        return *this;
    }

    // Move assignment operator
    LiteralString& operator=(LiteralString&& other) noexcept {
        if (this != &other) {
            m_string = std::move(other.m_string);
        }
        return *this;
    }

    // Assignment operator from std::string
    LiteralString& operator=(const std::string& str) {
        m_string = str;
        return *this;
    }

    const void print_string() const { std::cout << m_string; }
    const std::string& get_literal() const { return m_string; }
    bool is_empty() const { return m_string.empty(); }

  private:
    std::string m_string;
};
