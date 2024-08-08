
#include "pycanha-core/tmm/literalstring.hpp"

#include <iostream>

LiteralString::LiteralString(int value) : m_string(std::string()) {}

LiteralString::LiteralString() : m_string(std::string()) {}

LiteralString::LiteralString(const std::string& str) : m_string(str) {}

const std::string& LiteralString::get_literal() { return m_string; }

const void LiteralString::print_string() { std::cout << m_string; }

bool LiteralString::is_empty() const { return m_string.empty(); }