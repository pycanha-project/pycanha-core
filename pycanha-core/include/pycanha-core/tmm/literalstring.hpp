
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

#include <string>

class LiteralString {
    friend class Nodes;

  public:
    LiteralString(int value);
    LiteralString();
    LiteralString(const std::string& str);

    const void print_string();

    const std::string& get_literal();

    bool is_empty() const;

  private:
    std::string m_string;
};