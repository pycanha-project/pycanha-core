#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/thermaldata/lookup_table.hpp"

namespace pycanha {

class DataTableStore {
  public:
    LookupTableVec1D& add_table(const std::string& name,
                                LookupTableVec1D table) {
        auto [iterator, inserted] =
            _tables.insert_or_assign(name, std::move(table));
        (void)inserted;
        return iterator->second;
    }

    [[nodiscard]] LookupTableVec1D& get_table(const std::string& name) {
        const auto iterator = _tables.find(name);
        if (iterator == _tables.end()) {
            throw std::out_of_range("Table does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] const LookupTableVec1D& get_table(
        const std::string& name) const {
        const auto iterator = _tables.find(name);
        if (iterator == _tables.end()) {
            throw std::out_of_range("Table does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] bool has_table(const std::string& name) const noexcept {
        return _tables.contains(name);
    }

    void remove_table(const std::string& name) { _tables.erase(name); }

    [[nodiscard]] std::size_t size() const noexcept { return _tables.size(); }

    [[nodiscard]] std::vector<std::string> table_names() const {
        std::vector<std::string> names;
        names.reserve(_tables.size());
        for (const auto& [name, table] : _tables) {
            (void)table;
            names.push_back(name);
        }

        return names;
    }

  private:
    std::unordered_map<std::string, LookupTableVec1D> _tables;
};

}  // namespace pycanha
