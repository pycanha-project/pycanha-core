#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/thermaldata/data_model.hpp"

namespace pycanha {

class DataModelStore {
  public:
    DataModel& add_model(const std::string& name, DataModel model) {
        auto [iterator, inserted] =
            _models.insert_or_assign(name, std::move(model));
        (void)inserted;
        return iterator->second;
    }

    [[nodiscard]] DataModel& get_model(const std::string& name) {
        const auto iterator = _models.find(name);
        if (iterator == _models.end()) {
            throw std::out_of_range("Model does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] const DataModel& get_model(const std::string& name) const {
        const auto iterator = _models.find(name);
        if (iterator == _models.end()) {
            throw std::out_of_range("Model does not exist");
        }

        return iterator->second;
    }

    [[nodiscard]] bool has_model(const std::string& name) const noexcept {
        return _models.contains(name);
    }

    void remove_model(const std::string& name) { _models.erase(name); }

    [[nodiscard]] std::size_t size() const noexcept { return _models.size(); }

    [[nodiscard]] std::vector<std::string> model_names() const {
        std::vector<std::string> names;
        names.reserve(_models.size());
        for (const auto& [name, model] : _models) {
            (void)model;
            names.push_back(name);
        }

        return names;
    }

  private:
    std::unordered_map<std::string, DataModel> _models;
};

}  // namespace pycanha
