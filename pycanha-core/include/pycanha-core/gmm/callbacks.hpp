#pragma once
#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

#include "./id.hpp"

namespace pycanha::gmm {

class GeometryUpdateCallback {
  protected:
    std::vector<std::function<void(GeometryIdType)>>
        _callbacks;                    ///< The callback functions.
    std::vector<GeometryIdType> _ids;  ///< Ids of the callback functions as
                                       ///< the id of a GeometryModel.

  public:
    void callback_with_id(GeometryIdType id) {
        std::cout << "Callback fun. Id: " << id << '\n';
        // Do something
        for (const auto& callback : _callbacks) {
            std::cout << "Calling callback\n";
            if (callback) {
                callback(id);
            }
        }
    }
    void add_callback(const std::function<void(GeometryIdType)>& callback,
                      GeometryIdType geometry_model_id) {
        // Use std::any_of to check if the id already exists
        if (std::any_of(_ids.begin(), _ids.end(),
                        [geometry_model_id](const auto& id) {
                            return id == geometry_model_id;
                        })) {
            // Callback already added, so return
            return;
        }
        _callbacks.push_back(callback);
        _ids.push_back(geometry_model_id);
    }

    void remove_callback(GeometryIdType geometry_model_id) {
        auto it = std::find(_ids.begin(), _ids.end(), geometry_model_id);
        if (it != _ids.end()) {
            auto index = std::distance(_ids.begin(), it);
            _ids.erase(it);
            _callbacks.erase(_callbacks.begin() + index);
        }
    }
};

}  // namespace pycanha::gmm
