#pragma once
#include <cstdint>
#include <iostream>
#include <mutex>

namespace pycanha::gmm {

using GeometryIdType = uint64_t;

/**
 * @class UniqueID
 * @brief Provides a unique integer id.
 * @details Inherit from this class to get a unique id every time an object is
 * created. The id is thread safe.
 *
 */

class UniqueID {
    // Private members

    // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)

    // Mutex for thread safety
    static inline std::mutex mtx;
    // Start at 1. 0 might be reserved for no id.
    static inline uint64_t next_id = 1;

    // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
    GeometryIdType _id;

    // Helper function for ID generation
    static uint64_t generate_id() {
        std::lock_guard<std::mutex> const lock(mtx);
        return next_id++;
    }

  protected:
    UniqueID() : _id(generate_id()) {}

  public:
    [[nodiscard]] uint64_t get_id() const { return _id; }

    // Copy constructor also assigns a new id
    UniqueID(const UniqueID& /* other */) : _id(generate_id()) {}

    // Copy assignment also assigns a new id
    UniqueID& operator=(const UniqueID& other) {
        if (this != &other) {
            _id = generate_id();
        }
        return *this;
    }

    // Destructor (default)
    ~UniqueID() = default;

    // Move constructor (delete)
    UniqueID(UniqueID&&) = delete;

    // Move assignment operator (delete)
    UniqueID& operator=(UniqueID&&) = delete;
};
}  // namespace pycanha::gmm
