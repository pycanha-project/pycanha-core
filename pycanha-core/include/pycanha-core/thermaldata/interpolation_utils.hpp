#pragma once

#include <algorithm>
#include <stdexcept>

#include "pycanha-core/globals.hpp"

namespace pycanha::detail {

struct InterpLocation {
    Index lower_index = 0;
    double fraction = 0.0;
    bool below_range = false;
    bool above_range = false;
};

[[nodiscard]] inline InterpLocation locate(const double* x_data, Index size,
                                           double query) {
    if (x_data == nullptr) {
        throw std::invalid_argument("Interpolation data pointer is null");
    }
    if (size <= 0) {
        throw std::invalid_argument("Interpolation data must not be empty");
    }

    if (size == 1) {
        return {
            .lower_index = 0,
            .fraction = 0.0,
            .below_range = query<x_data[0], .above_range = query> x_data[0]};
    }

    if (query < x_data[0]) {
        const double dx = x_data[1] - x_data[0];
        return {.lower_index = 0,
                .fraction = (query - x_data[0]) / dx,
                .below_range = true,
                .above_range = false};
    }

    if (query > x_data[size - 1]) {
        const Index lower_index = size - 2;
        const double dx = x_data[size - 1] - x_data[size - 2];
        return {.lower_index = lower_index,
                .fraction = (query - x_data[lower_index]) / dx,
                .below_range = false,
                .above_range = true};
    }

    const double* begin = x_data;
    const double* end = x_data + size;
    const double* lower_bound = std::lower_bound(begin, end, query);
    const Index bound_index = static_cast<Index>(lower_bound - begin);

    if (bound_index == 0) {
        return {.lower_index = 0,
                .fraction = 0.0,
                .below_range = false,
                .above_range = false};
    }

    if ((bound_index < size) && (*lower_bound == query)) {
        if (bound_index == size - 1) {
            return {.lower_index = size - 2,
                    .fraction = 1.0,
                    .below_range = false,
                    .above_range = false};
        }

        return {.lower_index = bound_index,
                .fraction = 0.0,
                .below_range = false,
                .above_range = false};
    }

    const Index lower_index = bound_index - 1;
    const double dx = x_data[bound_index] - x_data[lower_index];
    return {.lower_index = lower_index,
            .fraction = (query - x_data[lower_index]) / dx,
            .below_range = false,
            .above_range = false};
}

}  // namespace pycanha::detail
