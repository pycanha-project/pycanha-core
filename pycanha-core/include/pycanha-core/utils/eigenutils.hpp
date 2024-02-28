#pragma once
#include <Eigen/Dense>

namespace pycanha::utils {

template <typename Derived>
inline bool is_sorted(const Eigen::MatrixBase<Derived>& v) {
    for (int i = 0; i < v.size() - 1; ++i) {
        if (v[i] > v[i + 1]) {
            return false;
        }
    }
    return true;
}

}  // namespace pycanha::utils
