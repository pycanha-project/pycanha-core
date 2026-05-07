#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"

namespace pycanha::gmm {

ThermalMesh::ThermalMesh() = default;

ThermalMesh::ThermalMesh(std::vector<double> dir1_cuts,
                         std::vector<double> dir2_cuts)
    : _dir1_cuts(std::move(dir1_cuts)), _dir2_cuts(std::move(dir2_cuts)) {
    validate_cuts(_dir1_cuts, "dir1");
    validate_cuts(_dir2_cuts, "dir2");
}

std::span<const double> ThermalMesh::dir1_cuts() const noexcept {
    return _dir1_cuts;
}

std::span<const double> ThermalMesh::dir2_cuts() const noexcept {
    return _dir2_cuts;
}

std::size_t ThermalMesh::num_faces_per_side() const noexcept {
    return (_dir1_cuts.size() - 1U) * (_dir2_cuts.size() - 1U);
}

FaceId ThermalMesh::face_id(std::size_t i, std::size_t j,
                            Side side) const noexcept {
    PYCANHA_ASSERT(i + 1U < _dir1_cuts.size(), "dir1 face index out of range");
    PYCANHA_ASSERT(j + 1U < _dir2_cuts.size(), "dir2 face index out of range");

    const std::size_t linear_index = i * (_dir2_cuts.size() - 1U) + j;
    return static_cast<FaceId>(static_cast<std::uint64_t>(
        2U * linear_index + static_cast<unsigned char>(side)));
}

void ThermalMesh::validate_cuts(std::span<const double> cuts,
                                const char* axis_name) {
    const bool valid = cuts.size() >= 2U &&
                       std::abs(cuts.front()) <= LENGTH_TOL &&
                       std::abs(cuts.back() - 1.0) <= LENGTH_TOL &&
                       std::is_sorted(cuts.begin(), cuts.end());

    PYCANHA_ASSERT(valid, "ThermalMesh cuts must be sorted and span [0, 1]");
    if (!valid) {
        throw std::invalid_argument(std::string("Invalid ThermalMesh ") +
                                    axis_name + " cuts");
    }
}

}  // namespace pycanha::gmm
