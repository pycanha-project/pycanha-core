#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"

namespace pycanha::gmm {

ThermalMesh::ThermalMesh() { validate(); }

ThermalMesh::ThermalMesh(std::vector<double> dir1_mesh,
                         std::vector<double> dir2_mesh)
    : _dir1_mesh(std::move(dir1_mesh)), _dir2_mesh(std::move(dir2_mesh)) {
    validate();
}

std::span<const double> ThermalMesh::get_dir1_mesh() const noexcept {
    return _dir1_mesh;
}

std::span<const double> ThermalMesh::get_dir2_mesh() const noexcept {
    return _dir2_mesh;
}

void ThermalMesh::set_dir1_mesh(std::vector<double> dir1_mesh) {
    std::vector<double> previous = std::move(_dir1_mesh);
    _dir1_mesh = std::move(dir1_mesh);
    try {
        validate();
    } catch (...) {
        _dir1_mesh = std::move(previous);
        throw;
    }
}

void ThermalMesh::set_dir2_mesh(std::vector<double> dir2_mesh) {
    std::vector<double> previous = std::move(_dir2_mesh);
    _dir2_mesh = std::move(dir2_mesh);
    try {
        validate();
    } catch (...) {
        _dir2_mesh = std::move(previous);
        throw;
    }
}

void ThermalMesh::set_side1_thick(double thick) {
    if (thick < 0.0) {
        throw std::invalid_argument(
            "ThermalMesh: side1 thickness must be >= 0");
    }
    _side1_thick = thick;
}

void ThermalMesh::set_side2_thick(double thick) {
    if (thick < 0.0) {
        throw std::invalid_argument(
            "ThermalMesh: side2 thickness must be >= 0");
    }
    _side2_thick = thick;
}

bool ThermalMesh::is_valid() const noexcept {
    // Guard against face-id overflow: 2 * n1_cells * n2_cells must fit in
    // MeshIndex. Use cut sizes (one more than the cell count) as a safe bound.
    const bool fits =
        2U * _dir1_mesh.size() * _dir2_mesh.size() <=
        static_cast<std::size_t>(std::numeric_limits<MeshIndex>::max());

    return fits && _dir1_mesh.size() >= 2U && _dir2_mesh.size() >= 2U &&
           std::abs(_dir1_mesh.front()) <= LENGTH_TOL &&
           std::abs(_dir1_mesh.back() - 1.0) <= LENGTH_TOL &&
           std::abs(_dir2_mesh.front()) <= LENGTH_TOL &&
           std::abs(_dir2_mesh.back() - 1.0) <= LENGTH_TOL &&
           std::ranges::is_sorted(_dir1_mesh) &&
           std::ranges::is_sorted(_dir2_mesh);
}

MeshIndex ThermalMesh::get_number_of_pair_faces() const noexcept {
    return to_meshidx((_dir1_mesh.size() - 1U) * (_dir2_mesh.size() - 1U));
}

NodeNum ThermalMesh::node_of(MeshIndex i, MeshIndex j,
                             unsigned side) const noexcept {
    const auto cell = (static_cast<std::int64_t>(i) *
                       static_cast<std::int64_t>(_dir2_mesh.size() - 1U)) +
                      static_cast<std::int64_t>(j);
    const std::int64_t start = side == 2U ? _node2_start : _node1_start;
    const std::int64_t step = side == 2U ? _node2_step : _node1_step;
    return static_cast<NodeNum>(start + (cell * step));
}

void ThermalMesh::validate() const {
    if (!is_valid()) {
        throw std::invalid_argument(
            "Invalid ThermalMesh: UV cuts must be sorted, span [0, 1], and "
            "have at least 2 entries per direction");
    }
}

}  // namespace pycanha::gmm
