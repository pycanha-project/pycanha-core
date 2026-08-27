#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
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

namespace {

void check_side(unsigned side, const char* who) {
    if (side != 1U && side != 2U) {
        throw std::invalid_argument(std::string(who) + ": side must be 1 or 2");
    }
}

}  // namespace

bool ThermalMesh::is_radiative_active(unsigned side) const {
    check_side(side, "ThermalMesh::is_radiative_active");
    return active_side_includes(_radiative_active_side, side);
}

bool ThermalMesh::is_conductive_active(unsigned side) const {
    check_side(side, "ThermalMesh::is_conductive_active");
    return active_side_includes(_conductive_active_side, side);
}

bool ThermalMesh::is_side_active(unsigned side) const {
    check_side(side, "ThermalMesh::is_side_active");
    return active_side_includes(_radiative_active_side, side) ||
           active_side_includes(_conductive_active_side, side);
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
    // MeshIndex. Use cut sizes (one more than the face pair count) as a safe
    // bound.
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

NodeNum ThermalMesh::node_of(MeshIndex i, MeshIndex j, unsigned side) const {
    if (side != 1U && side != 2U) {
        throw std::invalid_argument(
            "ThermalMesh::node_of: side must be 1 or 2");
    }
    if (i >= _dir1_mesh.size() - 1U || j >= _dir2_mesh.size() - 1U) {
        throw std::invalid_argument(
            "ThermalMesh::node_of: face_pair (i, j) is out of range");
    }
    // Direction 1 varies fastest: face_pair = i + j * n1. This is the face
    // order STEP-TAS uses for a meshed surface, so a face's index here is the
    // index it has in an exchanged model.
    const auto face_pair = (static_cast<std::int64_t>(j) *
                            static_cast<std::int64_t>(_dir1_mesh.size() - 1U)) +
                           static_cast<std::int64_t>(i);
    const std::int64_t start = side == 2U ? _node2_start : _node1_start;
    const std::int64_t step = side == 2U ? _node2_step : _node1_step;
    return static_cast<NodeNum>(start + (face_pair * step));
}

void ThermalMesh::validate() const {
    if (!is_valid()) {
        throw std::invalid_argument(
            "Invalid ThermalMesh: UV cuts must be sorted, span [0, 1], and "
            "have at least 2 entries per direction");
    }
}

}  // namespace pycanha::gmm
