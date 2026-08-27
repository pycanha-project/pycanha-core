#pragma once

// Shared scene/material helpers for the radiative GPU test files. All
// geometry keeps the model-order face convention: the first added item owns
// faces 0/1, the second 2/3, and so on (even = side 1).

#include <array>
#include <cstdint>
#include <memory>
#include <span>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"

namespace radiative_fixtures {

// Two coaxial unit plates `gap` apart, facing each other: plate A (faces
// 0/1, side 1 up at z = 0) and plate B (faces 2/3, side 1 down at z = gap).
[[nodiscard]] inline std::unique_ptr<pycanha::gmm::GeometryModel>
make_parallel_plates(double gap) {
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    auto model = std::make_unique<pycanha::gmm::GeometryModel>("plates");
    model->add(std::make_shared<GeometryItem>(
        "plate_a", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{}));
    // Winding chosen so the side-1 normal points down (-z), toward plate A.
    model->add(std::make_shared<GeometryItem>(
        "plate_b", Rectangle({0.0, 0.0, gap}, {0.0, 1.0, gap}, {1.0, 0.0, gap}),
        ThermalMesh{}));
    return model;
}

// Same two plates plus a 3x3 sheet between them (faces 4/5, side 1 up at
// z = gap/2) — the transmission test scene.
[[nodiscard]] inline std::unique_ptr<pycanha::gmm::GeometryModel>
make_plates_with_sheet(double gap) {
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    auto model = make_parallel_plates(gap);
    const double mid = gap / 2.0;
    model->add(std::make_shared<GeometryItem>(
        "sheet",
        Rectangle({-1.0, -1.0, mid}, {2.0, -1.0, mid}, {-1.0, 2.0, mid}),
        ThermalMesh{}));
    return model;
}

// Specular bench: a unit source plate (faces 0/1, side 1 up at z = 0), a
// 45-degree tilted 4x4 mirror above it (faces 2/3, side-1 normal
// (1, 0, -1)/sqrt(2): +z rays reflect to +x) and a large catcher plate at
// x = 5 facing -x (faces 4/5). The source cannot see the catcher directly.
[[nodiscard]] inline std::unique_ptr<pycanha::gmm::GeometryModel>
make_mirror_bench() {
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    auto model = std::make_unique<pycanha::gmm::GeometryModel>("mirror_bench");
    model->add(std::make_shared<GeometryItem>(
        "source", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{}));
    // Mirror plane x - z = -1.5 through its center (0.5, 0.5, 2); in-plane
    // axes u = (0, 1, 0) and v = (1, 0, 1)/sqrt(2), half sizes 2 and 2.
    model->add(std::make_shared<GeometryItem>(
        "mirror",
        Rectangle({-0.914214, -1.5, 0.585786}, {-0.914214, 2.5, 0.585786},
                  {1.914214, -1.5, 3.414214}),
        ThermalMesh{}));
    model->add(std::make_shared<GeometryItem>(
        "catcher",
        Rectangle({5.0, -6.0, -4.0}, {5.0, -6.0, 8.0}, {5.0, 6.0, -4.0}),
        ThermalMesh{}));
    return model;
}

// Closed unit-cube enclosure: six unit plates with side 1 facing inward
// (faces 0/1 floor, 2/3 ceiling, 4/5 wall x=0, 6/7 wall x=1, 8/9 wall y=0,
// 10/11 wall y=1). Nothing escapes to space.
[[nodiscard]] inline std::unique_ptr<pycanha::gmm::GeometryModel>
make_box_enclosure() {
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    auto model = std::make_unique<pycanha::gmm::GeometryModel>("box");
    model->add(std::make_shared<GeometryItem>(
        "floor", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{}));
    model->add(std::make_shared<GeometryItem>(
        "ceiling", Rectangle({0.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 1.0}),
        ThermalMesh{}));
    model->add(std::make_shared<GeometryItem>(
        "wall_x0", Rectangle({0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}),
        ThermalMesh{}));
    model->add(std::make_shared<GeometryItem>(
        "wall_x1", Rectangle({1.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}),
        ThermalMesh{}));
    model->add(std::make_shared<GeometryItem>(
        "wall_y0", Rectangle({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}),
        ThermalMesh{}));
    model->add(std::make_shared<GeometryItem>(
        "wall_y1", Rectangle({0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}),
        ThermalMesh{}));
    return model;
}

// A material table with one row per entry of `rows` (kernel DOF order
// [eps_ir, spec_ir, tau_ir, alpha_sol, spec_sol, tau_sol]); both faces of
// face pair p map to row pair_rows[p]. Every face starts active.
[[nodiscard]] inline pycanha::radiative::MaterialTable make_materials(
    std::span<const std::array<float, 6>> rows,
    std::span<const int> pair_rows) {
    pycanha::radiative::MaterialTable table;
    table.properties.resize(static_cast<Eigen::Index>(rows.size()), 6);
    for (std::size_t row = 0; row < rows.size(); ++row) {
        for (std::size_t dof = 0; dof < 6; ++dof) {
            table.properties(static_cast<Eigen::Index>(row),
                             static_cast<Eigen::Index>(dof)) = rows[row][dof];
        }
    }
    const auto num_faces = static_cast<Eigen::Index>(pair_rows.size()) * 2;
    table.face_material.resize(num_faces);
    table.face_active.resize(num_faces);
    for (Eigen::Index face = 0; face < num_faces; ++face) {
        table.face_material(face) =
            pair_rows[static_cast<std::size_t>(face / 2)];
        table.face_active(face) = true;
    }
    return table;
}

// Gray-diffuse row: the same emissivity/absorptivity in both bands, no
// specular reflection or transmission.
[[nodiscard]] inline std::array<float, 6> gray_row(float absorptivity) {
    return {absorptivity, 0.0F, 0.0F, absorptivity, 0.0F, 0.0F};
}

[[nodiscard]] inline double csr_value(
    const pycanha::radiative::SparseMatrix& matrix, Eigen::Index row,
    Eigen::Index col) {
    for (pycanha::radiative::SparseMatrix::InnerIterator entry(matrix, row);
         entry; ++entry) {
        if (entry.col() == col) {
            return entry.value();
        }
    }
    return 0.0;
}

// Bit-exact CSR equality: same shape, same stored columns, same value bits.
// Integer accumulator cells make that a hard guarantee across layouts and
// backends, so this is an equality check and not a tolerance claim.
[[nodiscard]] inline bool csr_bit_identical(
    const pycanha::radiative::SparseMatrix& lhs,
    const pycanha::radiative::SparseMatrix& rhs) {
    using Iterator = pycanha::radiative::SparseMatrix::InnerIterator;
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols() ||
        lhs.nonZeros() != rhs.nonZeros()) {
        return false;
    }
    for (Eigen::Index row = 0; row < lhs.rows(); ++row) {
        Iterator left(lhs, row);
        Iterator right(rhs, row);
        for (; left && right; ++left, ++right) {
            if (left.col() != right.col() || left.value() != right.value()) {
                return false;
            }
        }
        if (left || right) {
            return false;
        }
    }
    return true;
}

}  // namespace radiative_fixtures
