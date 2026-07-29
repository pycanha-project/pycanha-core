#pragma once

// Shared scene/material helpers for the radiative GPU test files. All
// geometry keeps the model-order slot convention: the first added item owns
// slots 0/1, the second 2/3, and so on (even = side 1).

#include <array>
#include <cstdint>
#include <memory>
#include <span>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/sparse.hpp"

namespace radiative_fixtures {

// Two coaxial unit plates `gap` apart, facing each other: plate A (slots
// 0/1, side 1 up at z = 0) and plate B (slots 2/3, side 1 down at z = gap).
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

// Same two plates plus a 3x3 sheet between them (slots 4/5, side 1 up at
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

// Specular bench: a unit source plate (slots 0/1, side 1 up at z = 0), a
// 45-degree tilted 4x4 mirror above it (slots 2/3, side-1 normal
// (1, 0, -1)/sqrt(2): +z rays reflect to +x) and a large catcher plate at
// x = 5 facing -x (slots 4/5). The source cannot see the catcher directly.
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
// (slots 0/1 floor, 2/3 ceiling, 4/5 wall x=0, 6/7 wall x=1, 8/9 wall y=0,
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
// [eps_ir, spec_ir, tau_ir, alpha_sol, spec_sol, tau_sol]); both slots of
// face pair p map to row pair_rows[p]. Every slot starts active.
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
    const auto num_slots = static_cast<Eigen::Index>(pair_rows.size()) * 2;
    table.face_material.resize(num_slots);
    table.face_active.resize(num_slots);
    for (Eigen::Index slot = 0; slot < num_slots; ++slot) {
        table.face_material(slot) =
            pair_rows[static_cast<std::size_t>(slot / 2)];
        table.face_active(slot) = true;
    }
    return table;
}

// Gray-diffuse row: the same emissivity/absorptivity in both bands, no
// specular reflection or transmission.
[[nodiscard]] inline std::array<float, 6> gray_row(float absorptivity) {
    return {absorptivity, 0.0F, 0.0F, absorptivity, 0.0F, 0.0F};
}

[[nodiscard]] inline double csr_value(
    const pycanha::radiative::SparseF64& matrix, std::int64_t row,
    std::int32_t col) {
    for (std::int64_t k = matrix.indptr(row); k < matrix.indptr(row + 1); ++k) {
        if (matrix.indices(k) == col) {
            return matrix.values(k);
        }
    }
    return 0.0;
}

}  // namespace radiative_fixtures
