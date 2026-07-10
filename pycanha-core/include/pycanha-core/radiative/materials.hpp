#pragma once

#include <Eigen/Dense>
#include <cstdint>

namespace pycanha::radiative {

// Named radiation bands. Band is a data-model dimension: containers and
// kernel dispatch key on it, so adding bands later is additive rather than
// a redesign.
enum class Band : std::uint8_t { IR = 0, Solar = 1 };

// Per-face optical material / activity tables consumed by the raytracer.
// Built by gmm::GeometryModel::material_table() from the per-side ThermalMesh
// data; power users can fill one by hand (geometry-only workflows).
struct MaterialTable {
    // One row per unique OpticalMaterial: the 6 DOF in kernel order
    // [eps_ir, spec_ir, tau_ir, alpha_sol, spec_sol, tau_sol] — matches
    // gmm::OpticalMaterial::Properties exactly.
    Eigen::Matrix<float, Eigen::Dynamic, 6> properties;
    // Per face SLOT (global, both sides): row index into `properties`, or -1
    // for "no material assigned" (treated as blackbody; warned at build).
    Eigen::VectorXi face_material;
    // Per face SLOT: emission/reception activity (ThermalMesh side activity).
    Eigen::Matrix<bool, Eigen::Dynamic, 1> face_active;

    [[nodiscard]] Eigen::Index num_materials() const noexcept {
        return properties.rows();
    }
    [[nodiscard]] Eigen::Index num_face_slots() const noexcept {
        return face_material.rows();
    }
};

}  // namespace pycanha::radiative
