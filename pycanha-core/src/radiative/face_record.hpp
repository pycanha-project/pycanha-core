#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/scene_part.hpp"

namespace pycanha::radiative::detail {

// One 32-bit record per face, holding everything a bounce needs to know about
// the face it hit: the activity and planet flags in the low bits and the
// material index above them. These used to be two buffers indexed by the same
// face id, so a bounce touched two cache lines for what is conceptually one
// record.
//
// THIS LAYOUT IS SHARED WITH THE KERNELS. It must stay in step with
// FACE_FLAG_ACTIVE, FACE_FLAG_PLANET and FACE_MATERIAL_SHIFT in
// kernels/common.slang, which is why it is packed here once for both the
// Vulkan and the Metal backend rather than in each of them.
inline constexpr std::uint32_t face_flag_active = 1U;
inline constexpr std::uint32_t face_flag_planet = 2U;
inline constexpr std::uint32_t face_material_shift = 2U;

// The material index is stored biased by one so that zero means "no material
// assigned" (a blackbody) and material row 0 stays a usable row.
inline constexpr std::uint32_t max_material_rows =
    (1U << (32U - face_material_shift)) - 2U;

[[nodiscard]] inline bool face_emits(std::uint32_t record) noexcept {
    return (record & face_flag_active) != 0U &&
           (record & face_flag_planet) == 0U;
}

[[nodiscard]] inline std::vector<std::uint32_t> pack_face_records(
    const MaterialTable& materials, std::span<const ScenePart> parts,
    std::uint32_t num_faces) {
    if (materials.num_materials() >
        static_cast<Eigen::Index>(max_material_rows)) {
        throw std::invalid_argument(
            "pycanha::radiative: too many optical materials to index from a "
            "face record");
    }

    std::vector<std::uint32_t> records(num_faces, 0U);
    for (std::uint32_t face = 0; face < num_faces; ++face) {
        std::uint32_t record = 0U;
        if (materials.face_active(face)) {
            record |= face_flag_active;
        }
        const int material = materials.face_material(face);
        if (material >= 0) {
            record |= (static_cast<std::uint32_t>(material) + 1U)
                      << face_material_shift;
        }
        records[face] = record;
    }

    // A celestial body absorbs every band and never emits, which is a property
    // of the PART, so it is stamped onto the faces that part carries. Both
    // sides of the pair are marked: the triangle defines them both.
    for (const ScenePart& part : parts) {
        if (part.kind != PartKind::CelestialBody) {
            continue;
        }
        const auto num_triangles = static_cast<Eigen::Index>(part.mesh.nt());
        for (Eigen::Index triangle = 0; triangle < num_triangles; ++triangle) {
            const auto base = part.mesh.face_ids(triangle);
            records[base] |= face_flag_planet;
            records[base + 1U] |= face_flag_planet;
        }
    }

    return records;
}

}  // namespace pycanha::radiative::detail
