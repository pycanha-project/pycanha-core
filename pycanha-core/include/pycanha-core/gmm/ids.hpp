#pragma once

#include <cstdint>

namespace pycanha::gmm {

enum class Kind : std::uint8_t {
    Item = 1,
    Group = 2,
    CutGroup = 3,
};

enum class GeometryId : std::uint64_t {};
enum class FaceId : std::uint64_t {};
using NodeNum = std::int64_t;

[[nodiscard]] constexpr std::uint64_t to_raw(GeometryId geometry_id) noexcept {
    return static_cast<std::uint64_t>(geometry_id);
}

[[nodiscard]] constexpr std::uint64_t to_raw(FaceId face_id) noexcept {
    return static_cast<std::uint64_t>(face_id);
}

[[nodiscard]] constexpr GeometryId make_geometry_id(
    Kind kind, std::uint32_t index) noexcept {
    constexpr std::uint64_t kind_shift = 56U;
    constexpr std::uint64_t index_mask = (std::uint64_t{1} << kind_shift) - 1U;
    return static_cast<GeometryId>(
        (static_cast<std::uint64_t>(kind) << kind_shift) |
        (static_cast<std::uint64_t>(index) & index_mask));
}

[[nodiscard]] constexpr Kind kind_of(GeometryId geometry_id) noexcept {
    constexpr std::uint64_t kind_shift = 56U;
    return static_cast<Kind>(to_raw(geometry_id) >> kind_shift);
}

[[nodiscard]] constexpr std::uint32_t index_of(
    GeometryId geometry_id) noexcept {
    constexpr std::uint64_t kind_shift = 56U;
    constexpr std::uint64_t index_mask = (std::uint64_t{1} << kind_shift) - 1U;
    return static_cast<std::uint32_t>(to_raw(geometry_id) & index_mask);
}

}  // namespace pycanha::gmm
