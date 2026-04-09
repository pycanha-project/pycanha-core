#pragma once
#include <Eigen/Dense>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pycanha {

// Global constants and type aliases used across the library.

// Tolerances (SI Units)

// Constexpr should be usually lowercase (checked by clang-tidy), but I want
// these to be uppercase NOLINTBEGIN(readability-identifier-naming)
constexpr double LENGTH_TOL = 1e-9;
constexpr double ANGLE_TOL = 1e-9;
constexpr double TOL = 1e-11;
constexpr double ZERO_THR_ATTR = std::numeric_limits<double>::epsilon() * 1e3;
constexpr double ALMOST_EQUAL_COUPLING_EPSILON = 1.0e-5;
// DOI: 10.1103/RevModPhys.97.025002
constexpr double STF_BOLTZ = 5.670374419e-8;
// NOLINTEND(readability-identifier-naming)

using Point2D = Eigen::Vector2d;
using Point3D = Eigen::Vector3d;

using Vector2D = Eigen::Vector2d;
using Vector3D = Eigen::Vector3d;

using Index = Eigen::Index;
using NodeNum = std::int32_t;
using MeshIndex = std::uint32_t;
using IntAddress = std::uint64_t;  // Unsigned 64 bit integer to pass memory
                                   // addresses

[[nodiscard]] constexpr Index to_idx(Index index) noexcept { return index; }

[[nodiscard]] constexpr Index to_idx(std::size_t size) noexcept {
    assert(size <= static_cast<std::size_t>(std::numeric_limits<Index>::max()));
    return static_cast<Index>(size);
}

[[nodiscard]] constexpr Index to_idx(MeshIndex mesh_index) noexcept {
    return static_cast<Index>(mesh_index);
}

[[nodiscard]] constexpr Index to_idx(int value) noexcept {
    return static_cast<Index>(value);
}

[[nodiscard]] constexpr std::size_t to_sizet(std::size_t size) noexcept {
    return size;
}

[[nodiscard]] constexpr std::size_t to_sizet(Index index) noexcept {
    assert(index >= 0);
    return static_cast<std::size_t>(index);
}

[[nodiscard]] constexpr std::size_t to_sizet(MeshIndex mesh_index) noexcept {
    return static_cast<std::size_t>(mesh_index);
}

[[nodiscard]] constexpr std::size_t to_sizet(int value) noexcept {
    assert(value >= 0);
    return static_cast<std::size_t>(value);
}

[[nodiscard]] constexpr MeshIndex to_meshidx(MeshIndex mesh_index) noexcept {
    return mesh_index;
}

[[nodiscard]] constexpr MeshIndex to_meshidx(Index index) noexcept {
    assert(index >= 0 &&
           index <= static_cast<Index>(std::numeric_limits<MeshIndex>::max()));
    return static_cast<MeshIndex>(index);
}

[[nodiscard]] constexpr MeshIndex to_meshidx(std::size_t size) noexcept {
    assert(size <=
           static_cast<std::size_t>(std::numeric_limits<MeshIndex>::max()));
    return static_cast<MeshIndex>(size);
}

[[nodiscard]] constexpr MeshIndex to_meshidx(int value) noexcept {
    assert(value >= 0);
    return static_cast<MeshIndex>(value);
}

[[nodiscard]] constexpr Index to_idx_safe(Index index) noexcept {
    return index;
}

[[nodiscard]] constexpr Index to_idx_safe(std::size_t size) {
    if (size > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        throw std::out_of_range(
            "to_idx_safe: size_t value exceeds Index range");
    }
    return static_cast<Index>(size);
}

[[nodiscard]] constexpr Index to_idx_safe(MeshIndex mesh_index) noexcept {
    return static_cast<Index>(mesh_index);
}

[[nodiscard]] constexpr Index to_idx_safe(int value) noexcept {
    return static_cast<Index>(value);
}

[[nodiscard]] constexpr std::size_t to_sizet_safe(std::size_t size) noexcept {
    return size;
}

[[nodiscard]] constexpr std::size_t to_sizet_safe(Index index) {
    if (index < 0) {
        throw std::out_of_range("to_sizet_safe: negative Index");
    }
    return static_cast<std::size_t>(index);
}

[[nodiscard]] constexpr std::size_t to_sizet_safe(
    MeshIndex mesh_index) noexcept {
    return static_cast<std::size_t>(mesh_index);
}

[[nodiscard]] constexpr std::size_t to_sizet_safe(int value) {
    if (value < 0) {
        throw std::out_of_range("to_sizet_safe: negative int");
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] constexpr MeshIndex to_meshidx_safe(
    MeshIndex mesh_index) noexcept {
    return mesh_index;
}

[[nodiscard]] constexpr MeshIndex to_meshidx_safe(Index index) {
    if (index < 0 ||
        index > static_cast<Index>(std::numeric_limits<MeshIndex>::max())) {
        throw std::out_of_range(
            "to_meshidx_safe: Index out of MeshIndex range");
    }
    return static_cast<MeshIndex>(index);
}

[[nodiscard]] constexpr MeshIndex to_meshidx_safe(std::size_t size) {
    if (size >
        static_cast<std::size_t>(std::numeric_limits<MeshIndex>::max())) {
        throw std::out_of_range(
            "to_meshidx_safe: size_t out of MeshIndex range");
    }
    return static_cast<MeshIndex>(size);
}

[[nodiscard]] constexpr MeshIndex to_meshidx_safe(int value) {
    if (value < 0) {
        throw std::out_of_range("to_meshidx_safe: negative int");
    }
    return static_cast<MeshIndex>(value);
}

constexpr MeshIndex INVALID_MESH_INDEX = std::numeric_limits<MeshIndex>::max();

}  // namespace pycanha
