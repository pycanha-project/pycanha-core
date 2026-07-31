// Metal-only: the instance transform crosses a layout boundary the Vulkan
// backend does not have. The shader reads the transform as three row vectors
// (world = R * p + t), but a Metal instance descriptor stores it as
// MTLPackedFloat4x3 — column-major, with the translation in the last column.
// Getting that transpose wrong still traces rays, just against geometry
// placed somewhere else, so it is checked against the same
// gmm::CoordinateTransformation the scene was built from.

#import <Metal/Metal.h>

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "radiative/mtl_scene.hpp"

namespace {

namespace detail = pycanha::radiative::detail;

// What the GPU does with the instance descriptor: world = M * (p, 1) with M
// stored column-major, columns[3] being the translation.
[[nodiscard]] pycanha::Vector3D apply_packed(const MTLPackedFloat4x3& matrix,
                                             const pycanha::Vector3D& point) {
    pycanha::Vector3D out = pycanha::Vector3D::Zero();
    for (std::size_t row = 0; row < 3; ++row) {
        double value = matrix.columns[3].elements[row];
        for (std::size_t col = 0; col < 3; ++col) {
            value += static_cast<double>(matrix.columns[col].elements[row]) *
                     point(static_cast<Eigen::Index>(col));
        }
        out(static_cast<Eigen::Index>(row)) = value;
    }
    return out;
}

// Same packing the scene does when it fills InstanceData for the shader.
[[nodiscard]] detail::InstanceDataGpu instance_from(
    const pycanha::gmm::CoordinateTransformation& tf) {
    detail::InstanceDataGpu instance{};
    for (std::size_t row = 0; row < 3; ++row) {
        const auto row_index = static_cast<Eigen::Index>(row);
        for (std::size_t col = 0; col < 3; ++col) {
            const auto col_index = static_cast<Eigen::Index>(col);
            instance.tf_rows.at(row).at(col) =
                static_cast<float>(tf.rotation()(row_index, col_index));
        }
        instance.tf_rows.at(row).at(3) =
            static_cast<float>(tf.translation()(row_index));
    }
    return instance;
}

}  // namespace

TEST_CASE("radiative Metal: instance transform matches the scene transform",
          "[radiative][metal]") {
    // A rotation about each axis plus a translation, so a transposed
    // rotation or a translation written into the wrong column cannot pass.
    const pycanha::gmm::CoordinateTransformation tf =
        pycanha::gmm::CoordinateTransformation::from_euler(
            pycanha::Vector3D(1.5, -2.25, 0.75),
            pycanha::Vector3D(0.3, -0.7, 1.1));
    const detail::InstanceDataGpu instance = instance_from(tf);
    const MTLPackedFloat4x3 packed = detail::to_mtl_transform(instance);

    const std::array<pycanha::Vector3D, 4> points = {
        pycanha::Vector3D(0.0, 0.0, 0.0), pycanha::Vector3D(1.0, 0.0, 0.0),
        pycanha::Vector3D(0.0, 1.0, 0.0), pycanha::Vector3D(-2.0, 3.5, 4.25)};
    for (const pycanha::Vector3D& point : points) {
        const pycanha::Vector3D expected = tf.apply(point);
        const pycanha::Vector3D actual = apply_packed(packed, point);
        // f32 storage of the rotation is the only source of error.
        constexpr double tol = 1e-5;
        REQUIRE_THAT(actual.x(),
                     Catch::Matchers::WithinAbs(expected.x(), tol));
        REQUIRE_THAT(actual.y(),
                     Catch::Matchers::WithinAbs(expected.y(), tol));
        REQUIRE_THAT(actual.z(),
                     Catch::Matchers::WithinAbs(expected.z(), tol));
    }
}
