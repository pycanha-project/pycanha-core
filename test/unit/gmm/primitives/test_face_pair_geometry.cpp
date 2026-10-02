#include <array>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/face_pair_geometry.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/primitive.hpp"
#include "pycanha-core/gmm/primitives/quadrilateral.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/primitives/triangle.hpp"

// Catch2's assertion macros expand to branches, which the complexity check
// counts against every test case.
// NOLINTBEGIN(readability-function-cognitive-complexity)

namespace {

using pycanha::MeshIndex;
using pycanha::Point3D;
using pycanha::Vector3D;
using pycanha::gmm::Cone;
using pycanha::gmm::Cube;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Disc;
using pycanha::gmm::FacePairGeometry;
using pycanha::gmm::FacePairGeometryEvaluator;
using pycanha::gmm::Paraboloid;
using pycanha::gmm::Primitive;
using pycanha::gmm::Quadrilateral;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Sphere;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::Triangle;

constexpr double pi = std::numbers::pi;

// Uneven cuts, so no face pair is a scaled copy of another.
[[nodiscard]] std::vector<double> uneven_cuts(int count, double power) {
    std::vector<double> cuts;
    for (int cut = 0; cut <= count; ++cut) {
        cuts.push_back(std::pow(static_cast<double>(cut) / count, power));
    }
    return cuts;
}

// The surface point at cut fractions (s, t), sampled as the mesher samples
// it, through the primitive's own to_cartesian. For the paraboloid t is the
// square root of the height fraction, which keeps the map smooth at the apex.
using SurfaceMap = std::function<Point3D(double, double)>;

[[nodiscard]] SurfaceMap surface_map(const Primitive& primitive) {
    return std::visit(
        [](const auto& shape) -> SurfaceMap {
            using T = std::decay_t<decltype(shape)>;
            if constexpr (std::is_same_v<T, Rectangle> ||
                          std::is_same_v<T, Triangle> ||
                          std::is_same_v<T, Quadrilateral>) {
                return [shape](double s, double t) {
                    return shape.to_cartesian({s, t});
                };
            } else if constexpr (std::is_same_v<T, Disc>) {
                return [shape](double s, double t) {
                    const double angle =
                        shape.start_angle() +
                        (s * (shape.end_angle() - shape.start_angle()));
                    const double radius =
                        shape.inner_radius() +
                        (t * (shape.outer_radius() - shape.inner_radius()));
                    return shape.to_cartesian({angle * radius, radius});
                };
            } else if constexpr (std::is_same_v<T, Cylinder>) {
                return [shape](double s, double t) {
                    const double angle =
                        shape.start_angle() +
                        (s * (shape.end_angle() - shape.start_angle()));
                    return shape.to_cartesian(
                        {angle * shape.radius(),
                         t * (shape.p2() - shape.p1()).norm()});
                };
            } else if constexpr (std::is_same_v<T, Cone>) {
                return [shape](double s, double t) {
                    const double angle =
                        shape.start_angle() +
                        (s * (shape.end_angle() - shape.start_angle()));
                    const double radius =
                        shape.radius1() +
                        (t * (shape.radius2() - shape.radius1()));
                    return shape.to_cartesian(
                        {angle * radius, t * (shape.p2() - shape.p1()).norm()});
                };
            } else if constexpr (std::is_same_v<T, Sphere>) {
                return [shape](double s, double t) {
                    const double low =
                        std::asin(shape.base_truncation() / shape.radius());
                    const double high =
                        std::asin(shape.apex_truncation() / shape.radius());
                    const double longitude =
                        shape.start_angle() +
                        (s * (shape.end_angle() - shape.start_angle()));
                    const double latitude = low + (t * (high - low));
                    return shape.to_cartesian(
                        {shape.radius() * longitude * std::cos(latitude),
                         shape.radius() * latitude});
                };
            } else if constexpr (std::is_same_v<T, Paraboloid>) {
                return [shape](double s, double u) {
                    const double angle =
                        shape.start_angle() +
                        (s * (shape.end_angle() - shape.start_angle()));
                    const double radius = shape.radius() * u;
                    return shape.to_cartesian(
                        {angle * radius,
                         u * u * (shape.p2() - shape.p1()).norm()});
                };
            } else {
                return [](double, double) { return Point3D::Zero().eval(); };
            }
        },
        primitive);
}

// Area and centroid of the region [s0, s1] x [t0, t1] of a smooth map,
// independent of the evaluator: 20 x 20 Gauss-Legendre on |P_s x P_t|, with
// fourth-order central differences for the derivatives.
[[nodiscard]] FacePairGeometry quadrature(const SurfaceMap& map, double s0,
                                          double s1, double t0, double t1) {
    constexpr std::array<double, 10> half_nodes{
        0.0765265211334973, 0.2277858511416451, 0.3737060887154195,
        0.5108670019508271, 0.6360536807265150, 0.7463319064601508,
        0.8391169718222188, 0.9122344282513259, 0.9639719272779138,
        0.9931285991850949};
    constexpr std::array<double, 10> half_weights{
        0.1527533871307258, 0.1491729864726037, 0.1420961093183820,
        0.1316886384491766, 0.1181945319615184, 0.1019301198172404,
        0.0832767415767048, 0.0626720483341091, 0.0406014298003869,
        0.0176140071391521};
    // The rule is symmetric: the negative nodes mirror the positive ones.
    std::vector<std::pair<double, double>> rule;
    for (std::size_t point = half_nodes.size(); point-- > 0;) {
        rule.emplace_back(-half_nodes.at(point), half_weights.at(point));
    }
    for (std::size_t point = 0; point < half_nodes.size(); ++point) {
        rule.emplace_back(half_nodes.at(point), half_weights.at(point));
    }
    // Returns a concrete vector: an Eigen expression here would outlive the
    // temporaries it refers to.
    const auto derivative = [](const std::function<Point3D(double)>& along,
                               double at, double step) -> Vector3D {
        return ((8.0 * (along(at + step) - along(at - step))) -
                (along(at + (2.0 * step)) - along(at - (2.0 * step)))) /
               (12.0 * step);
    };
    FacePairGeometry result;
    Vector3D moment = Vector3D::Zero();
    const double step = 1e-4;
    for (const auto& [xa, wa] : rule) {
        const double s = ((s0 + s1) / 2.0) + ((s1 - s0) / 2.0 * xa);
        for (const auto& [xb, wb] : rule) {
            const double t = ((t0 + t1) / 2.0) + ((t1 - t0) / 2.0 * xb);
            const Vector3D along_s =
                derivative([&](double v) { return map(v, t); }, s, step);
            const Vector3D along_t =
                derivative([&](double v) { return map(s, v); }, t, step);
            const double element = along_s.cross(along_t).norm() * wa * wb *
                                   ((s1 - s0) / 2.0) * ((t1 - t0) / 2.0);
            result.area += element;
            moment += element * map(s, t);
        }
    }
    result.centroid = moment / result.area;
    return result;
}

struct Case {
    std::string name;
    Primitive primitive;
    double total_area;
};

[[nodiscard]] std::vector<Case> cases() {
    const Point3D origin{0.2, -0.4, 1.1};
    const Point3D axis{0.2, -0.4, 2.1};  // unit axis along z from origin
    const Point3D reference{1.2, -0.4, 1.1};
    std::vector<Case> all;
    const auto add = [&all](std::string name, Primitive primitive) {
        const double area = std::visit(
            [](const auto& shape) { return shape.surface_area(); }, primitive);
        all.push_back({.name = std::move(name),
                       .primitive = std::move(primitive),
                       .total_area = area});
    };
    add("rectangle", Rectangle({0, 0, 0}, {2.0, 0.5, 0}, {-0.25, 1.0, 0.3}));
    add("triangle", Triangle({0, 0, 0}, {1.0, 0.1, 0}, {0.2, 0.9, 0.3}));
    add("quadrilateral",
        Quadrilateral({0, 0, 0}, {1.0, 0, 0}, {1.3, 0.8, 0}, {-0.1, 0.6, 0}));
    add("disc", Disc(origin, axis, reference, 0.0, 0.5, 0.0, 2.0 * pi));
    add("annulus", Disc(origin, axis, reference, 0.1, 0.5, 0.3, 2.0));
    add("cylinder", Cylinder(origin, axis, reference, 0.4, -0.5, 1.5));
    add("cone", Cone(origin, axis, reference, 0.5, 0.2, 0.0, 2.0 * pi));
    add("apex cone", Cone(origin, axis, reference, 0.5, 0.0, 0.2, 3.0));
    add("sphere",
        Sphere(origin, axis, reference, 0.7, -0.7, 0.7, 0.0, 2.0 * pi));
    add("sphere band",
        Sphere(origin, axis, reference, 0.7, -0.3, 0.5, 0.2, 2.5));
    add("paraboloid",
        Paraboloid(origin, {0.2, -0.4, 1.9}, reference, 0.5, 0.0, 2.0 * pi));
    return all;
}

}  // namespace

TEST_CASE("face-pair areas add up to the primitive's surface area",
          "[gmm][face_pair_geometry]") {
    for (const Case& shape : cases()) {
        INFO(shape.name);
        for (const auto& [dir1, dir2] :
             {std::pair{1, 1}, std::pair{7, 5}, std::pair{64, 48}}) {
            const ThermalMesh mesh(uneven_cuts(dir1, 1.3),
                                   uneven_cuts(dir2, 0.8));
            const FacePairGeometryEvaluator evaluator(shape.primitive, mesh);
            REQUIRE(evaluator.is_supported());
            double total = 0.0;
            for (int j = 0; j < dir2; ++j) {
                for (int i = 0; i < dir1; ++i) {
                    const FacePairGeometry face = evaluator(
                        static_cast<MeshIndex>(i), static_cast<MeshIndex>(j));
                    REQUIRE(face.area > 0.0);
                    total += face.area;
                }
            }
            REQUIRE(std::abs(total - shape.total_area) <=
                    1e-12 * shape.total_area);
        }
    }
}

TEST_CASE("face-pair area and centroid match an independent quadrature",
          "[gmm][face_pair_geometry]") {
    for (const Case& shape : cases()) {
        INFO(shape.name);
        const std::vector<double> dir1 = uneven_cuts(5, 1.3);
        const std::vector<double> dir2 = uneven_cuts(4, 0.8);
        const ThermalMesh mesh(dir1, dir2);
        const FacePairGeometryEvaluator evaluator(shape.primitive, mesh);
        const SurfaceMap map = surface_map(shape.primitive);
        const bool paraboloid =
            std::holds_alternative<Paraboloid>(shape.primitive);
        for (std::size_t j = 0; j + 1U < dir2.size(); ++j) {
            for (std::size_t i = 0; i + 1U < dir1.size(); ++i) {
                const FacePairGeometry exact = evaluator(
                    static_cast<MeshIndex>(i), static_cast<MeshIndex>(j));
                const FacePairGeometry reference = quadrature(
                    map, dir1[i], dir1[i + 1U],
                    paraboloid ? std::sqrt(dir2[j]) : dir2[j],
                    paraboloid ? std::sqrt(dir2[j + 1U]) : dir2[j + 1U]);
                REQUIRE(std::abs(exact.area - reference.area) <=
                        1e-9 * reference.area);
                REQUIRE((exact.centroid - reference.centroid).norm() <= 1e-9);
            }
        }
    }
}

TEST_CASE("face-pair geometry of flat primitives equals their triangles",
          "[gmm][face_pair_geometry]") {
    // A rectangle cut unevenly: every face pair is a rectangle whose area and
    // centre are known in closed form.
    const Rectangle plate({0, 0, 0}, {2.0, 0, 0}, {0, 0.5, 0});
    const std::vector<double> dir1{0.0, 0.1, 0.35, 1.0};
    const std::vector<double> dir2{0.0, 0.6, 1.0};
    const FacePairGeometryEvaluator evaluator(plate, ThermalMesh(dir1, dir2));
    for (std::size_t j = 0; j + 1U < dir2.size(); ++j) {
        for (std::size_t i = 0; i + 1U < dir1.size(); ++i) {
            const FacePairGeometry face =
                evaluator(static_cast<MeshIndex>(i), static_cast<MeshIndex>(j));
            const double width = 2.0 * (dir1[i + 1U] - dir1[i]);
            const double height = 0.5 * (dir2[j + 1U] - dir2[j]);
            REQUIRE(std::abs(face.area - (width * height)) <= 1e-15);
            REQUIRE(std::abs(face.centroid.x() -
                             (1.0 * (dir1[i] + dir1[i + 1U]))) <= 1e-15);
            REQUIRE(std::abs(face.centroid.y() -
                             (0.25 * (dir2[j] + dir2[j + 1U]))) <= 1e-15);
        }
    }
}

TEST_CASE("a collapsed cut interval has zero area",
          "[gmm][face_pair_geometry]") {
    const Cylinder tube({0, 0, 0}, {0, 0, 1}, {1, 0, 0}, 0.5, 0.0, 2.0 * pi);
    const FacePairGeometryEvaluator evaluator(
        tube, ThermalMesh({0.0, 0.5, 0.5, 1.0}, {0.0, 1.0}));
    REQUIRE(evaluator(1, 0).area == 0.0);
    REQUIRE(evaluator(0, 0).area > 0.0);
    REQUIRE_FALSE(
        FacePairGeometryEvaluator(Cube({0, 0, 0}, {1, 1, 1}), ThermalMesh{})
            .is_supported());
}

// NOLINTEND(readability-function-cognitive-complexity)
