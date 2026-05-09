#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <map>
#include <numbers>
#include <string>

#include "pycanha-core/gmm/cutting/manifold_cut_backend.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/mesh_options.hpp"
#include "pycanha-core/gmm/mesh/ops/sort.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cone.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/disc.hpp"
#include "pycanha-core/gmm/primitives/paraboloid.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/scene/cut_group.hpp"
#include "pycanha-core/gmm/scene/item.hpp"

namespace {

using pycanha::gmm::Cone;
using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cube;
using pycanha::gmm::CutGroup;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Disc;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::Item;
using pycanha::gmm::MeshOptions;
using pycanha::gmm::Paraboloid;
using pycanha::gmm::Primitive;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Sphere;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriMesh;
namespace cutting = pycanha::gmm::cutting;
namespace mesh_ops = pycanha::gmm::mesh::ops;

[[nodiscard]] std::string material_name(std::uint64_t face_id) {
    return "face_" + std::to_string(face_id);
}

[[nodiscard]] std::array<double, 3> material_color(std::size_t material_index,
                                                   std::size_t material_count) {
    const double hue = material_count > 0U
                           ? static_cast<double>(material_index) /
                                 static_cast<double>(material_count)
                           : 0.0;
    const double scaled_hue = hue * 6.0;
    const double chroma = 0.75;
    const double secondary =
        chroma * (1.0 - std::abs(std::fmod(scaled_hue, 2.0) - 1.0));
    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;

    if (scaled_hue < 1.0) {
        red = chroma;
        green = secondary;
    } else if (scaled_hue < 2.0) {
        red = secondary;
        green = chroma;
    } else if (scaled_hue < 3.0) {
        green = chroma;
        blue = secondary;
    } else if (scaled_hue < 4.0) {
        green = secondary;
        blue = chroma;
    } else if (scaled_hue < 5.0) {
        red = secondary;
        blue = chroma;
    } else {
        red = chroma;
        blue = secondary;
    }

    const double match = 0.2;
    return {red + match, green + match, blue + match};
}

void write_mtl(const std::filesystem::path& path,
               const std::map<std::uint64_t, std::size_t>& material_indices) {
    std::ofstream output(path);
    REQUIRE(output.is_open());

    output << std::fixed << std::setprecision(6);
    for (const auto& [face_id, material_index] : material_indices) {
        const auto [red, green, blue] =
            material_color(material_index, material_indices.size());
        output << "newmtl " << material_name(face_id) << '\n';
        output << "Kd " << red << ' ' << green << ' ' << blue << '\n';
        output << "Ka 0.100000 0.100000 0.100000\n";
        output << "Ks 0.000000 0.000000 0.000000\n";
        output << "Ns 1.000000\n\n";
    }
}

void write_obj(const std::filesystem::path& path, TriMesh mesh) {
    if (mesh.triangles.rows() > 0) {
        mesh_ops::apply_permutation(mesh,
                                    mesh_ops::permutation_by_face_id(mesh));
    }

    std::map<std::uint64_t, std::size_t> material_indices;
    for (Eigen::Index index = 0; index < mesh.face_ids.rows(); ++index) {
        material_indices.try_emplace(mesh.face_ids[index],
                                     material_indices.size());
    }

    const auto mtl_path = path.parent_path() / (path.stem().string() + ".mtl");
    write_mtl(mtl_path, material_indices);

    std::ofstream output(path);
    REQUIRE(output.is_open());
    output << "mtllib " << mtl_path.filename().string() << '\n';
    output << "o " << path.stem().string() << '\n';

    for (Eigen::Index index = 0; index < mesh.vertices.rows(); ++index) {
        output << "v " << mesh.vertices(index, 0) << ' '
               << mesh.vertices(index, 1) << ' ' << mesh.vertices(index, 2)
               << '\n';
    }

    std::uint64_t current_face_id = std::numeric_limits<std::uint64_t>::max();
    for (Eigen::Index index = 0; index < mesh.triangles.rows(); ++index) {
        const auto face_id = mesh.face_ids[index];
        if (face_id != current_face_id) {
            current_face_id = face_id;
            output << "g " << material_name(face_id) << '\n';
            output << "usemtl " << material_name(face_id) << '\n';
        }
        output << "f " << mesh.triangles(index, 0) + 1 << ' '
               << mesh.triangles(index, 1) + 1 << ' '
               << mesh.triangles(index, 2) + 1 << '\n';
    }
}

[[nodiscard]] std::filesystem::path cut_export_dir() {
    return std::filesystem::current_path() / ".." / ".." / ".." /
           "mesh_exports" / "cutting";
}

[[nodiscard]] Item make_panel_item() {
    return Item(Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}),
                ThermalMesh{{0.0, 0.25, 0.5, 1.0}, {0.0, 0.3, 0.6, 1.0}});
}

[[nodiscard]] Item make_disc_item() {
    return Item(Disc({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.45, 0.0, 0.0}, 0.0,
                     1.45, 0.0, 2.0 * std::numbers::pi),
                ThermalMesh{{0.0, 0.2, 0.4, 0.6, 0.8, 1.0},
                            {0.0, 0.25, 0.5, 0.75, 1.0}});
}

[[nodiscard]] Item make_cylinder_item() {
    return Item(
        Cylinder({0.0, 0.0, -1.4}, {0.0, 0.0, 1.4}, {1.0, 0.0, -1.4}, 1.0, 0.0,
                 2.0 * std::numbers::pi),
        ThermalMesh{{0.0, 0.2, 0.45, 0.7, 1.0}, {0.0, 0.25, 0.5, 0.75, 1.0}});
}

[[nodiscard]] Item make_sphere_item() {
    return Item(Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.2, 0.0, 0.0}, 1.2,
                       -1.2, 1.2, 0.0, 2.0 * std::numbers::pi),
                ThermalMesh{{0.0, 0.2, 0.4, 0.6, 0.8, 1.0},
                            {0.0, 0.25, 0.5, 0.75, 1.0}});
}

[[nodiscard]] Item make_paraboloid_item() {
    return Item(
        Paraboloid({0.0, 0.0, 0.0}, {0.0, 0.0, 2.6}, {1.5, 0.0, 0.0}, 1.5, 0.0,
                   2.0 * std::numbers::pi),
        ThermalMesh{{0.0, 0.25, 0.5, 0.75, 1.0}, {0.0, 0.25, 0.5, 0.75, 1.0}});
}

[[nodiscard]] TriMesh make_geometry_model_cut_mesh() {
    CutGroup trim;
    trim.add_cutter(Cylinder({1.0, 1.0, -1.0}, {1.0, 1.0, 1.0},
                             {1.3, 1.0, -1.0}, 0.3, 0.0,
                             2.0 * std::numbers::pi));

    GeometryModel model("scene");
    model.add_cut_group("trim", std::move(trim));
    model.add_item("panel", make_panel_item(), "trim");

    const auto& unified_mesh = model.unified_mesh();
    return {unified_mesh.vertices, unified_mesh.triangles,
            unified_mesh.face_ids};
}

}  // namespace

TEST_CASE("Cutting exports representative OBJ meshes", "[gmm][cutting][mesh]") {
    const auto export_dir = cut_export_dir();
    std::filesystem::create_directories(export_dir);

    const MeshOptions options{0.02};
    cutting::ManifoldCutBackend backend;

    const Item panel = make_panel_item();

    const std::array<Primitive, 1> single_cutter{
        Cylinder({1.0, 1.0, -1.0}, {1.0, 1.0, 1.0}, {1.35, 1.0, -1.0}, 0.35,
                 0.0, 2.0 * std::numbers::pi)};
    write_obj(export_dir / "cut_rectangle_cylinder.obj",
              backend.cut(panel, std::span<const Primitive>{single_cutter},
                          CoordinateTransformation{}, options));

    const std::array<Primitive, 2> double_cutter{
        Cylinder({0.7, 0.7, -1.0}, {0.7, 0.7, 1.0}, {0.95, 0.7, -1.0}, 0.25,
                 0.0, 2.0 * std::numbers::pi),
        Cylinder({1.35, 1.25, -1.0}, {1.35, 1.25, 1.0}, {1.58, 1.25, -1.0},
                 0.23, 0.0, 2.0 * std::numbers::pi)};
    write_obj(export_dir / "cut_rectangle_two_cylinders.obj",
              backend.cut(panel, std::span<const Primitive>{double_cutter},
                          CoordinateTransformation{}, options));

    const std::array<Primitive, 1> cylinder_cuts_cylinder{
        Cylinder({-1.6, 0.0, 0.0}, {1.6, 0.0, 0.0}, {-1.6, 0.35, 0.0}, 0.35,
                 0.0, 2.0 * std::numbers::pi)};
    write_obj(export_dir / "cylinder_cuts_cylinder.obj",
              backend.cut(make_cylinder_item(),
                          std::span<const Primitive>{cylinder_cuts_cylinder},
                          CoordinateTransformation{}, options));

    const std::array<Primitive, 1> sphere_cuts_disc{
        Sphere({0.25, 0.15, 0.0}, {0.25, 0.15, 1.0}, {0.8, 0.15, 0.0}, 0.55,
               -0.55, 0.55, 0.0, 2.0 * std::numbers::pi)};
    write_obj(export_dir / "sphere_cuts_disc.obj",
              backend.cut(make_disc_item(),
                          std::span<const Primitive>{sphere_cuts_disc},
                          CoordinateTransformation{}, options));

    const std::array<Primitive, 1> cone_cuts_sphere{
        Cone({0.35, 0.0, -1.5}, {0.35, 0.0, 1.5}, {0.95, 0.0, -1.5}, 0.6, 0.2,
             0.0, 2.0 * std::numbers::pi)};
    write_obj(export_dir / "cone_cuts_sphere.obj",
              backend.cut(make_sphere_item(),
                          std::span<const Primitive>{cone_cuts_sphere},
                          CoordinateTransformation{}, options));

    const std::array<Primitive, 1> cube_cuts_rectangle{
        Cube({1.05, 0.85, 0.0}, {0.8, 0.8, 0.7})};
    write_obj(
        export_dir / "cube_cuts_rectangle.obj",
        backend.cut(panel, std::span<const Primitive>{cube_cuts_rectangle},
                    CoordinateTransformation{}, options));

    const std::array<Primitive, 1> sphere_cuts_paraboloid{
        Sphere({0.3, 0.15, 0.9}, {0.3, 0.15, 1.9}, {1.0, 0.15, 0.9}, 0.7, -0.7,
               0.7, 0.0, 2.0 * std::numbers::pi)};
    write_obj(export_dir / "sphere_cuts_paraboloid.obj",
              backend.cut(make_paraboloid_item(),
                          std::span<const Primitive>{sphere_cuts_paraboloid},
                          CoordinateTransformation{}, options));

    const std::array<Primitive, 1> cylinder_cuts_sphere{
        Cylinder({-1.4, 0.2, 0.0}, {1.4, 0.2, 0.0}, {-1.4, 0.55, 0.0}, 0.35,
                 0.0, 2.0 * std::numbers::pi)};
    write_obj(export_dir / "cylinder_cuts_sphere.obj",
              backend.cut(make_sphere_item(),
                          std::span<const Primitive>{cylinder_cuts_sphere},
                          CoordinateTransformation{}, options));

    write_obj(export_dir / "cut_geometrymodel_panel.obj",
              make_geometry_model_cut_mesh());

    REQUIRE(std::filesystem::exists(export_dir / "cut_rectangle_cylinder.obj"));
    REQUIRE(std::filesystem::exists(export_dir / "cut_rectangle_cylinder.mtl"));
    REQUIRE(std::filesystem::exists(export_dir /
                                    "cut_rectangle_two_cylinders.obj"));
    REQUIRE(std::filesystem::exists(export_dir /
                                    "cut_rectangle_two_cylinders.mtl"));
    REQUIRE(std::filesystem::exists(export_dir / "cylinder_cuts_cylinder.obj"));
    REQUIRE(std::filesystem::exists(export_dir / "cylinder_cuts_cylinder.mtl"));
    REQUIRE(std::filesystem::exists(export_dir / "sphere_cuts_disc.obj"));
    REQUIRE(std::filesystem::exists(export_dir / "sphere_cuts_disc.mtl"));
    REQUIRE(std::filesystem::exists(export_dir / "cone_cuts_sphere.obj"));
    REQUIRE(std::filesystem::exists(export_dir / "cone_cuts_sphere.mtl"));
    REQUIRE(std::filesystem::exists(export_dir / "cube_cuts_rectangle.obj"));
    REQUIRE(std::filesystem::exists(export_dir / "cube_cuts_rectangle.mtl"));
    REQUIRE(std::filesystem::exists(export_dir / "sphere_cuts_paraboloid.obj"));
    REQUIRE(std::filesystem::exists(export_dir / "sphere_cuts_paraboloid.mtl"));
    REQUIRE(std::filesystem::exists(export_dir / "cylinder_cuts_sphere.obj"));
    REQUIRE(std::filesystem::exists(export_dir / "cylinder_cuts_sphere.mtl"));
    REQUIRE(
        std::filesystem::exists(export_dir / "cut_geometrymodel_panel.obj"));
    REQUIRE(
        std::filesystem::exists(export_dir / "cut_geometrymodel_panel.mtl"));
}