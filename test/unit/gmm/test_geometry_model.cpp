#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <numbers>
#include <sstream>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::Cylinder;
using pycanha::gmm::Geometry;
using pycanha::gmm::GeometryGroup;
using pycanha::gmm::GeometryGroupCutted;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::Rectangle;
using pycanha::gmm::Sphere;
using pycanha::gmm::ThermalMesh;

class LoggerRegistryGuard {
  public:
    LoggerRegistryGuard() { drop_loggers(); }
    ~LoggerRegistryGuard() { drop_loggers(); }

    LoggerRegistryGuard(const LoggerRegistryGuard&) = delete;
    LoggerRegistryGuard& operator=(const LoggerRegistryGuard&) = delete;
    LoggerRegistryGuard(LoggerRegistryGuard&&) = delete;
    LoggerRegistryGuard& operator=(LoggerRegistryGuard&&) = delete;

  private:
    static void drop_loggers() {
        spdlog::drop("pycanha-core");
        spdlog::drop("pycanha-core.profiling");
        spdlog::drop("pycanha-python");
    }
};

[[nodiscard]] std::shared_ptr<GeometryItem> make_panel(const char* name) {
    return std::make_shared<GeometryItem>(
        name, Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 1.0}});
}

[[nodiscard]] std::shared_ptr<GeometryItem> make_sphere_cutter(
    const char* name) {
    using std::numbers::pi;
    return std::make_shared<GeometryItem>(
        name,
        Sphere({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 1.0, -1.0,
               1.0, 0.0, 2.0 * pi),
        ThermalMesh{});
}

}  // namespace

TEST_CASE("GeometryModel manages hierarchy and lookups",
          "[gmm][geometrymodel]") {
    GeometryModel model("scene");

    auto rig = std::make_shared<GeometryGroup>(
        "rig", std::vector<std::shared_ptr<Geometry>>{},
        CoordinateTransformation::from_translation({1.0, 0.0, 0.0}));
    auto trim = std::make_shared<GeometryGroupCutted>(
        "trim", std::vector<std::shared_ptr<Geometry>>{},
        std::vector<std::shared_ptr<GeometryItem>>{});
    model.add(rig);
    model.add(trim);
    model.add(make_panel("panel"), "rig");

    REQUIRE(model.contains("rig"));
    REQUIRE(model.contains("panel"));
    REQUIRE(model.contains(rig));
    REQUIRE(model.get_group("rig") != nullptr);
    REQUIRE(model.get_cut_group("trim") != nullptr);
    REQUIRE(model.get_item("panel") != nullptr);
    REQUIRE(model.get_group("panel") == nullptr);
    REQUIRE(model.get_cut_group("rig") == nullptr);
    REQUIRE(model.get_group("rig")->children().size() == 1U);
    REQUIRE(model.get_group("rig")->children()[0]->name() == "panel");

    REQUIRE_THROWS(model.add(make_panel("panel")));  // duplicate name
    REQUIRE_THROWS(model.add(make_panel("ghost"), "missing-parent"));
}

TEST_CASE("GeometryModel remove unknown name logs warning",
          "[gmm][geometrymodel]") {
    const LoggerRegistryGuard logger_registry_guard;
    std::ostringstream output;
    auto logger = pycanha::create_ostream_logger("pycanha-core", output);
    spdlog::register_logger(logger);

    GeometryModel model("scene");
    model.remove("missing");
    logger->flush();

    REQUIRE(output.str().contains("Geometry 'missing' doesn't exist"));
}

TEST_CASE("GeometryModel rename and reparent enforce invariants",
          "[gmm][geometrymodel]") {
    GeometryModel model("scene");
    model.add(std::make_shared<GeometryGroup>("rig"));
    model.add(std::make_shared<GeometryGroup>("other"));
    model.add(std::make_shared<GeometryGroup>("child"), "rig");
    model.add(make_panel("panel"), "rig");
    model.add(make_panel("tube"), "other");

    REQUIRE_THROWS(model.rename("panel", "tube"));
    REQUIRE_THROWS(model.reparent("rig", "child"));  // cycle

    model.reparent("panel", "other");
    REQUIRE(model.get_group("rig")->children().size() == 1U);    // child group
    REQUIRE(model.get_group("other")->children().size() == 2U);  // tube, panel

    const auto version_before_rename = model.get_structure_version();
    model.rename("panel", "panel-renamed");
    REQUIRE(model.contains("panel-renamed"));
    REQUIRE_FALSE(model.contains("panel"));
    REQUIRE(model.get_structure_version() == version_before_rename + 1U);
}

TEST_CASE("GeometryModel structure version tracks structural changes",
          "[gmm][geometrymodel]") {
    GeometryModel model("scene");
    const auto version_before = model.get_structure_version();

    model.add(std::make_shared<GeometryGroup>("rig"));
    REQUIRE(model.get_structure_version() == version_before + 1U);
    model.add(make_panel("panel"), "rig");
    REQUIRE(model.get_structure_version() == version_before + 2U);
    model.add(std::make_shared<GeometryGroupCutted>(
                  "trim", std::vector<std::shared_ptr<Geometry>>{},
                  std::vector<std::shared_ptr<GeometryItem>>{}));
    REQUIRE(model.get_structure_version() == version_before + 3U);

    // Default mesh options do not bump the structure version.
    const auto version_before_mesh_options = model.get_structure_version();
    model.set_default_mesh_options({1.0e-4});
    REQUIRE(model.get_structure_version() == version_before_mesh_options);

    const auto version_before_remove = model.get_structure_version();
    model.remove("trim");
    REQUIRE(model.get_structure_version() == version_before_remove + 1U);
}

TEST_CASE("GeometryGroupCutted rejects non-solid cutters",
          "[gmm][geometrymodel]") {
    using std::numbers::pi;

    GeometryGroupCutted cut_group(
        "trim", std::vector<std::shared_ptr<Geometry>>{make_panel("target")},
        std::vector<std::shared_ptr<GeometryItem>>{});

    REQUIRE_NOTHROW(cut_group.cut_with(make_sphere_cutter("sphere")));
    REQUIRE_NOTHROW(cut_group.cut_with(std::make_shared<GeometryItem>(
        "tube",
        Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {1.0, 0.0, 0.0}, 0.5, 0.0,
                 2.0 * pi),
        ThermalMesh{})));
    REQUIRE_THROWS(cut_group.cut_with(make_panel("flat")));
}
