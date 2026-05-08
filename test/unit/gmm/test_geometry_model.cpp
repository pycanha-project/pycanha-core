#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <numbers>
#include <optional>
#include <sstream>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/cylinder.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/primitives/sphere.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/cut_group.hpp"
#include "pycanha-core/gmm/scene/group.hpp"
#include "pycanha-core/gmm/scene/item.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::CutGroup;
using pycanha::gmm::Cylinder;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::Group;
using pycanha::gmm::Item;
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

[[nodiscard]] Item make_panel_item() {
    return Item(Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
                ThermalMesh{{0.0, 0.5, 1.0}, {0.0, 1.0}});
}

[[nodiscard]] Sphere make_sphere_cutter() {
    using std::numbers::pi;
    return {
        {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 1.0, -1.0, 1.0, 0.0,
        2.0 * pi};
}

}  // namespace

TEST_CASE("GeometryModel manages hierarchy and lookups",
          "[gmm][geometrymodel]") {
    GeometryModel model("scene");

    const auto rig_id = model.add_group(
        "rig",
        Group(CoordinateTransformation::from_translation({1.0, 0.0, 0.0})));
    const auto trim_id = model.add_cut_group("trim", CutGroup{});
    const auto panel_id = model.add_item("panel", make_panel_item(), "rig");

    REQUIRE(model.contains("rig"));
    REQUIRE(model.contains("panel"));
    REQUIRE(model.id_optional("rig") == rig_id);
    REQUIRE(model.id_optional("panel") == panel_id);
    REQUIRE(model.name_of(panel_id) == std::optional<std::string>{"panel"});
    REQUIRE(model.group_optional("rig") != nullptr);
    REQUIRE(model.cut_group_optional("trim") != nullptr);
    REQUIRE(model.item_optional("panel") != nullptr);
    REQUIRE(model.group_optional("panel") == nullptr);
    REQUIRE(model.cut_group_optional("rig") == nullptr);
    REQUIRE(model.group_optional("rig")->child_item_indices().size() == 1U);
    REQUIRE(model.group_optional("rig")->child_group_indices().empty());
    REQUIRE(model.group_optional("rig")->child_cut_group_indices().empty());
    REQUIRE(model.group_optional("trim") == nullptr);
    REQUIRE(model.name_of(trim_id) == std::optional<std::string>{"trim"});

    REQUIRE_THROWS(model.add_item("panel", make_panel_item()));
    REQUIRE_THROWS(
        model.add_item("ghost", make_panel_item(), "missing-parent"));
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
    model.add_group("rig", Group{});
    model.add_group("other", Group{});
    model.add_group("child", Group{}, "rig");
    model.add_item("panel", make_panel_item(), "rig");
    model.add_item("tube", make_panel_item(), "other");

    REQUIRE_THROWS(model.rename("panel", "tube"));
    REQUIRE_THROWS(model.reparent("rig", "child"));

    model.reparent("panel", "other");
    REQUIRE(model.group_optional("rig")->child_item_indices().empty());
    REQUIRE(model.group_optional("other")->child_item_indices().size() == 2U);

    const auto version_before_rename = model.get_structure_version();
    model.rename("panel", "panel-renamed");
    REQUIRE(model.contains("panel-renamed"));
    REQUIRE_FALSE(model.contains("panel"));
    REQUIRE(model.get_structure_version() == version_before_rename + 1U);
}

TEST_CASE("GeometryModel structure version tracks only structural changes",
          "[gmm][geometrymodel]") {
    GeometryModel model("scene");
    const auto version_before = model.get_structure_version();

    model.add_group("rig", Group{});
    REQUIRE(model.get_structure_version() == version_before + 1U);
    model.add_item("panel", make_panel_item(), "rig");
    REQUIRE(model.get_structure_version() == version_before + 2U);
    model.add_cut_group("trim", CutGroup{});
    REQUIRE(model.get_structure_version() == version_before + 3U);

    const auto version_before_mesh_options = model.get_structure_version();
    model.set_default_mesh_options({1.0e-4});
    REQUIRE(model.get_structure_version() == version_before_mesh_options);

    auto* item = model.item_optional("panel");
    REQUIRE(item != nullptr);
    item->set_primitive(
        Rectangle({0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
    REQUIRE(model.get_structure_version() == version_before_mesh_options);

    const auto version_before_remove = model.get_structure_version();
    model.remove("trim");
    REQUIRE(model.get_structure_version() == version_before_remove + 1U);
}

TEST_CASE("CutGroup rejects non-solid primitives", "[gmm][geometrymodel]") {
    using std::numbers::pi;

    CutGroup cut_group;
    REQUIRE_NOTHROW(cut_group.add_cutter(make_sphere_cutter()));
    REQUIRE_NOTHROW(
        cut_group.add_cutter(Cylinder({0.0, 0.0, 0.0}, {0.0, 0.0, 2.0},
                                      {1.0, 0.0, 0.0}, 0.5, 0.0, 2.0 * pi)));
    REQUIRE_THROWS(cut_group.add_cutter(
        Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0})));
}
