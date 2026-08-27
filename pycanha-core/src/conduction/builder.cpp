#include "pycanha-core/conduction/builder.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "pycanha-core/conduction/links.hpp"
#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/conduction/profile.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha::conduction {

namespace {

using gmm::GeometryItem;
using gmm::NO_NODE;

// What one node accumulates over all the faces that map to it.
struct NodeAccumulator {
    double capacitance = 0.0;
    double area = 0.0;
    Vector3D weighted_position = Vector3D::Zero();
    const gmm::BulkMaterial* bulk = nullptr;
    bool mixed_bulk = false;
};

// Per-face geometry read once off the world mesh, so the item transforms and
// the tessellation are accounted for exactly as the radiative path sees them.
struct SlotGeometry {
    std::vector<double> area;
    std::vector<Vector3D> weighted_position;
};

// Everything one side of one item contributes.
struct SideProperties {
    /// Takes part in either physics. That is the "this side of the shell
    /// exists" test, so it is what decides whether the side defines nodes and
    /// hands them its thermal mass, its area and its centroid.
    bool active = false;
    /// Takes part in conduction. Only the conductors depend on this: a
    /// radiative-only side still has a node and still carries its mass.
    bool conductive = false;
    bool has_nodes = false;
    double capacitance_per_area = 0.0;
    const gmm::BulkMaterial* bulk = nullptr;
};

struct ItemSides {
    SideProperties side1;
    SideProperties side2;

    [[nodiscard]] SideProperties& of(unsigned side) noexcept {
        return side == 1U ? side1 : side2;
    }
    [[nodiscard]] const SideProperties& of(unsigned side) const noexcept {
        return side == 1U ? side1 : side2;
    }
    [[nodiscard]] bool any_active() const noexcept {
        return side1.active || side2.active;
    }
    /// Some side both carries node numbers and takes part in a physics, so the
    /// item has nodes to contribute to.
    [[nodiscard]] bool contributes_nodes() const noexcept {
        return (side1.has_nodes && side1.active) ||
               (side2.has_nodes && side2.active);
    }
    /// The same, narrowed to conduction: whether any conductor can come out of
    /// this item.
    [[nodiscard]] bool contributes_conductors() const noexcept {
        return (side1.has_nodes && side1.conductive) ||
               (side2.has_nodes && side2.conductive);
    }
};

using NodePair = std::pair<NodeNum, NodeNum>;

// The whole build in flight: read-only geometry on one side, the node and
// conductor sets being assembled on the other. Nothing is written to the tmm
// until every item has been walked.
struct BuildContext {
    const gmm::TriMeshD* world_mesh;
    SlotGeometry faces;
    TmmBuildOptions options;
    std::map<NodeNum, NodeAccumulator> node_accumulators;
    std::map<NodePair, double> conductors;
    TmmBuildReport report;
};

[[nodiscard]] NodePair ordered_pair(NodeNum first, NodeNum second) noexcept {
    return first <= second ? NodePair{first, second} : NodePair{second, first};
}

[[nodiscard]] SlotGeometry slot_geometry(const gmm::TriMeshD& mesh) {
    const auto faces = to_sizet(to_idx(mesh.nf()));
    SlotGeometry geometry{
        .area = std::vector<double>(faces, 0.0),
        .weighted_position = std::vector<Vector3D>(faces, Vector3D::Zero())};

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const auto triangle = mesh.triangles.row(tri_idx);
        const Point3D vertex_0 = mesh.vertices.row(to_idx(triangle(0)));
        const Point3D vertex_1 = mesh.vertices.row(to_idx(triangle(1)));
        const Point3D vertex_2 = mesh.vertices.row(to_idx(triangle(2)));
        const double area =
            0.5 * (vertex_1 - vertex_0).cross(vertex_2 - vertex_0).norm();
        const Point3D centroid = (vertex_0 + vertex_1 + vertex_2) / 3.0;

        // The two sides of a face pair share the pair's area and centroid.
        const auto face = to_sizet(to_idx(mesh.face_ids(tri_idx)));
        geometry.area[face] += area;
        geometry.area[face + 1U] += area;
        geometry.weighted_position[face] += area * centroid;
        geometry.weighted_position[face + 1U] += area * centroid;
    }
    return geometry;
}

// Whether a diagnostic describes expected behaviour rather than something the
// build had to drop or assume. The discrete link path, a face-pair band
// reaching the axis and a side excluded by the active-side selector are all
// normal, as is an item meshed for the radiative path alone and therefore
// carrying no node numbers.
[[nodiscard]] bool is_benign(const DiagnosticCode code) noexcept {
    switch (code) {
        case DiagnosticCode::DiscreteLinkFallback:
        case DiagnosticCode::AxisSingularity:
        case DiagnosticCode::InactiveSideSkipped:
        case DiagnosticCode::NoNodeNumbers:
            return true;
        default:
            return false;
    }
}

void log_diagnostic(const BuildDiagnostic& diagnostic) {
    if (is_benign(diagnostic.code)) {
        // One record per item, so this scales with the model rather than with
        // the call. The build's summary line carries the outcome instead.
        SPDLOG_LOGGER_DEBUG(pycanha::get_logger(), "build_tmm_from_gmm: {}",
                            diagnostic.message);
        return;
    }

    SPDLOG_LOGGER_WARN(pycanha::get_logger(), "build_tmm_from_gmm: {}",
                       diagnostic.message);
}

void report_diagnostic(TmmBuildReport& report, DiagnosticCode code,
                       std::string geometry_name, std::string message) {
    report.diagnostics.push_back(
        BuildDiagnostic{.code = code,
                        .geometry_name = std::move(geometry_name),
                        .message = std::move(message)});
    log_diagnostic(report.diagnostics.back());
}

// Collects the items that can contribute, reporting the ones that cannot.
// The walk carries an explicit stack rather than recursing, so a deeply nested
// scene cannot overflow it.
[[nodiscard]] std::vector<std::shared_ptr<GeometryItem>> collect_items(
    const gmm::GeometryModel& geometry, TmmBuildReport& report) {
    struct Pending {
        std::shared_ptr<gmm::Geometry> node;
        bool inside_cut;
    };

    std::vector<std::shared_ptr<GeometryItem>> items;
    std::vector<Pending> stack;
    const auto push_children =
        [&stack](std::span<const std::shared_ptr<gmm::Geometry>> children,
                 bool inside_cut) {
            std::ranges::transform(
                children, std::back_inserter(stack),
                [inside_cut](const std::shared_ptr<gmm::Geometry>& child) {
                    return Pending{.node = child, .inside_cut = inside_cut};
                });
        };
    push_children(geometry.children(), /*inside_cut=*/false);

    while (!stack.empty()) {
        const Pending pending = stack.back();
        stack.pop_back();
        bool inside_cut = pending.inside_cut;

        if (const auto cut_group =
                std::dynamic_pointer_cast<gmm::GeometryGroupCutted>(
                    pending.node);
            cut_group != nullptr && !inside_cut) {
            report_diagnostic(report, DiagnosticCode::CutGeometrySkipped,
                              cut_group->name(),
                              "cut group '" + cut_group->name() +
                                  "' skipped: a boolean-cut mesh has no intact "
                                  "face-pair grid to integrate over");
            inside_cut = true;
        }

        if (const auto item =
                std::dynamic_pointer_cast<GeometryItem>(pending.node);
            item != nullptr) {
            if (inside_cut) {
                ++report.items_skipped;
            } else {
                items.push_back(item);
            }
            continue;
        }

        push_children(pending.node->children(), inside_cut);
    }
    return items;
}

// A band reaching the axis of revolution is the one place the plain
// around-the-axis integral does not apply, because the temperature difference
// between two neighbouring angular face pairs vanishes there instead of staying
// constant across the band. Worth recording which form was used.
void report_axis_singularity(const MeridianProfile& profile,
                             const gmm::ThermalMesh& thermal_mesh,
                             const std::string& item_name,
                             TmmBuildReport& report) {
    if (thermal_mesh.get_dir1_mesh().size() < 3U) {
        // A single angular face pair has no around-the-axis conductor at all,
        // not even a ring closure, so nothing is truncated.
        return;
    }
    if (!profile.on_axis(0.0) && !profile.on_axis(1.0)) {
        return;
    }
    report_diagnostic(
        report, DiagnosticCode::AxisSingularity, item_name,
        "'" + item_name +
            "' has a face-pair band reaching the axis of revolution; its "
            "around-the-axis conductance uses the near-axis form (meridian "
            "length over the reference radius), since the temperature "
            "difference between angular face pairs vanishes at the axis");
}

void describe_side(const gmm::ThermalMesh& thermal_mesh, unsigned side,
                   const std::string& item_name, SideProperties& properties,
                   TmmBuildReport& report) {
    // A side that takes part in either physics exists as far as the tmm is
    // concerned: it defines nodes and hands them the shell's thermal mass on
    // that side, whether or not it also conducts. Conduction is gated
    // separately and only reaches the conductors.
    properties.active = thermal_mesh.is_side_active(side);
    properties.conductive = thermal_mesh.is_conductive_active(side);

    // Only a side that carries node numbers can contribute, so has_nodes gates
    // every diagnostic below: a shell that is simply single-sided must not fill
    // the report with noise.
    if (!properties.has_nodes) {
        return;
    }
    if (!properties.active) {
        report_diagnostic(report, DiagnosticCode::InactiveSideSkipped,
                          item_name,
                          "'" + item_name + "' side " + std::to_string(side) +
                              " carries node numbers but takes part in neither "
                              "physics: no node, no capacitance, no conductor");
        return;
    }
    if (!properties.conductive) {
        report_diagnostic(report, DiagnosticCode::InactiveSideSkipped,
                          item_name,
                          "'" + item_name + "' side " + std::to_string(side) +
                              " is radiative only: it defines nodes and gives "
                              "them its capacitance, but no conductor");
    }

    const auto& material = side == 1U ? thermal_mesh.get_side1_material()
                                      : thermal_mesh.get_side2_material();
    const double thickness = side == 1U ? thermal_mesh.get_side1_thick()
                                        : thermal_mesh.get_side2_thick();
    if (material == nullptr) {
        report_diagnostic(report, DiagnosticCode::MissingBulk, item_name,
                          "'" + item_name + "' side " + std::to_string(side) +
                              " is active but has no bulk material: its nodes "
                              "get no capacitance and no conductors");
        return;
    }
    properties.bulk = material.get();
    if (thickness <= 0.0) {
        report_diagnostic(report, DiagnosticCode::ZeroThickness, item_name,
                          "'" + item_name + "' side " + std::to_string(side) +
                              " has zero thickness: no capacitance and no "
                              "conductors");
        return;
    }
    // Conductivity only ever feeds the conductors, so a side that does not
    // conduct has nothing to say about it.
    if (properties.conductive && material->get_conductivity() <= 0.0) {
        report_diagnostic(report, DiagnosticCode::ZeroConductivity, item_name,
                          "'" + item_name + "' side " + std::to_string(side) +
                              " has zero conductivity: it carries capacitance "
                              "but no conductors");
    }
    properties.capacitance_per_area =
        material->get_density() * material->get_specific_heat() * thickness;
}

void accumulate_node(NodeAccumulator& accumulator, double area,
                     const Vector3D& weighted_position,
                     const SideProperties& properties) {
    accumulator.area += area;
    accumulator.capacitance += properties.capacitance_per_area * area;
    accumulator.weighted_position += weighted_position;
    if (properties.bulk == nullptr) {
        return;
    }
    if (accumulator.bulk == nullptr) {
        accumulator.bulk = properties.bulk;
    } else if (accumulator.bulk != properties.bulk) {
        accumulator.mixed_bulk = true;
    }
}

// One item's face-pair grid, resolved against the world mesh: the face of a
// face pair side and the node number it carries.
class ItemFacePairs {
  public:
    ItemFacePairs(const gmm::TriMeshD& world_mesh,
                  const gmm::TriMeshD::PrimitiveRange& range,
                  const gmm::ThermalMesh& thermal_mesh)
        : _world_mesh(&world_mesh),
          _first_face(to_idx(range.first_face_id)),
          _count((thermal_mesh.get_dir1_mesh().size() - 1U) *
                 (thermal_mesh.get_dir2_mesh().size() - 1U)) {}

    [[nodiscard]] std::size_t count() const noexcept { return _count; }

    [[nodiscard]] Eigen::Index face_of(std::size_t face_pair,
                                       unsigned side) const noexcept {
        return _first_face + to_idx(2U * face_pair) + to_idx(side - 1U);
    }

    // Node numbers come from the world mesh rather than from the ThermalMesh
    // directly, so face pairs the mesher dropped as degenerate stay unassigned
    // instead of inventing a node.
    [[nodiscard]] NodeNum node_of(std::size_t face_pair,
                                  unsigned side) const noexcept {
        const Eigen::Index face = face_of(face_pair, side);
        if (face >= _world_mesh->node_numbers.rows()) {
            return NO_NODE;
        }
        return _world_mesh->node_numbers(face);
    }

  private:
    const gmm::TriMeshD* _world_mesh;
    Eigen::Index _first_face;
    std::size_t _count;
};

void accumulate_item_nodes(const ItemFacePairs& face_pairs,
                           const ItemSides& sides, BuildContext& context) {
    for (std::size_t face_pair = 0; face_pair < face_pairs.count();
         ++face_pair) {
        for (const unsigned side : {1U, 2U}) {
            const SideProperties& properties = sides.of(side);
            // Per face, so when both sides map to one node only the sides that
            // are actually there hand it area and mass.
            if (!properties.active) {
                continue;
            }
            const NodeNum node = face_pairs.node_of(face_pair, side);
            if (node == NO_NODE) {
                continue;
            }
            const auto face = to_sizet(face_pairs.face_of(face_pair, side));
            accumulate_node(context.node_accumulators[node],
                            context.faces.area[face],
                            context.faces.weighted_position[face], properties);
        }
    }
}

void accumulate_item_conductors(const GeometryItem& item,
                                const ItemFacePairs& face_pairs,
                                BuildContext& context) {
    const gmm::ThermalMesh& thermal_mesh = item.thermal_mesh();

    if (context.options.intra_primitive_conductors) {
        const auto links = intra_primitive_links(item.primitive(), thermal_mesh,
                                                 context.options);
        context.report.face_pair_links_computed += links.size();
        for (const auto& link : links) {
            const NodeNum node_a =
                face_pairs.node_of(link.face_pair_a, link.side);
            const NodeNum node_b =
                face_pairs.node_of(link.face_pair_b, link.side);
            if (node_a == NO_NODE || node_b == NO_NODE || node_a == node_b) {
                continue;  // unassigned, or the same node on both ends
            }
            context.conductors[ordered_pair(node_a, node_b)] +=
                link.conductance;
        }
    }

    if (!context.options.through_thickness_conductors) {
        return;
    }
    for (std::size_t face_pair = 0; face_pair < face_pairs.count();
         ++face_pair) {
        const NodeNum node_1 = face_pairs.node_of(face_pair, 1U);
        const NodeNum node_2 = face_pairs.node_of(face_pair, 2U);
        if (node_1 == NO_NODE || node_2 == NO_NODE || node_1 == node_2) {
            continue;
        }
        const auto face = to_sizet(face_pairs.face_of(face_pair, 1U));
        const double conductance = through_thickness_conductance(
            thermal_mesh, context.faces.area[face]);
        if (conductance > 0.0) {
            context.conductors[ordered_pair(node_1, node_2)] += conductance;
        }
    }
}

void process_item(const std::shared_ptr<GeometryItem>& item,
                  const gmm::TriMeshD::PrimitiveRange& range,
                  BuildContext& context) {
    const std::string& item_name = item->name();
    const gmm::ThermalMesh& thermal_mesh = item->thermal_mesh();
    const ItemFacePairs face_pairs(*context.world_mesh, range, thermal_mesh);

    ItemSides sides;
    for (const unsigned side : {1U, 2U}) {
        SideProperties& properties = sides.of(side);
        for (std::size_t face_pair = 0; face_pair < face_pairs.count();
             ++face_pair) {
            if (face_pairs.node_of(face_pair, side) != NO_NODE) {
                properties.has_nodes = true;
                break;
            }
        }
        describe_side(thermal_mesh, side, item_name, properties,
                      context.report);
    }

    if (!sides.any_active()) {
        ++context.report.items_skipped;
        return;
    }
    if (!sides.contributes_nodes()) {
        report_diagnostic(
            context.report, DiagnosticCode::NoNodeNumbers, item_name,
            "'" + item_name +
                "' has no node numbers on any active side: it contributes "
                "nothing");
        ++context.report.items_skipped;
        return;
    }

    ++context.report.items_processed;
    accumulate_item_nodes(face_pairs, sides, context);

    // Both of these describe how the in-plane conductors are obtained, so
    // neither has anything to say when they are turned off, nor on an item
    // whose nodes come from radiation alone.
    if (context.options.intra_primitive_conductors &&
        sides.contributes_conductors()) {
        if (const auto profile = profile_of(item->primitive());
            profile.has_value()) {
            report_axis_singularity(*profile, thermal_mesh, item_name,
                                    context.report);
        } else {
            report_diagnostic(
                context.report, DiagnosticCode::DiscreteLinkFallback, item_name,
                "'" + item_name +
                    "' has no closed-form conduction profile, so its "
                    "conductors come from the discrete shared-edge path");
        }
    }

    accumulate_item_conductors(*item, face_pairs, context);
}

void require_empty_tmm(const ThermalMathematicalModel& tmm) {
    if (tmm.nodes().get_num_nodes() != 0) {
        throw std::invalid_argument(
            "build_tmm_from_gmm: the thermal mathematical model already has "
            "nodes; the builder needs an empty tmm");
    }
    if (tmm.conductive_couplings().matrices().get_num_total_couplings() != 0) {
        throw std::invalid_argument(
            "build_tmm_from_gmm: the thermal mathematical model already has "
            "conductive couplings; the builder needs an empty tmm");
    }
}

// Everything is known by now, so this cannot fail half-way through.
void commit(ThermalMathematicalModel& tmm, BuildContext& context) {
    for (const auto& [node_num, accumulator] : context.node_accumulators) {
        if (accumulator.mixed_bulk) {
            report_diagnostic(
                context.report, DiagnosticCode::MixedBulkOnNode, "",
                "node " + std::to_string(node_num) +
                    " gathers faces with different bulk materials; their "
                    "capacitances are summed as they stand");
        }

        Node node(node_num);
        node.set_type(DIFFUSIVE_NODE);
        node.set_T(context.options.initial_temperature);
        node.set_C(accumulator.capacitance);
        node.set_a(accumulator.area);
        const Vector3D position =
            accumulator.area > 0.0
                ? Vector3D{accumulator.weighted_position / accumulator.area}
                : Vector3D::Zero();
        node.set_fx(position.x());
        node.set_fy(position.y());
        node.set_fz(position.z());
        tmm.add_node(std::move(node));
        ++context.report.nodes_created;
    }

    for (const auto& [pair, conductance] : context.conductors) {
        if (conductance <= context.options.min_conductance) {
            continue;
        }
        tmm.add_conductive_coupling(pair.first, pair.second, conductance);
        ++context.report.conductors_created;
    }
}

}  // namespace

std::string_view to_string(DiagnosticCode code) noexcept {
    switch (code) {
        case DiagnosticCode::CutGeometrySkipped:
            return "CutGeometrySkipped";
        case DiagnosticCode::UnmeshedPrimitive:
            return "UnmeshedPrimitive";
        case DiagnosticCode::InactiveSideSkipped:
            return "InactiveSideSkipped";
        case DiagnosticCode::MissingBulk:
            return "MissingBulk";
        case DiagnosticCode::ZeroThickness:
            return "ZeroThickness";
        case DiagnosticCode::ZeroConductivity:
            return "ZeroConductivity";
        case DiagnosticCode::MixedBulkOnNode:
            return "MixedBulkOnNode";
        case DiagnosticCode::DiscreteLinkFallback:
            return "DiscreteLinkFallback";
        case DiagnosticCode::NoNodeNumbers:
            return "NoNodeNumbers";
        case DiagnosticCode::DegenerateFacePair:
            return "DegenerateFacePair";
        case DiagnosticCode::AxisSingularity:
            return "AxisSingularity";
    }
    return "Unknown";
}

TmmBuildReport build_tmm_from_gmm(ThermalModel& model,
                                  const TmmBuildOptions& options) {
    ThermalMathematicalModel& tmm = model.tmm();
    require_empty_tmm(tmm);

    const gmm::GeometryModel& geometry = model.gmm();
    const gmm::TriMeshD& world_mesh = geometry.root_group_mesh();
    BuildContext context{.world_mesh = &world_mesh,
                         .faces = slot_geometry(world_mesh),
                         .options = options,
                         .node_accumulators = {},
                         .conductors = {},
                         .report = {}};

    std::unordered_map<std::uint64_t, gmm::TriMeshD::PrimitiveRange> range_of;
    for (const auto& range : world_mesh.primitives) {
        range_of[static_cast<std::uint64_t>(range.geometry_id)] = range;
    }

    for (const auto& item : collect_items(geometry, context.report)) {
        if (std::holds_alternative<gmm::Cube>(item->primitive())) {
            // Defensive: a Cube is cutter-only, so the world mesh above would
            // already have refused it. Without this the item would fall
            // through to the triangle fallback below.
            ++context.report.items_skipped;
            report_diagnostic(context.report, DiagnosticCode::UnmeshedPrimitive,
                              item->name(),
                              "'" + item->name() +
                                  "' is a Cube, which is cutter-only and "
                                  "produces no faces, nodes or conductors");
            continue;
        }

        const auto range_it =
            range_of.find(static_cast<std::uint64_t>(item->id()));
        if (range_it == range_of.end()) {
            ++context.report.items_skipped;
            continue;
        }
        process_item(item, range_it->second, context);
    }

    commit(tmm, context);

    // One summary line, at the severity of the worst diagnostic: a clean build
    // stays off the console, one that dropped or assumed something announces
    // itself there and points at the detail already recorded above.
    const auto dropped = static_cast<std::size_t>(std::ranges::count_if(
        context.report.diagnostics, [](const BuildDiagnostic& diagnostic) {
            return !is_benign(diagnostic.code);
        }));
    if (dropped == 0U) {
        SPDLOG_LOGGER_INFO(
            pycanha::get_logger(),
            "Built tmm from gmm - {} items ({} skipped), {} nodes, "
            "{} conductors; no issues",
            context.report.items_processed, context.report.items_skipped,
            context.report.nodes_created, context.report.conductors_created);
    } else {
        SPDLOG_LOGGER_WARN(
            pycanha::get_logger(),
            "Built tmm from gmm - {} items ({} skipped), {} nodes, "
            "{} conductors; {} of {} diagnostics report something dropped or "
            "assumed",
            context.report.items_processed, context.report.items_skipped,
            context.report.nodes_created, context.report.conductors_created,
            dropped, context.report.diagnostics.size());
    }

    return std::move(context.report);
}

}  // namespace pycanha::conduction
