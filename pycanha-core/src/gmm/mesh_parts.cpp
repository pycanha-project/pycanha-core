// GeometryModel::mesh_parts / material_table — the GMM entry points feeding
// the radiative raytracer (rigid-part split + per-face material tables).

#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/materials/optical_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/gmm/scene/scene_mesh_detail.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/scene_part.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha::gmm {

namespace {

struct PartBuild {
    TriMeshD mesh;
    CoordinateTransformation transform;  // part -> world
    radiative::PartKind kind = radiative::PartKind::Spacecraft;
    bool encountered = false;
};

// One pending node of the iterative pre-order walk. `chain` holds the
// ancestor transforms from the current part's root down to the node's
// parent; `parent_to_world` places that parent frame in the world.
struct WalkFrame {
    std::reference_wrapper<const Geometry> node;
    std::size_t part_idx;
    CoordinateTransformation parent_to_world;
    std::vector<const CoordinateTransformation*> chain;
};

// Pushes `group`'s children onto the walk stack (reversed, so they pop in
// pre-order), all sharing the same part / placement / ancestor chain.
void push_children(std::vector<WalkFrame>& stack, const GeometryGroup& group,
                   std::size_t part_idx,
                   const CoordinateTransformation& parent_to_world,
                   const std::vector<const CoordinateTransformation*>& chain) {
    static_cast<void>(std::ranges::transform(
        std::views::reverse(group.children()), std::back_inserter(stack),
        [&](const std::shared_ptr<Geometry>& child) {
            return WalkFrame{.node = *child,
                             .part_idx = part_idx,
                             .parent_to_world = parent_to_world,
                             .chain = chain};
        }));
}

// Iterative DFS mirror of the GeometryGroup::mesh() walk that routes each
// leaf mesh into its part while threading ONE global face-id offset through
// all parts (so face ids match the unified model mesh exactly). Ancestor
// transforms are applied to each leaf sequentially, innermost first — the
// same per-vertex operations in the same order as the hierarchical walk, so
// part vertices are bit-identical to the model mesh's.
void walk_parts(
    const GeometryGroup& root,
    const std::unordered_map<const Geometry*, std::size_t>& split_to_part,
    std::vector<PartBuild>& parts) {
    pycanha::MeshIndex offset = 0;
    std::vector<WalkFrame> stack;
    push_children(stack, root, 0, root.transform(), {&root.transform()});

    while (!stack.empty()) {
        const WalkFrame frame = std::move(stack.back());
        stack.pop_back();
        const Geometry& node = frame.node.get();
        const auto* as_group = dynamic_cast<const GeometryGroup*>(&node);

        const auto split_it = split_to_part.find(&node);
        if (split_it != split_to_part.end()) {
            PartBuild& part = parts[split_it->second];
            part.encountered = true;
            part.transform = node.transform().compose(frame.parent_to_world);
            if (as_group != nullptr) {
                // Part frame = the group's own frame: children walk with an
                // empty ancestor chain.
                push_children(stack, *as_group, split_it->second,
                              part.transform, {});
            } else {
                // Leaf split target (item / cut group): its cached mesh is
                // in the parent frame (own transform applied) — undo it to
                // get the part-local frame.
                TriMeshD piece = node.mesh();
                detail::apply_transform_in_place(piece,
                                                 node.transform().inverse());
                detail::concatenate_offset(part.mesh, piece, offset);
            }
            continue;
        }

        if (as_group != nullptr) {
            std::vector<const CoordinateTransformation*> chain = frame.chain;
            chain.push_back(&node.transform());
            const CoordinateTransformation to_world =
                node.transform().compose(frame.parent_to_world);
            push_children(stack, *as_group, frame.part_idx, to_world, chain);
            continue;
        }

        // Leaf: copy the cached subtree mesh (parent frame) and hoist it to
        // the part frame, innermost ancestor transform first.
        TriMeshD piece = node.mesh();
        for (const auto* ancestor : std::views::reverse(frame.chain)) {
            detail::apply_transform_in_place(piece, *ancestor);
        }
        detail::concatenate_offset(parts[frame.part_idx].mesh, piece, offset);
    }
}

}  // namespace

std::vector<radiative::ScenePart> GeometryModel::mesh_parts(
    std::span<const std::string> split) const {
    // Part 0 is the remainder (world frame, identity transform); one part per
    // split name follows in the given order.
    std::vector<PartBuild> parts(split.size() + 1);
    parts[0].encountered = true;

    std::unordered_map<const Geometry*, std::size_t> split_to_part;
    for (std::size_t idx = 0; idx < split.size(); ++idx) {
        const auto target = find(split[idx]);
        if (target == nullptr) {
            throw std::invalid_argument(
                "GeometryModel::mesh_parts: unknown split name '" + split[idx] +
                "'");
        }
        if (!split_to_part.emplace(target.get(), idx + 1).second) {
            throw std::invalid_argument(
                "GeometryModel::mesh_parts: duplicate split name '" +
                split[idx] + "'");
        }
        parts[idx + 1].kind = radiative::PartKind::Articulated;
    }

    walk_parts(*_root, split_to_part, parts);

    for (std::size_t idx = 0; idx < split.size(); ++idx) {
        if (!parts[idx + 1].encountered) {
            throw std::invalid_argument(
                "GeometryModel::mesh_parts: split name '" + split[idx] +
                "' is nested inside a cut group and cannot form a rigid "
                "part");
        }
    }

    std::vector<radiative::ScenePart> result;
    result.reserve(parts.size());
    for (std::size_t idx = 0; idx < parts.size(); ++idx) {
        if (idx == 0 && !split.empty() && parts[idx].mesh.nt() == 0) {
            continue;  // omit an empty remainder part
        }
        result.push_back(radiative::ScenePart{
            .mesh = parts[idx].mesh.cast<float>(),
            .transform = parts[idx].transform,
            .kind = parts[idx].kind,
            .part_id = static_cast<std::uint32_t>(result.size())});
    }
    return result;
}

radiative::MaterialTable GeometryModel::material_table() const {
    const TriMeshF& model_mesh = mesh();
    const auto num_slots = static_cast<Eigen::Index>(model_mesh.nf());

    radiative::MaterialTable table;
    table.face_material = Eigen::VectorXi::Constant(num_slots, -1);
    table.face_active =
        Eigen::Matrix<bool, Eigen::Dynamic, 1>::Constant(num_slots, false);

    std::vector<const OpticalMaterial*> unique_materials;
    std::unordered_map<const OpticalMaterial*, int> material_row;
    const auto row_of = [&](const std::shared_ptr<OpticalMaterial>& material) {
        if (material == nullptr) {
            return -1;
        }
        const auto [it, inserted] = material_row.emplace(
            material.get(), static_cast<int>(unique_materials.size()));
        if (inserted) {
            unique_materials.push_back(material.get());
        }
        return it->second;
    };

    // Ranges are processed in order; overlapping ranges (a fully-cut-away
    // item leaves a zero-width range at the next item's offset) resolve
    // last-writer-wins, which restores the legitimate assignment.
    for (const auto& range : model_mesh.primitives) {
        const auto node_it =
            _by_id.find(static_cast<std::uint64_t>(range.geometry_id));
        const auto item =
            node_it == _by_id.end()
                ? nullptr
                : std::dynamic_pointer_cast<GeometryItem>(node_it->second);
        if (item == nullptr) {
            SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                               "material_table: mesh range without a "
                               "registered GeometryItem (id {})",
                               static_cast<std::uint64_t>(range.geometry_id));
            continue;
        }

        const ThermalMesh& thermal_mesh = item->thermal_mesh();
        const int side1_row = row_of(thermal_mesh.get_side1_optical());
        const int side2_row = row_of(thermal_mesh.get_side2_optical());
        if (side1_row < 0 || side2_row < 0) {
            SPDLOG_LOGGER_WARN(
                pycanha::get_logger(),
                "material_table: item '{}' has no optical "
                "material on side {} (faces treated as "
                "blackbody)",
                item->name(),
                side1_row < 0 ? (side2_row < 0 ? "1/2" : "1") : "2");
        }

        const auto first = static_cast<Eigen::Index>(range.first_face_id);
        const auto last = std::min(
            static_cast<Eigen::Index>(range.last_face_id), num_slots - 2);
        for (Eigen::Index slot = first; slot <= last; slot += 2) {
            table.face_material[slot] = side1_row;
            table.face_material[slot + 1] = side2_row;
            // The raytracer table is radiative-only: a conductive-only side is
            // inactive here even though it still carries a node.
            table.face_active[slot] = thermal_mesh.is_radiative_active(1U);
            table.face_active[slot + 1] = thermal_mesh.is_radiative_active(2U);
        }
    }

    table.properties.resize(static_cast<Eigen::Index>(unique_materials.size()),
                            6);
    for (std::size_t row = 0; row < unique_materials.size(); ++row) {
        const auto& props = unique_materials[row]->get_th_optical_properties();
        for (int dof = 0; dof < 6; ++dof) {
            table.properties(static_cast<Eigen::Index>(row), dof) =
                static_cast<float>(props.at(static_cast<std::size_t>(dof)));
        }
    }
    return table;
}

}  // namespace pycanha::gmm
