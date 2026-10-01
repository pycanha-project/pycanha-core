#include "pycanha-core/conduction/network_part.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "engine_detail.hpp"
#include "pycanha-core/conduction/links.hpp"
#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/conduction/profile.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/face_pair_geometry.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/gmm/scene/resolve.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/parallel_for.hpp"

namespace pycanha::conduction {

namespace detail {

bool is_benign(const DiagnosticCode code) noexcept {
    switch (code) {
        case DiagnosticCode::DiscreteLinkFallback:
        case DiagnosticCode::AxisSingularity:
        case DiagnosticCode::InactiveSideSkipped:
        case DiagnosticCode::NoNodeNumbers:
        case DiagnosticCode::UnmeshedPrimitive:
            return true;
        default:
            return false;
    }
}

void report_diagnostic(TmmBuildReport& report, DiagnosticCode code,
                       std::string geometry_name, std::string message) {
    report.diagnostics.push_back(
        BuildDiagnostic{.code = code,
                        .geometry_name = std::move(geometry_name),
                        .message = std::move(message)});
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

}  // namespace detail

namespace {

using detail::report_diagnostic;
using gmm::NO_NODE;

// At or below this surviving fraction a face pair is gone; below one minus it
// the face pair is cut.
constexpr double removed_fraction = 1e-9;

// A face pair of this area or less (a collapsed cut interval) carries nothing:
// the mesher drops triangles below the same area.
constexpr double degenerate_area = LENGTH_TOL * LENGTH_TOL;

// How many node numbers a diagnostic quotes.
constexpr std::size_t quoted_nodes = 5;

enum class FacePairState : std::uint8_t { ABSENT, CUT, WHOLE };

// Everything one side of one item contributes.
struct SideProperties {
    /// Takes part in either physics. That is the "this side of the shell
    /// exists" test, so it is what decides whether the side defines nodes and
    /// hands them its thermal mass and its centroid.
    bool active = false;
    /// Takes part in conduction. Only the conductors depend on this: a
    /// radiative-only side still has a node and still carries its mass.
    bool conductive = false;
    bool has_nodes = false;
    double capacitance_per_area = 0.0;
    const gmm::BulkMaterial* bulk = nullptr;
    std::int64_t node_start = NO_NODE;
    std::int64_t node_step = 0;

    [[nodiscard]] bool contributes() const noexcept {
        return active && has_nodes;
    }
    [[nodiscard]] NodeNum node_of(std::size_t face_pair) const noexcept {
        return static_cast<NodeNum>(
            node_start + (static_cast<std::int64_t>(face_pair) * node_step));
    }
};

// Node numbers quoted in a diagnostic, and whether there were more.
class QuotedNodes {
  public:
    void add(NodeNum node) {
        ++_count;
        if (_quoted.size() < quoted_nodes) {
            _quoted.push_back(node);
        }
    }
    [[nodiscard]] std::size_t count() const noexcept { return _count; }
    [[nodiscard]] std::string text() const {
        std::string text;
        for (const NodeNum node : _quoted) {
            text += (text.empty() ? "" : ", ") + std::to_string(node);
        }
        return _count > _quoted.size() ? text + ", ..." : text;
    }

  private:
    std::vector<NodeNum> _quoted;
    std::size_t _count = 0;
};

void describe_side(const gmm::ThermalMesh& thermal_mesh, unsigned side,
                   std::size_t face_pairs, const std::string& item_name,
                   SideProperties& properties, TmmBuildReport& report) {
    properties.active = thermal_mesh.is_side_active(side);
    properties.conductive = thermal_mesh.is_conductive_active(side);
    properties.node_start = side == 1U ? thermal_mesh.get_node1_start()
                                       : thermal_mesh.get_node2_start();
    properties.node_step = side == 1U ? thermal_mesh.get_node1_step()
                                      : thermal_mesh.get_node2_step();
    // Some face pair carries a node number: every one when the numbers step,
    // unless the single face pair is the unassigned one.
    properties.has_nodes =
        properties.node_step == 0
            ? properties.node_start != NO_NODE
            : face_pairs > 1U || properties.node_start != NO_NODE;

    // Only a side that carries node numbers can contribute, so has_nodes gates
    // every diagnostic below: a shell that is simply single-sided must not fill
    // the report with noise.
    if (!properties.has_nodes) {
        return;
    }
    const std::string side_name =
        "'" + item_name + "' side " + std::to_string(side);
    if (!properties.active) {
        report_diagnostic(report, DiagnosticCode::InactiveSideSkipped,
                          item_name,
                          side_name +
                              " carries node numbers but takes part in neither "
                              "physics: no node, no capacitance, no conductor");
        return;
    }
    if (!properties.conductive) {
        report_diagnostic(report, DiagnosticCode::InactiveSideSkipped,
                          item_name,
                          side_name +
                              " is radiative only: it defines nodes and gives "
                              "them its capacitance, but no conductor");
    }

    const auto& material = side == 1U ? thermal_mesh.get_side1_material()
                                      : thermal_mesh.get_side2_material();
    const double thickness = side == 1U ? thermal_mesh.get_side1_thick()
                                        : thermal_mesh.get_side2_thick();
    if (material == nullptr) {
        report_diagnostic(report, DiagnosticCode::MissingBulk, item_name,
                          side_name +
                              " is active but has no bulk material: its nodes "
                              "get no capacitance and no conductors");
        return;
    }
    properties.bulk = material.get();
    if (thickness <= 0.0) {
        report_diagnostic(report, DiagnosticCode::ZeroThickness, item_name,
                          side_name +
                              " has zero thickness: no capacitance and no "
                              "conductors");
        return;
    }
    // Conductivity only ever feeds the conductors, so a side that does not
    // conduct has nothing to say about it.
    if (properties.conductive && material->get_conductivity() <= 0.0) {
        report_diagnostic(report, DiagnosticCode::ZeroConductivity, item_name,
                          side_name +
                              " has zero conductivity: it carries capacitance "
                              "but no conductors");
    }
    properties.capacitance_per_area =
        material->get_density() * material->get_specific_heat() * thickness;
}

// A band reaching the axis of revolution is the one place the plain
// around-the-axis integral does not apply, because the temperature difference
// between two neighbouring angular face pairs vanishes there instead of staying
// constant across the band. Worth recording which form was used.
void report_link_form(const gmm::GeometryItem& item, TmmBuildReport& report) {
    const std::string& item_name = item.name();
    const std::optional<MeridianProfile> profile = profile_of(item.primitive());
    if (!profile.has_value()) {
        report_diagnostic(
            report, DiagnosticCode::DiscreteLinkFallback, item_name,
            "'" + item_name +
                "' has no closed-form conduction profile, so its conductors "
                "come from the discrete shared-edge path");
        return;
    }
    if (item.thermal_mesh().get_dir1_mesh().size() < 3U) {
        // A single angular face pair has no around-the-axis conductor at all,
        // not even a ring closure, so nothing is truncated.
        return;
    }
    if (!profile->on_axis(0.0) && !profile->on_axis(1.0)) {
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

// Every node number a side hands out, NO_NODE excluded.
template <typename Visit>
void for_each_side_node(std::span<const SideProperties* const> sides,
                        std::size_t face_pairs, const Visit& visit) {
    for (const SideProperties* side : sides) {
        for (std::size_t face_pair = 0; face_pair < face_pairs; ++face_pair) {
            const NodeNum node = side->node_of(face_pair);
            if (node != NO_NODE) {
                visit(node);
            }
        }
    }
}

// Local index of every node an item feeds, in increasing number order. A
// flat table over the item's node-number range when it is compact -- the
// usual case, where the numbers step through the face pairs -- the sorted
// numbers and a binary search otherwise.
class NodeIndex {
  public:
    NodeIndex() = default;
    NodeIndex(std::span<const SideProperties* const> sides,
              std::size_t face_pairs) {
        std::int64_t lowest = std::numeric_limits<std::int64_t>::max();
        std::int64_t highest = std::numeric_limits<std::int64_t>::min();
        for (const SideProperties* side : sides) {
            const std::int64_t last =
                side->node_start +
                (static_cast<std::int64_t>(face_pairs - 1U) * side->node_step);
            lowest = std::min({lowest, side->node_start, last});
            highest = std::max({highest, side->node_start, last});
        }
        _base = lowest;
        _dense = highest - lowest + 1 <=
                 (8 * static_cast<std::int64_t>(face_pairs)) + 1024;
        if (_dense) {
            build_table(sides, face_pairs,
                        static_cast<std::size_t>(highest - lowest + 1));
        } else {
            for_each_side_node(sides, face_pairs, [this](NodeNum node) {
                _numbers.push_back(node);
            });
            std::ranges::sort(_numbers);
            const auto duplicates = std::ranges::unique(_numbers);
            _numbers.erase(duplicates.begin(), duplicates.end());
        }
    }

    [[nodiscard]] std::vector<NodeNum>& numbers() noexcept { return _numbers; }

    // Local index of a node this index was built from.
    [[nodiscard]] std::int32_t operator()(NodeNum node) const {
        const std::int32_t local =
            _dense ? _table[offset(node)]
                   : static_cast<std::int32_t>(
                         std::ranges::lower_bound(_numbers, node) -
                         _numbers.begin());
        return _remap.empty() ? local : _remap[static_cast<std::size_t>(local)];
    }

    // Drops the nodes that received nothing; later lookups of the others
    // return their new position.
    void set_remap(std::vector<std::int32_t> remap) {
        _remap = std::move(remap);
    }

  private:
    void build_table(std::span<const SideProperties* const> sides,
                     std::size_t face_pairs, std::size_t span) {
        _table.assign(span, -1);
        for_each_side_node(sides, face_pairs,
                           [this](NodeNum node) { _table[offset(node)] = 0; });
        std::int32_t next = 0;
        for (std::size_t entry = 0; entry < _table.size(); ++entry) {
            if (_table[entry] == 0) {
                _table[entry] = next++;
                _numbers.push_back(static_cast<NodeNum>(
                    _base + static_cast<std::int64_t>(entry)));
            }
        }
    }

    [[nodiscard]] std::size_t offset(NodeNum node) const noexcept {
        return static_cast<std::size_t>(static_cast<std::int64_t>(node) -
                                        _base);
    }

    std::int64_t _base = 0;
    bool _dense = true;
    std::vector<std::int32_t> _table;
    std::vector<NodeNum> _numbers;
    std::vector<std::int32_t> _remap;
};

// Sorts a row by column, keeping the order of equal columns: that is the order
// their contributions are summed in.
void sort_row(std::span<std::int32_t> cols, std::span<double> values) {
    const std::size_t size = cols.size();
    if (size <= 32U) {
        for (std::size_t next = 1; next < size; ++next) {
            const std::int32_t col = cols[next];
            const double value = values[next];
            std::size_t position = next;
            while (position > 0 && cols[position - 1] > col) {
                cols[position] = cols[position - 1];
                values[position] = values[position - 1];
                --position;
            }
            cols[position] = col;
            values[position] = value;
        }
        return;
    }
    std::vector<std::size_t> order(size);
    std::ranges::iota(order, 0U);
    std::ranges::stable_sort(
        order, {}, [&cols](std::size_t entry) { return cols[entry]; });
    const std::vector<std::int32_t> sorted_cols(cols.begin(), cols.end());
    const std::vector<double> sorted_values(values.begin(), values.end());
    for (std::size_t entry = 0; entry < size; ++entry) {
        cols[entry] = sorted_cols[order[entry]];
        values[entry] = sorted_values[order[entry]];
    }
}

// Conductor contributions collected per row (the lower node), then summed per
// (row, col) in the order they arrived. A few fixed slots per row -- enough
// for the upper-triangle neighbours of a grid -- and a list for the rows that
// have more, so the contributions never go through one big list.
class RowCollector {
  public:
    explicit RowCollector(std::size_t rows)
        : _slot_cols(rows * row_slots),
          _slot_values(rows * row_slots),
          _used(rows, 0U) {}

    void add(std::int32_t local_a, std::int32_t local_b, double value) {
        const std::int32_t row = std::min(local_a, local_b);
        const std::int32_t col = std::max(local_a, local_b);
        std::uint8_t& used = _used[static_cast<std::size_t>(row)];
        if (used < row_slots) {
            const std::size_t slot =
                (static_cast<std::size_t>(row) * row_slots) + used;
            _slot_cols[slot] = col;
            _slot_values[slot] = value;
            ++used;
        } else {
            _overflow.push_back({.row = row, .col = col, .value = value});
        }
    }

    // Sorted, summed couplings into the part's three coupling arrays.
    void reduce_into(NetworkPart& part) {
        // Rows are visited in order, so the overflow is taken in row order
        // too, keeping the order of the entries within a row.
        std::ranges::stable_sort(
            _overflow, {}, [](const Overflow& entry) { return entry.row; });
        const std::size_t total = std::accumulate(
            _used.begin(), _used.end(), _overflow.size(),
            [](std::size_t sum, std::uint8_t used) { return sum + used; });
        part.coupling_index_1.reserve(total);
        part.coupling_index_2.reserve(total);
        part.conductance.reserve(total);
        std::size_t next_overflow = 0;
        for (std::size_t row = 0; row < _used.size(); ++row) {
            const auto [cols, values] = row_entries(row, next_overflow);
            sort_row(cols, values);
            for (std::size_t read = 0; read < cols.size(); ++read) {
                if (read > 0 && cols[read] == cols[read - 1U]) {
                    part.conductance.back() += values[read];
                    continue;
                }
                part.coupling_index_1.push_back(static_cast<std::int32_t>(row));
                part.coupling_index_2.push_back(cols[read]);
                part.conductance.push_back(values[read]);
            }
        }
    }

  private:
    static constexpr std::size_t row_slots = 4U;

    struct Overflow {
        std::int32_t row;
        std::int32_t col;
        double value;
    };

    // One row's contributions, in arrival order: its slots, then its share
    // of the overflow, gathered in a scratch buffer only when there is one.
    [[nodiscard]] std::pair<std::span<std::int32_t>, std::span<double>>
    row_entries(std::size_t row, std::size_t& next_overflow) {
        const std::size_t begin = row * row_slots;
        const std::span<std::int32_t> cols =
            std::span(_slot_cols).subspan(begin, _used[row]);
        const std::span<double> values =
            std::span(_slot_values).subspan(begin, _used[row]);
        const auto in_row = [&](std::size_t entry) {
            return entry < _overflow.size() &&
                   std::cmp_equal(_overflow[entry].row, row);
        };
        if (!in_row(next_overflow)) {
            return {cols, values};
        }
        _row_cols.assign(cols.begin(), cols.end());
        _row_values.assign(values.begin(), values.end());
        for (; in_row(next_overflow); ++next_overflow) {
            _row_cols.push_back(_overflow[next_overflow].col);
            _row_values.push_back(_overflow[next_overflow].value);
        }
        return {_row_cols, _row_values};
    }

    std::vector<std::int32_t> _slot_cols;
    std::vector<double> _slot_values;
    std::vector<std::uint8_t> _used;
    std::vector<Overflow> _overflow;
    std::vector<std::int32_t> _row_cols;
    std::vector<double> _row_values;
};

// The build of one item's network part, one pass per step.
class PartBuilder {
  public:
    PartBuilder(const gmm::GeometryItem& item,
                const gmm::CoordinateTransformation& to_root,
                const TmmBuildOptions& options,
                std::span<const double> surviving_fraction,
                std::span<const Vector3D> surviving_centroid)
        : _item(&item),
          _to_root(&to_root),
          _options(&options),
          _fraction(surviving_fraction),
          _centroid(surviving_centroid),
          _geometry(item.primitive(), item.thermal_mesh()) {}

    [[nodiscard]] NetworkPart build() {
        if (!describe()) {
            return std::move(_part);
        }
        index_nodes();
        accumulate_face_pairs();
        drop_empty_nodes();
        place_nodes();
        collect_couplings();
        report_outcome();
        return std::move(_part);
    }

  private:
    // Checks the item and its sides; false when it contributes nothing.
    [[nodiscard]] bool describe();
    void index_nodes();
    void accumulate_face_pairs();
    void accumulate_face_pair(std::size_t i, std::size_t j);
    void drop_empty_nodes();
    void place_nodes();
    void collect_couplings();
    void report_outcome();

    [[nodiscard]] const std::string& name() const { return _item->name(); }
    [[nodiscard]] const gmm::ThermalMesh& mesh() const {
        return _item->thermal_mesh();
    }

    const gmm::GeometryItem* _item;
    const gmm::CoordinateTransformation* _to_root;
    const TmmBuildOptions* _options;
    std::span<const double> _fraction;
    std::span<const Vector3D> _centroid;
    gmm::FacePairGeometryEvaluator _geometry;

    NetworkPart _part;
    std::array<SideProperties, 2> _sides{};
    std::size_t _dir1_face_pairs = 0;
    std::size_t _face_pairs = 0;
    NodeIndex _index;
    std::vector<FacePairState> _states;
    // Through-thickness conductance of each face pair, kept only when both
    // sides conduct and can carry different nodes.
    std::vector<double> _through;
    std::size_t _degenerate = 0;
    QuotedNodes _mixed;
};

bool PartBuilder::describe() {
    TmmBuildReport& report = _part.report;
    if (!_geometry.is_supported() || !mesh().is_valid()) {
        ++report.items_skipped;
        report_diagnostic(report, DiagnosticCode::UnmeshedPrimitive, name(),
                          "'" + name() +
                              "' is cutter-only and produces no faces, nodes "
                              "or conductors");
        return false;
    }
    _dir1_face_pairs = mesh().get_dir1_mesh().size() - 1U;
    _face_pairs = _dir1_face_pairs * (mesh().get_dir2_mesh().size() - 1U);
    if ((!_fraction.empty() && _fraction.size() != _face_pairs) ||
        _centroid.size() != _fraction.size()) {
        throw std::invalid_argument(
            "build_network_part: '" + name() +
            "' needs one surviving fraction and one centroid per face pair");
    }

    for (const unsigned side : {1U, 2U}) {
        describe_side(mesh(), side, _face_pairs, name(), _sides.at(side - 1U),
                      report);
    }
    if (!_sides[0].active && !_sides[1].active) {
        ++report.items_skipped;
        return false;
    }
    if (!_sides[0].contributes() && !_sides[1].contributes()) {
        report_diagnostic(
            report, DiagnosticCode::NoNodeNumbers, name(),
            "'" + name() +
                "' has no node numbers on any active side: it contributes "
                "nothing");
        ++report.items_skipped;
        return false;
    }
    ++report.items_processed;
    if (_options->intra_primitive_conductors &&
        ((_sides[0].has_nodes && _sides[0].conductive) ||
         (_sides[1].has_nodes && _sides[1].conductive))) {
        report_link_form(*_item, report);
    }
    return true;
}

void PartBuilder::index_nodes() {
    std::vector<const SideProperties*> feeding;
    for (const SideProperties& side : _sides) {
        if (side.contributes()) {
            feeding.push_back(&side);
        }
    }
    _index = NodeIndex(feeding, _face_pairs);
    _part.node_numbers = std::move(_index.numbers());
    const std::size_t num_nodes = _part.node_numbers.size();
    for (auto* column :
         {&_part.thermal_capacity, &_part.position_x, &_part.position_y,
          &_part.position_z, &_part.position_weight}) {
        column->assign(num_nodes, 0.0);
    }
    _part.node_sides.assign(num_nodes, 0U);
    _part.side_bulk = {_sides[0].bulk, _sides[1].bulk};

    const bool through_thickness =
        _options->through_thickness_conductors && _sides[0].contributes() &&
        _sides[1].contributes() && _sides[0].conductive && _sides[1].conductive;
    _through.assign(through_thickness ? _face_pairs : 0U, 0.0);
    _states.assign(_face_pairs, FacePairState::WHOLE);
}

// Exact geometry of every face pair, scaled by what survives of it.
void PartBuilder::accumulate_face_pairs() {
    for (std::size_t face_pair = 0; face_pair < _face_pairs; ++face_pair) {
        accumulate_face_pair(face_pair % _dir1_face_pairs,
                             face_pair / _dir1_face_pairs);
    }
}

void PartBuilder::accumulate_face_pair(std::size_t i, std::size_t j) {
    const std::size_t face_pair = i + (j * _dir1_face_pairs);
    const gmm::FacePairGeometry face =
        _geometry(static_cast<MeshIndex>(i), static_cast<MeshIndex>(j));
    const double fraction = _fraction.empty() ? 1.0 : _fraction[face_pair];
    if (face.area <= degenerate_area || fraction <= removed_fraction) {
        ++(face.area <= degenerate_area ? _degenerate
                                        : _part.report.face_pairs_removed);
        _states[face_pair] = FacePairState::ABSENT;
        return;
    }
    Point3D centroid = face.centroid;
    if (fraction < 1.0 - removed_fraction) {
        ++_part.report.face_pairs_cut;
        _states[face_pair] = FacePairState::CUT;
        centroid = _to_root->inverse().apply(_centroid[face_pair]);
    }
    const double weight = face.area * std::min(fraction, 1.0);
    for (std::size_t side = 0; side < _sides.size(); ++side) {
        const SideProperties& properties = _sides.at(side);
        const NodeNum node = properties.node_of(face_pair);
        if (!properties.contributes() || node == NO_NODE) {
            continue;
        }
        const auto local = static_cast<std::size_t>(_index(node));
        _part.thermal_capacity[local] +=
            properties.capacitance_per_area * weight;
        _part.position_weight[local] += weight;
        _part.position_x[local] += weight * centroid.x();
        _part.position_y[local] += weight * centroid.y();
        _part.position_z[local] += weight * centroid.z();
        _part.node_sides[local] |= static_cast<std::uint8_t>(1U << side);
    }
    const NodeNum node_1 = _sides[0].node_of(face_pair);
    const NodeNum node_2 = _sides[1].node_of(face_pair);
    if (!_through.empty() && node_1 != NO_NODE && node_2 != NO_NODE &&
        node_1 != node_2) {
        _through[face_pair] = through_thickness_conductance(mesh(), weight);
    }
}

// Nodes whose face pairs all collapsed or were cut away are not created.
void PartBuilder::drop_empty_nodes() {
    const std::size_t num_nodes = _part.node_numbers.size();
    if (std::ranges::find(_part.node_sides, std::uint8_t{0U}) ==
        _part.node_sides.end()) {
        return;
    }
    std::vector<std::int32_t> remap(num_nodes, -1);
    std::size_t kept = 0;
    for (std::size_t local = 0; local < num_nodes; ++local) {
        if (_part.node_sides[local] == 0U) {
            continue;
        }
        remap[local] = static_cast<std::int32_t>(kept);
        _part.node_numbers[kept] = _part.node_numbers[local];
        _part.node_sides[kept] = _part.node_sides[local];
        for (auto* column :
             {&_part.thermal_capacity, &_part.position_x, &_part.position_y,
              &_part.position_z, &_part.position_weight}) {
            (*column)[kept] = (*column)[local];
        }
        ++kept;
    }
    _part.node_numbers.resize(kept);
    _part.node_sides.resize(kept);
    for (auto* column :
         {&_part.thermal_capacity, &_part.position_x, &_part.position_y,
          &_part.position_z, &_part.position_weight}) {
        column->resize(kept);
    }
    _index.set_remap(std::move(remap));
}

// Positions to the root frame, and the nodes gathering two bulk materials.
void PartBuilder::place_nodes() {
    const bool sides_differ = _part.side_bulk[0] != nullptr &&
                              _part.side_bulk[1] != nullptr &&
                              _part.side_bulk[0] != _part.side_bulk[1];
    for (std::size_t local = 0; local < _part.node_numbers.size(); ++local) {
        const Point3D placed = _to_root->apply(
            Point3D{_part.position_x[local], _part.position_y[local],
                    _part.position_z[local]} /
            _part.position_weight[local]);
        _part.position_x[local] = placed.x();
        _part.position_y[local] = placed.y();
        _part.position_z[local] = placed.z();
        if (sides_differ && _part.node_sides[local] == 3U) {
            _mixed.add(_part.node_numbers[local]);
        }
    }
}

// Every conductor contribution, in the order the old per-pair sums added
// them: the in-plane links as generated, then the through-thickness ones.
void PartBuilder::collect_couplings() {
    RowCollector rows(_part.node_numbers.size());
    if (_options->intra_primitive_conductors) {
        TmmBuildReport& report = _part.report;
        for_each_intra_primitive_link(
            _item->primitive(), mesh(), *_options,
            [&](const FacePairLink& link) {
                ++report.face_pair_links_computed;
                const FacePairState state_a = _states[link.face_pair_a];
                const FacePairState state_b = _states[link.face_pair_b];
                if (state_a == FacePairState::ABSENT ||
                    state_b == FacePairState::ABSENT) {
                    return;
                }
                if (state_a == FacePairState::CUT ||
                    state_b == FacePairState::CUT) {
                    ++report.links_removed;
                    return;
                }
                const SideProperties& side = _sides.at(link.side - 1U);
                const NodeNum node_a = side.node_of(link.face_pair_a);
                const NodeNum node_b = side.node_of(link.face_pair_b);
                if (node_a == NO_NODE || node_b == NO_NODE ||
                    node_a == node_b) {
                    return;  // unassigned, or the same node on both ends
                }
                rows.add(_index(node_a), _index(node_b), link.conductance);
            });
    }
    for (std::size_t face_pair = 0; face_pair < _through.size(); ++face_pair) {
        if (_through[face_pair] > 0.0) {
            rows.add(_index(_sides[0].node_of(face_pair)),
                     _index(_sides[1].node_of(face_pair)), _through[face_pair]);
        }
    }
    rows.reduce_into(_part);
}

void PartBuilder::report_outcome() {
    TmmBuildReport& report = _part.report;
    report.nodes_created = _part.node_numbers.size();
    report.conductors_created = _part.conductance.size();
    if (_degenerate > 0) {
        report_diagnostic(report, DiagnosticCode::DegenerateFacePair, name(),
                          "'" + name() + "' has " +
                              std::to_string(_degenerate) +
                              " face pairs of zero area: they carry nothing");
    }
    if (report.face_pairs_cut + report.face_pairs_removed > 0) {
        report_diagnostic(
            report, DiagnosticCode::CutFacePairs, name(),
            "'" + name() +
                "' is cut: " + std::to_string(report.face_pairs_cut) +
                " face pairs keep part of their area (capacity scaled), " +
                std::to_string(report.face_pairs_removed) +
                " are cut away, and " + std::to_string(report.links_removed) +
                " in-plane links touching cut face pairs are removed");
    }
    if (_mixed.count() > 0) {
        report_diagnostic(report, DiagnosticCode::MixedBulkOnNode, name(),
                          std::to_string(_mixed.count()) + " nodes of '" +
                              name() +
                              "' gather faces with different bulk materials "
                              "(" +
                              _mixed.text() +
                              "); their capacitances are summed as they stand");
    }
}

}  // namespace

NetworkPart build_network_part(const gmm::GeometryItem& item,
                               const gmm::CoordinateTransformation& to_root,
                               const TmmBuildOptions& options,
                               std::span<const double> surviving_fraction,
                               std::span<const Vector3D> surviving_centroid) {
    return PartBuilder(item, to_root, options, surviving_fraction,
                       surviving_centroid)
        .build();
}

// ---------------------------------------------------------------------------
// Commit
// ---------------------------------------------------------------------------

namespace {

// Where each part's local node went in the merged node list, with the merged
// nodes' summed attributes.
struct MergedNodes {
    NetworkPart part;
    std::vector<std::vector<std::int32_t>> to_merged;
};

// The merged node list of parts that share or interleave node numbers:
// capacities summed in part order, positions combined with their weights.
[[nodiscard]] MergedNodes merge_nodes(std::span<const NetworkPart* const> parts,
                                      TmmBuildReport& report) {
    struct Entry {
        NodeNum number;
        std::uint32_t part;
        std::uint32_t local;
    };
    std::vector<Entry> entries;
    MergedNodes merged;
    merged.to_merged.resize(parts.size());
    for (std::size_t part = 0; part < parts.size(); ++part) {
        const auto& numbers = parts[part]->node_numbers;
        merged.to_merged[part].resize(numbers.size());
        for (std::size_t local = 0; local < numbers.size(); ++local) {
            entries.push_back({.number = numbers[local],
                               .part = static_cast<std::uint32_t>(part),
                               .local = static_cast<std::uint32_t>(local)});
        }
    }
    std::ranges::stable_sort(entries, {},
                             [](const Entry& entry) { return entry.number; });

    // The first bulk material each merged node met, and from which part: a
    // different one from another part mixes materials across items (a node
    // mixed within one part was reported with that part).
    NetworkPart& out = merged.part;
    std::vector<std::pair<const gmm::BulkMaterial*, std::uint32_t>> first_bulk;
    std::vector<bool> reported;
    QuotedNodes mixed;
    for (const Entry& entry : entries) {
        if (out.node_numbers.empty() ||
            out.node_numbers.back() != entry.number) {
            out.node_numbers.push_back(entry.number);
            for (auto* column :
                 {&out.thermal_capacity, &out.position_x, &out.position_y,
                  &out.position_z, &out.position_weight}) {
                column->push_back(0.0);
            }
            out.node_sides.push_back(0U);
            first_bulk.emplace_back(nullptr, entry.part);
            reported.push_back(false);
        }
        const std::size_t target = out.node_numbers.size() - 1U;
        merged.to_merged[entry.part][entry.local] =
            static_cast<std::int32_t>(target);
        const NetworkPart& source = *parts[entry.part];
        const double weight = source.position_weight[entry.local];
        out.thermal_capacity[target] += source.thermal_capacity[entry.local];
        out.position_weight[target] += weight;
        out.position_x[target] += weight * source.position_x[entry.local];
        out.position_y[target] += weight * source.position_y[entry.local];
        out.position_z[target] += weight * source.position_z[entry.local];
        out.node_sides[target] |= source.node_sides[entry.local];
        for (std::size_t side = 0; side < source.side_bulk.size(); ++side) {
            const gmm::BulkMaterial* bulk = source.side_bulk.at(side);
            if ((source.node_sides[entry.local] & (1U << side)) == 0U ||
                bulk == nullptr) {
                continue;
            }
            auto& [first, first_part] = first_bulk[target];
            if (first == nullptr) {
                first = bulk;
                first_part = entry.part;
            } else if (first != bulk && first_part != entry.part &&
                       !reported[target]) {
                reported[target] = true;
                mixed.add(entry.number);
            }
        }
    }
    for (std::size_t node = 0; node < out.node_numbers.size(); ++node) {
        if (out.position_weight[node] > 0.0) {
            out.position_x[node] /= out.position_weight[node];
            out.position_y[node] /= out.position_weight[node];
            out.position_z[node] /= out.position_weight[node];
        }
    }
    if (mixed.count() > 0) {
        report_diagnostic(report, DiagnosticCode::MixedBulkOnNode, "",
                          std::to_string(mixed.count()) +
                              " nodes shared between items gather faces with "
                              "different bulk materials (" +
                              mixed.text() +
                              "); their capacitances are summed as they stand");
    }
    return merged;
}

// Parts that share or interleave node numbers, merged into one: capacities
// and conductances summed in part order, positions combined with their
// weights.
[[nodiscard]] NetworkPart merge_parts(std::span<const NetworkPart* const> parts,
                                      TmmBuildReport& report) {
    MergedNodes merged = merge_nodes(parts, report);
    RowCollector rows(merged.part.node_numbers.size());
    for (std::size_t part = 0; part < parts.size(); ++part) {
        const NetworkPart& source = *parts[part];
        const auto& to_merged = merged.to_merged[part];
        for (std::size_t entry = 0; entry < source.conductance.size();
             ++entry) {
            rows.add(to_merged[static_cast<std::size_t>(
                         source.coupling_index_1[entry])],
                     to_merged[static_cast<std::size_t>(
                         source.coupling_index_2[entry])],
                     source.conductance[entry]);
        }
    }
    rows.reduce_into(merged.part);
    return std::move(merged.part);
}

// Drops the conductances at or below the threshold, in place.
void drop_weak_conductances(NetworkPart& part, double threshold) {
    std::size_t kept = 0;
    for (std::size_t entry = 0; entry < part.conductance.size(); ++entry) {
        if (part.conductance[entry] <= threshold) {
            continue;
        }
        part.coupling_index_1[kept] = part.coupling_index_1[entry];
        part.coupling_index_2[kept] = part.coupling_index_2[entry];
        part.conductance[kept] = part.conductance[entry];
        ++kept;
    }
    part.coupling_index_1.resize(kept);
    part.coupling_index_2.resize(kept);
    part.conductance.resize(kept);
}

void add_part_report(TmmBuildReport& report, const TmmBuildReport& part) {
    report.items_processed += part.items_processed;
    report.items_skipped += part.items_skipped;
    report.face_pair_links_computed += part.face_pair_links_computed;
    report.face_pairs_cut += part.face_pairs_cut;
    report.face_pairs_removed += part.face_pairs_removed;
    report.links_removed += part.links_removed;
    report.diagnostics.insert(report.diagnostics.end(),
                              part.diagnostics.begin(), part.diagnostics.end());
}

// The parts with nodes, by their first node number; empty when their numbers
// interleave, so that they cannot simply be appended one after the other.
[[nodiscard]] std::vector<const NetworkPart*> append_order(
    std::span<const NetworkPart* const> parts) {
    std::vector<const NetworkPart*> order;
    std::ranges::copy_if(
        parts, std::back_inserter(order),
        [](const NetworkPart* part) { return !part->node_numbers.empty(); });
    std::ranges::stable_sort(order, {}, [](const NetworkPart* part) {
        return part->node_numbers.front();
    });
    for (std::size_t entry = 1; entry < order.size(); ++entry) {
        if (order[entry - 1U]->node_numbers.back() >=
            order[entry]->node_numbers.front()) {
            return {};
        }
    }
    return order;
}

// Every node is an append at the end of the diffusive block, every part's
// couplings one chunk of a single append.
void write_parts(ThermalMathematicalModel& tmm,
                 std::span<const NetworkPart* const> order,
                 const TmmBuildOptions& options, TmmBuildReport& report) {
    Nodes& nodes = tmm.nodes();
    const std::size_t total_nodes =
        std::accumulate(order.begin(), order.end(), std::size_t{0},
                        [](std::size_t sum, const NetworkPart* part) {
                            return sum + part->node_numbers.size();
                        });
    nodes.reserve(to_idx(total_nodes));
    std::vector<CouplingChunk> chunks;
    std::int32_t offset = 0;
    for (const NetworkPart* part : order) {
        report.nodes_created +=
            nodes
                .add_nodes(NodeBatch{.type = 'D',
                                     .numbers = part->node_numbers,
                                     .capacity = part->thermal_capacity,
                                     .fx = part->position_x,
                                     .fy = part->position_y,
                                     .fz = part->position_z})
                .accepted;
        chunks.push_back(CouplingChunk{.idx_1 = part->coupling_index_1,
                                       .idx_2 = part->coupling_index_2,
                                       .values = part->conductance,
                                       .offset = offset});
        offset += static_cast<std::int32_t>(part->node_numbers.size());
    }
    // The tmm was empty, so every temperature is one of the builder's.
    std::ranges::fill(nodes.T_vector, options.initial_temperature);
    report.conductors_created =
        tmm.conductive_couplings().append_couplings(chunks).accepted;
}

// Nodes a cut left without any conductor.
void report_uncoupled_nodes(std::span<const NetworkPart* const> order,
                            TmmBuildReport& report) {
    QuotedNodes isolated;
    for (const NetworkPart* part : order) {
        if (order.size() > 1U && part->report.links_removed == 0) {
            continue;
        }
        std::vector<bool> coupled(part->node_numbers.size(), false);
        for (std::size_t entry = 0; entry < part->conductance.size(); ++entry) {
            coupled[static_cast<std::size_t>(part->coupling_index_1[entry])] =
                true;
            coupled[static_cast<std::size_t>(part->coupling_index_2[entry])] =
                true;
        }
        for (std::size_t local = 0; local < coupled.size(); ++local) {
            if (!coupled[local]) {
                isolated.add(part->node_numbers[local]);
            }
        }
    }
    if (isolated.count() > 0) {
        report_diagnostic(
            report, DiagnosticCode::UncoupledNodes, "",
            std::to_string(isolated.count()) +
                " nodes of cut items have no conductive coupling left (" +
                isolated.text() +
                "); a steady-state solve is singular unless something else "
                "attaches them");
    }
}

// Every diagnostic, then one summary line at the severity of the worst: a
// clean build stays off the console, one that dropped or assumed something
// announces itself there and points at the detail already recorded above.
void log_build(const TmmBuildReport& report) {
    for (const BuildDiagnostic& diagnostic : report.diagnostics) {
        detail::log_diagnostic(diagnostic);
    }
    const auto dropped = static_cast<std::size_t>(std::ranges::count_if(
        report.diagnostics, [](const BuildDiagnostic& diagnostic) {
            return !detail::is_benign(diagnostic.code);
        }));
    if (dropped == 0U) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Built tmm from gmm - {} items ({} skipped), {} "
                           "nodes, {} conductors; no issues",
                           report.items_processed, report.items_skipped,
                           report.nodes_created, report.conductors_created);
        return;
    }
    SPDLOG_LOGGER_WARN(
        pycanha::get_logger(),
        "Built tmm from gmm - {} items ({} skipped), {} nodes, {} "
        "conductors; {} of {} diagnostics report something dropped or "
        "assumed",
        report.items_processed, report.items_skipped, report.nodes_created,
        report.conductors_created, dropped, report.diagnostics.size());
}

}  // namespace

TmmBuildReport commit_network_parts(ThermalMathematicalModel& tmm,
                                    std::span<const NetworkPart> parts,
                                    const TmmBuildOptions& options) {
    std::vector<const NetworkPart*> pointers(parts.size());
    std::ranges::transform(parts, pointers.begin(),
                           [](const NetworkPart& part) { return &part; });
    return commit_network_parts(tmm, pointers, options);
}

TmmBuildReport commit_network_parts(ThermalMathematicalModel& tmm,
                                    std::span<const NetworkPart* const> parts,
                                    const TmmBuildOptions& options) {
    detail::require_empty_tmm(tmm);

    TmmBuildReport report;
    for (const NetworkPart* part : parts) {
        add_part_report(report, part->report);
    }

    // Parts whose numbers do not interleave are written as they are; the
    // others are merged into one first.
    std::vector<const NetworkPart*> order = append_order(parts);
    NetworkPart merged;
    if (order.empty() || options.min_conductance > 0.0) {
        merged = merge_parts(parts, report);
        drop_weak_conductances(merged, options.min_conductance);
        order.assign(1U, &merged);
    }
    write_parts(tmm, order, options, report);
    if (report.links_removed > 0) {
        report_uncoupled_nodes(order, report);
    }
    log_build(report);
    return report;
}

// ---------------------------------------------------------------------------
// Node areas
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] bool has_faces(const gmm::GeometryItem& item) {
    return gmm::FacePairGeometryEvaluator(item.primitive(), item.thermal_mesh())
        .is_supported();
}

// The triangulated area of each face pair of one target's mesh, handed to the
// node of each active side.
void collect_face_areas(
    const gmm::TriMeshD& mesh, const gmm::ThermalMesh& thermal_mesh,
    std::vector<std::pair<NodeNum, double>>& contributions) {
    std::vector<double> face_area(mesh.nf(), 0.0);
    for (Eigen::Index triangle = 0; triangle < mesh.triangles.rows();
         ++triangle) {
        const auto corners = mesh.triangles.row(triangle);
        const Point3D vertex_0 = mesh.vertices.row(to_idx(corners(0)));
        const Point3D vertex_1 = mesh.vertices.row(to_idx(corners(1)));
        const Point3D vertex_2 = mesh.vertices.row(to_idx(corners(2)));
        face_area[to_sizet(to_idx(mesh.face_ids(triangle)))] +=
            0.5 * (vertex_1 - vertex_0).cross(vertex_2 - vertex_0).norm();
    }
    for (std::size_t face = 0; face + 1U < face_area.size(); face += 2U) {
        // The triangles carry the side-1 face id; both sides share them.
        const double area = face_area[face] + face_area[face + 1U];
        for (const unsigned side : {1U, 2U}) {
            const NodeNum node = mesh.node_numbers(to_idx(face + side - 1U));
            if (area > 0.0 && node != NO_NODE &&
                thermal_mesh.is_side_active(side)) {
                contributions.emplace_back(node, area);
            }
        }
    }
}

}  // namespace

BulkReport assign_node_areas(ThermalModel& model) {
    const gmm::GeometryGroup& root = *model.gmm().root_group();
    const auto targets = gmm::detail::collect_targets(root, root.transform());

    // Cut targets are meshed in parallel; an uncut one reads its item's cached
    // triangulation, which filling mutates, so those stay here.
    std::vector<gmm::TriMeshD> cut_meshes(targets.size());
    std::vector<std::size_t> cut_targets;
    for (std::size_t target = 0; target < targets.size(); ++target) {
        const gmm::GeometryItem& item = targets[target].item;
        if (!has_faces(item)) {
            continue;
        }
        if (targets[target].cutters.empty()) {
            static_cast<void>(item.mesh());
        } else {
            cut_targets.push_back(target);
        }
    }
    const unsigned workers =
        cut_targets.size() > 1U
            ? std::min<unsigned>(std::thread::hardware_concurrency(),
                                 static_cast<unsigned>(cut_targets.size()))
            : 1U;
    utils::parallel_for_index(
        cut_targets.size(), workers, [&](std::size_t entry) {
            cut_meshes[cut_targets[entry]] =
                gmm::detail::mesh_target(targets[cut_targets[entry]]);
        });

    std::vector<std::pair<NodeNum, double>> contributions;
    for (std::size_t target = 0; target < targets.size(); ++target) {
        const gmm::GeometryItem& item = targets[target].item;
        if (has_faces(item)) {
            collect_face_areas(targets[target].cutters.empty()
                                   ? item.mesh()
                                   : cut_meshes[target],
                               item.thermal_mesh(), contributions);
        }
    }

    // Summed per node, in target order.
    std::ranges::stable_sort(contributions, {},
                             [](const auto& entry) { return entry.first; });
    std::vector<NodeNum> node_nums;
    std::vector<double> areas;
    for (const auto& [node, area] : contributions) {
        if (!node_nums.empty() && node_nums.back() == node) {
            areas.back() += area;
        } else {
            node_nums.push_back(node);
            areas.push_back(area);
        }
    }
    return model.tmm().nodes().set_values(NodeAttribute::A, node_nums, areas);
}

}  // namespace pycanha::conduction
