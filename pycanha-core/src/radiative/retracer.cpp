#include "pycanha-core/radiative/retracer.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/parameters/table.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "pycanha-core/solvers/retracer.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "retracer_network.hpp"

namespace pycanha::radiative {

namespace {

// The temperature of a face without a node [K].
constexpr double no_node_temperature = 273.15;
// The sum of the IR properties may exceed 1 by this much, as the scene
// allows for its single-precision rows.
constexpr double property_sum_slack = 1e-9;
// A face row for a face without a material; never read by the kernels,
// valid for the scene's checks.
constexpr std::array<float, 6> black_row{1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F};

// The stored pairs of an upper-triangular matrix by internal node index, as
// node numbers, with their values.
struct NodePairs {
    std::vector<NodeNum> first;
    std::vector<NodeNum> second;
    std::vector<double> values;
};

NodePairs node_pairs(const SparseMatrix& matrix, const Nodes& nodes) {
    const std::vector<NodeNum> numbers = nodes.node_numbers();
    NodePairs pairs;
    const auto stored = static_cast<std::size_t>(matrix.nonZeros());
    pairs.first.reserve(stored);
    pairs.second.reserve(stored);
    pairs.values.reserve(stored);
    for (Index row = 0; row < matrix.outerSize(); ++row) {
        for (SparseMatrix::InnerIterator it(matrix, row); it; ++it) {
            pairs.first.push_back(numbers[static_cast<std::size_t>(row)]);
            pairs.second.push_back(numbers[static_cast<std::size_t>(it.col())]);
            pairs.values.push_back(it.value());
        }
    }
    return pairs;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
}

}  // namespace

RadiativeRetracer::RowState RadiativeRetracer::row_state(PatternState state) {
    switch (state) {
        case PatternState::LOW:
            return RowState::LOW;
        case PatternState::MID:
            return RowState::MID;
        case PatternState::HIGH:
            break;
    }
    return RowState::HIGH;
}

RadiativeRetracer::Geometry RadiativeRetracer::read_geometry(
    const gmm::GeometryModel& gmm) {
    const MaterialTable table = gmm.material_table();
    const auto& node_numbers = gmm.mesh().node_numbers;
    const auto num_faces = static_cast<std::size_t>(table.num_faces());
    if (num_faces == 0) {
        throw std::invalid_argument(
            "RadiativeRetracer: the geometry model has no faces");
    }

    Geometry geometry;
    geometry.face_nodes.assign(node_numbers.begin(), node_numbers.end());
    geometry.face_nodes.resize(num_faces, gmm::NO_NODE);
    geometry.face_materials.assign(table.face_material.begin(),
                                   table.face_material.end());
    geometry.face_active.assign(table.face_active.begin(),
                                table.face_active.end());
    geometry.materials.resize(static_cast<std::size_t>(table.num_materials()));
    for (std::size_t row = 0; row < geometry.materials.size(); ++row) {
        Material& material = geometry.materials[row];
        for (std::size_t dof = 0; dof < material.properties.size(); ++dof) {
            material.properties.at(dof) = table.properties(
                static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(dof));
        }
        if (row < table.emissivity_ir_tables.size()) {
            material.emissivity_table = table.emissivity_ir_tables[row];
        }
    }

    // One row per face, so that every face can take its own temperature.
    MaterialTable& rows = geometry.rows;
    rows.properties.resize(table.num_faces(), 6);
    rows.face_material = Eigen::VectorXi::Constant(table.num_faces(), -1);
    rows.face_active = table.face_active;
    for (std::size_t face = 0; face < num_faces; ++face) {
        const auto row = static_cast<Eigen::Index>(face);
        const int material = geometry.face_materials[face];
        const std::array<float, 6>& values =
            material < 0
                ? black_row
                : geometry.materials[static_cast<std::size_t>(material)]
                      .properties;
        if (material >= 0) {
            rows.face_material(row) = static_cast<int>(face);
        }
        for (std::size_t dof = 0; dof < values.size(); ++dof) {
            rows.properties(row, static_cast<Eigen::Index>(dof)) =
                values.at(dof);
        }
    }
    return geometry;
}

RadiativeRetracer::RadiativeRetracer(
    Device& device, const gmm::GeometryModel& gmm,
    std::shared_ptr<ThermalMathematicalModel> tmm, NodeNum space_node,
    TraceSettings trace, AccumConfig accum,
    std::set<PatternState> pattern_states, RetraceThreshold retrace_dT,
    bool write_back)
    : _tmm(std::move(tmm)),
      _space_node(space_node),
      _trace(trace),
      _accum(accum),
      _pattern_states(std::move(pattern_states)),
      _retrace_dT(std::move(retrace_dT)),
      _write_back(write_back),
      _geometry(read_geometry(gmm)),
      _scene(device, gmm.mesh_parts(), _geometry.rows),
      _accumulator(_scene, Band::IR, _accum) {
    if (_tmm == nullptr) {
        throw std::invalid_argument(
            "RadiativeRetracer: the thermal model is null");
    }
    const auto negative = [](double threshold) { return !(threshold >= 0.0); };
    const bool refused = std::visit(
        [&negative](const auto& thresholds) {
            if constexpr (std::is_same_v<std::decay_t<decltype(thresholds)>,
                                         double>) {
                return negative(thresholds);
            } else {
                return std::ranges::any_of(thresholds,
                                           [&negative](const auto& entry) {
                                               return negative(entry.second);
                                           });
            }
        },
        _retrace_dT);
    if (refused) {
        throw std::invalid_argument(
            "RadiativeRetracer: retrace_dT must not be negative");
    }
}

RadiativeRetracer::~RadiativeRetracer() = default;

void RadiativeRetracer::initialize() {
    const auto start = std::chrono::steady_clock::now();
    _initialized = false;
    _statistics = RetracerStatistics{};
    _overflow = SparseMatrix();
    _load.clear();
    check_model();
    resolve_tables();
    build_checks();

    const bool tabled = std::ranges::any_of(
        _geometry.materials,
        [](const Material& material) { return material.view.has_value(); });
    std::vector<SparseMatrix> traces;
    if (tabled) {
        // Without a table every state has the rows of the current one.
        for (const PatternState state : _pattern_states) {
            fill_rows(row_state(state));
            traces.push_back(trace_nodes());
        }
    }
    fill_rows(RowState::CURRENT);
    traces.push_back(trace_nodes());
    _statistics.initialize_traces = static_cast<int>(traces.size());
    _statistics.traces = _statistics.initialize_traces;

    std::vector<const SparseMatrix*> patterns(traces.size());
    std::ranges::transform(traces, patterns.begin(),
                           [](const SparseMatrix& trace) { return &trace; });
    const auto num_nodes = static_cast<Index>(_tmm->nodes().T_vector.size());
    // The pattern holds zeros: NEW inserts the pairs the model does not hold
    // yet and keeps the others, which write_couplings() overwrites below.
    const NodePairs pattern =
        node_pairs(detail::pattern_union(patterns, num_nodes), _tmm->nodes());
    static_cast<void>(_tmm->radiative_couplings().add_couplings(
        pattern.first, pattern.second, pattern.values, CouplingMerge::NEW));

    ThermalNetwork& network = _tmm->network();
    network.synchronize_structure();
    CouplingMatrices& matrices = network.radiative_matrices();
    matrices.compress();
    write_couplings(traces.back());
    store_trace_temperatures();

    _structure = _tmm->structure_versions();
    _parameters_structure = _tmm->parameters().get_structure_version();
    _statistics.pattern_couplings = matrices.get_num_total_couplings();
    _initialized = true;
    SPDLOG_LOGGER_INFO(get_logger(),
                       "RadiativeRetracer initialized: {} traces, {} radiative "
                       "couplings, {} checked nodes, {:.3g} s",
                       _statistics.initialize_traces,
                       _statistics.pattern_couplings, _checked.size(),
                       seconds_since(start));
}

void RadiativeRetracer::check_model() {
    ThermalNetwork& network = _tmm->network();
    network.synchronize_structure();
    const Nodes& nodes = _tmm->nodes();
    const auto space_index = nodes.get_idx_from_node_num(_space_node);
    if (!space_index.has_value()) {
        throw std::invalid_argument(std::format(
            "RadiativeRetracer: the space node {} is not in the thermal model",
            _space_node));
    }
    _space_index = *space_index;

    const std::size_t num_faces = _geometry.face_nodes.size();
    _face_index.assign(num_faces, -1);
    std::int64_t without_node = 0;
    for (std::size_t face = 0; face < num_faces; ++face) {
        const NodeNum node = _geometry.face_nodes[face];
        if (node == gmm::NO_NODE) {
            if (_geometry.face_active[face]) {
                ++without_node;
            }
            continue;
        }
        const auto index = nodes.get_idx_from_node_num(node);
        if (!index.has_value()) {
            throw std::invalid_argument(std::format(
                "RadiativeRetracer: node {} of face {} is not in the thermal "
                "model",
                node, face));
        }
        _face_index[face] = *index;
    }
    _statistics.faces_without_node = without_node;
    if (without_node > 0) {
        SPDLOG_LOGGER_WARN(
            get_logger(),
            "RadiativeRetracer: {} active faces have no node; their optical "
            "properties are taken at {} K and their couplings dropped",
            without_node, no_node_temperature);
    }
}

void RadiativeRetracer::resolve_tables() {
    const Parameters& parameters = _tmm->parameters();
    for (Material& material : _geometry.materials) {
        material.view.reset();
        if (material.emissivity_table.empty()) {
            continue;
        }
        const Table* table = parameters.find_table(material.emissivity_table);
        if (table == nullptr) {
            throw std::invalid_argument(std::format(
                "RadiativeRetracer: the emissivity table '{}' is not in the "
                "parameters",
                material.emissivity_table));
        }
        const LookupTable1D& lookup = table->lookup_table();
        if (lookup.extrapolation_method() == ExtrapolationMethod::Linear) {
            throw std::invalid_argument(std::format(
                "RadiativeRetracer: the emissivity table '{}' extrapolates "
                "linearly, which can leave [0, 1]; use Constant or Throw",
                material.emissivity_table));
        }
        // Between breakpoints a table stays within its breakpoint values,
        // and outside them it holds the end values or throws, so checking
        // the breakpoints checks every temperature.
        const double others = static_cast<double>(material.properties[1]) +
                              static_cast<double>(material.properties[2]);
        const auto& x = lookup.x();
        const auto& y = lookup.y();
        Eigen::Index low = 0;
        Eigen::Index high = 0;
        for (Eigen::Index point = 0; point < y.size(); ++point) {
            const double emissivity = y(point);
            const bool in_range = emissivity >= 0.0 && emissivity <= 1.0;
            if (!in_range || emissivity + others > 1.0 + property_sum_slack) {
                throw std::invalid_argument(std::format(
                    "RadiativeRetracer: the emissivity table '{}' gives {} at "
                    "{} K: the emissivity must be in [0, 1] and, with the IR "
                    "specular reflectance and transmissivity ({}), add up to "
                    "at most 1",
                    material.emissivity_table, emissivity, x(point), others));
            }
            low = emissivity < y(low) ? point : low;
            high = emissivity > y(high) ? point : high;
        }
        material.low_temperature = x(low);
        material.high_temperature = x(high);
        material.view = table->view(parameters.time_ptr());
    }
}

void RadiativeRetracer::build_checks() {
    const Nodes& nodes = _tmm->nodes();
    _checked.clear();
    _thresholds.clear();
    if (const auto* uniform = std::get_if<double>(&_retrace_dT)) {
        // Every node owning an active face; a per-node map narrows it.
        std::set<Index> owners;
        for (std::size_t face = 0; face < _face_index.size(); ++face) {
            if (_face_index[face] >= 0 && _geometry.face_active[face]) {
                owners.insert(_face_index[face]);
            }
        }
        _checked.assign(owners.begin(), owners.end());
        _thresholds.assign(_checked.size(), *uniform);
        return;
    }
    for (const auto& [node, threshold] :
         std::get<std::map<NodeNum, double>>(_retrace_dT)) {
        const auto index = nodes.get_idx_from_node_num(node);
        if (!index.has_value()) {
            throw std::invalid_argument(std::format(
                "RadiativeRetracer: retrace_dT names node {}, which is not in "
                "the thermal model",
                node));
        }
        _checked.push_back(*index);
        _thresholds.push_back(threshold);
    }
}

double RadiativeRetracer::face_temperature(std::size_t face) const {
    const Index index = _face_index[face];
    return index < 0 ? no_node_temperature
                     : _tmm->nodes().T_vector[static_cast<std::size_t>(index)];
}

void RadiativeRetracer::fill_rows(RowState state) {
    for (std::size_t face = 0; face < _face_index.size(); ++face) {
        const int material_row = _geometry.face_materials[face];
        if (material_row < 0) {
            continue;
        }
        const Material& material =
            _geometry.materials[static_cast<std::size_t>(material_row)];
        if (!material.view.has_value()) {
            continue;
        }
        const pycanha::detail::TableView& view = *material.view;
        double emissivity = 0.0;
        switch (state) {
            case RowState::LOW:
                emissivity = pycanha::detail::table_value(
                    view, material.low_temperature);
                break;
            case RowState::HIGH:
                emissivity = pycanha::detail::table_value(
                    view, material.high_temperature);
                break;
            case RowState::MID:
                emissivity = 0.5 * (pycanha::detail::table_value(
                                        view, material.low_temperature) +
                                    pycanha::detail::table_value(
                                        view, material.high_temperature));
                break;
            case RowState::CURRENT:
                emissivity =
                    pycanha::detail::table_value(view, face_temperature(face));
                break;
        }
        _geometry.rows.properties(static_cast<Eigen::Index>(face), 0) =
            static_cast<float>(emissivity);
    }
}

SparseMatrix RadiativeRetracer::trace_nodes() {
    _scene.update_materials(_geometry.rows);
    _accumulator.reset();
    _scene.accumulate_exchange(_accumulator, _trace);
    const ExchangeResult result = _accumulator.result();
    detail::NodeCouplings nodes = detail::reduce_to_nodes(
        result.factors, _face_index, _space_index,
        static_cast<Index>(_tmm->nodes().T_vector.size()));
    _statistics.space_total = nodes.space_total;
    _statistics.inactive_total = nodes.inactive_total;
    _statistics.lost_total = nodes.lost_total;
    _statistics.intra_node_total = nodes.intra_node_total;
    _statistics.last_trace = result.stats;
    return std::move(nodes.couplings);
}

void RadiativeRetracer::write_couplings(const SparseMatrix& couplings) {
    detail::SplitResult split = detail::split_into_slots(
        couplings, _tmm->network().radiative_matrices());
    _overflow = std::move(split.overflow);
    _statistics.overflow_couplings = _overflow.nonZeros();
    _statistics.overflow_share = split.trace_total > 0.0
                                     ? split.overflow_total / split.trace_total
                                     : 0.0;
    if (&couplings != &_last_couplings) {
        _last_couplings = couplings;
    }
    refresh_load();
}

void RadiativeRetracer::retrace() {
    const auto start = std::chrono::steady_clock::now();
    fill_rows(RowState::CURRENT);
    write_couplings(trace_nodes());
    store_trace_temperatures();
    ++_solve_traces;
    ++_statistics.traces;
    _statistics.last_trace_time = seconds_since(start);
    _statistics.trace_time += _statistics.last_trace_time;
    SPDLOG_LOGGER_DEBUG(get_logger(),
                        "RadiativeRetracer: trace {}, {} overflow couplings, "
                        "{:.3g} s",
                        _statistics.traces, _statistics.overflow_couplings,
                        _statistics.last_trace_time);
}

bool RadiativeRetracer::retrace_due() const {
    const auto& temperatures = _tmm->nodes().T_vector;
    for (std::size_t check = 0; check < _checked.size(); ++check) {
        const double change =
            temperatures[static_cast<std::size_t>(_checked[check])] -
            _trace_temperatures[check];
        if (std::abs(change) > _thresholds[check]) {
            return true;
        }
    }
    return false;
}

void RadiativeRetracer::store_trace_temperatures() {
    const auto& temperatures = _tmm->nodes().T_vector;
    _trace_temperatures.resize(_checked.size());
    for (std::size_t check = 0; check < _checked.size(); ++check) {
        _trace_temperatures[check] =
            temperatures[static_cast<std::size_t>(_checked[check])];
    }
}

void RadiativeRetracer::refresh_load() {
    if (_overflow.nonZeros() == 0) {
        _load.clear();
        return;
    }
    detail::overflow_load(_overflow, _tmm->nodes().T_vector, _load);
}

void RadiativeRetracer::begin_solve() {
    if (!_initialized) {
        throw std::runtime_error(
            "RadiativeRetracer: call initialize() before the solve");
    }
    if (_tmm->structure_versions() != _structure ||
        _tmm->parameters().get_structure_version() != _parameters_structure) {
        throw std::runtime_error(
            "RadiativeRetracer: the thermal model's nodes, couplings or "
            "tables changed since initialize(); call initialize() again");
    }
    _solve_traces = 0;
    if (retrace_due()) {
        retrace();
    } else {
        // Formulas or the user may have changed the couplings since.
        write_couplings(_last_couplings);
    }
}

void RadiativeRetracer::update() {
    if (retrace_due()) {
        retrace();
    } else {
        refresh_load();
    }
}

bool RadiativeRetracer::accepts_convergence(double tolerance) const {
    const auto& temperatures = _tmm->nodes().T_vector;
    for (std::size_t check = 0; check < _checked.size(); ++check) {
        const double change =
            temperatures[static_cast<std::size_t>(_checked[check])] -
            _trace_temperatures[check];
        if (std::abs(change) > std::max(_thresholds[check], tolerance)) {
            return false;
        }
    }
    return true;
}

std::span<const double> RadiativeRetracer::internal_load() const {
    return _load;
}

solvers::RetracerCounters RadiativeRetracer::counters() const {
    return {.retraces = _solve_traces,
            .overflow_couplings = _statistics.overflow_couplings,
            .overflow_share = _statistics.overflow_share};
}

bool RadiativeRetracer::end_solve() {
    if (!_write_back || _overflow.nonZeros() == 0) {
        return false;
    }
    const NodePairs written = node_pairs(_overflow, _tmm->nodes());
    static_cast<void>(_tmm->radiative_couplings().add_couplings(
        written.first, written.second, written.values, CouplingMerge::NEW));
    _overflow = SparseMatrix();
    _load.clear();
    _initialized = false;
    return true;
}

}  // namespace pycanha::radiative
