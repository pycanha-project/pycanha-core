#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/table.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "pycanha-core/solvers/retracer.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha {
class ThermalMathematicalModel;
}  // namespace pycanha

namespace pycanha::gmm {
class GeometryModel;
}  // namespace pycanha::gmm

namespace pycanha::radiative {

/// Optical states traced by RadiativeRetracer::initialize() to find which
/// node pairs need a radiative coupling. Each material whose IR emissivity
/// is a table takes, in every face:
enum class PatternState : std::uint8_t {
    LOW,   ///< its lowest emissivity (the smallest value of the table),
    MID,   ///< the mean of its lowest and highest emissivity,
    HIGH,  ///< its highest emissivity.
};

/// What a RadiativeRetracer has done since its last initialize().
struct RetracerStatistics {
    /// Traces made by initialize() (one per pattern state, plus the one at
    /// the current temperatures).
    int initialize_traces = 0;
    /// Every trace since initialize(), those of initialize() included.
    int traces = 0;
    /// Radiative couplings in the model after initialize().
    std::int64_t pattern_couplings = 0;
    /// Active faces without a node, whose optical properties are taken at
    /// 273.15 K and whose couplings are dropped.
    std::int64_t faces_without_node = 0;
    /// Of the last trace: couplings with no slot in the model and their
    /// share of the sum of every coupling [m^2/m^2].
    std::int64_t overflow_couplings = 0;
    double overflow_share = 0.0;
    /// Of the last trace, over the faces with a node [m^2]: the couplings to
    /// space, those absorbed by inactive faces and the Monte-Carlo residue,
    /// the last two dropped from the couplings, and the couplings between
    /// two faces of the same node.
    double space_total = 0.0;
    double inactive_total = 0.0;
    double lost_total = 0.0;
    double intra_node_total = 0.0;
    /// Wall time of the last retrace and of every retrace since
    /// initialize() [s], the traces of initialize() not included: the
    /// property rows, the trace on the device, the read back, the reduction
    /// to nodes and the write into the model.
    double last_trace_time = 0.0;
    double trace_time = 0.0;
    /// The ray statistics of the last trace.
    TraceStats last_trace;
};

/// IR radiative couplings that follow temperature-dependent emissivities.
/**
 * The retracer owns a scene built from the geometry model at construction,
 * with one optical property row per face, and every radiative coupling of
 * the thermal model. Its configuration (trace and accumulator settings, the
 * seed among them, pattern states, retrace thresholds, write-back) is fixed
 * at construction. A face takes the optical properties of its material,
 * with the IR emissivity read from the material's table (see
 * gmm::OpticalMaterial::emissivity_ir_table) at the temperature of the
 * face's node; a face without a node is taken at 273.15 K.
 *
 * initialize() traces at the selected pattern states and at the current
 * temperatures, inserts into the thermal model the node pairs of the union
 * of those traces (with the couplings to the space node) and writes every
 * radiative coupling of the model: the value of the trace at the current
 * temperatures, or 0 where it has none. A radiative coupling the user needs
 * besides the traced ones must therefore be a formula or a callback, which
 * run after the retracer. Call it before the solver's initialize(), and
 * again after a solve that wrote couplings back.
 *
 * During a solve (Solver::set_retracer) it traces again before a callback
 * when a checked node's temperature moved more than its threshold since the
 * last trace, and accepts the solver's convergence only when every checked
 * node is within max(threshold, tolerance) of the temperatures of the trace
 * the last linear solve used. A traced coupling without a slot in the
 * model's radiative matrices (the overflow) is carried as an internal load,
 * exact at convergence; with write-back, end_solve() inserts the overflow
 * of the last trace into the model, which then needs initialize() again.
 *
 * Only the IR band is traced: solar loads and solar optical properties are
 * not touched. Inactive faces and the Monte-Carlo residue are dropped from
 * the couplings and totalled in statistics().
 */
class RadiativeRetracer final : public solvers::Retracer {
  public:
    /// One threshold [K] for every node owning an active face, or a
    /// threshold per node number, checked on the listed nodes only.
    using RetraceThreshold = std::variant<double, std::map<NodeNum, double>>;

    /// `device` must outlive the retracer. The geometry model is read here
    /// only: later changes to it are not seen. Throws std::invalid_argument
    /// for a negative threshold, an empty geometry or a geometry the scene
    /// refuses.
    RadiativeRetracer(
        Device& device, const gmm::GeometryModel& gmm,
        std::shared_ptr<ThermalMathematicalModel> tmm, NodeNum space_node,
        TraceSettings trace = {}, AccumConfig accum = {},
        std::set<PatternState> pattern_states = {PatternState::LOW,
                                                 PatternState::MID,
                                                 PatternState::HIGH},
        RetraceThreshold retrace_dT = 0.0, bool write_back = true);
    ~RadiativeRetracer() override;

    RadiativeRetracer(const RadiativeRetracer&) = delete;
    RadiativeRetracer& operator=(const RadiativeRetracer&) = delete;
    RadiativeRetracer(RadiativeRetracer&&) = delete;
    RadiativeRetracer& operator=(RadiativeRetracer&&) = delete;

    /// Checks the model and the tables, traces at the pattern states and at
    /// the current temperatures, inserts the pattern and writes every
    /// radiative coupling. Throws std::invalid_argument when the space node
    /// or a node of the geometry is not in the thermal model, a table is
    /// missing or extrapolates linearly, an emissivity leaves [0, 1] or the
    /// IR properties of a material add up to more than 1 at a breakpoint of
    /// its table, or a threshold names a node that is not in the model.
    void initialize();
    [[nodiscard]] bool initialized() const noexcept { return _initialized; }

    [[nodiscard]] NodeNum space_node() const noexcept { return _space_node; }
    [[nodiscard]] const TraceSettings& trace_settings() const noexcept {
        return _trace;
    }
    [[nodiscard]] const AccumConfig& accum_config() const noexcept {
        return _accum;
    }
    [[nodiscard]] const std::set<PatternState>& pattern_states()
        const noexcept {
        return _pattern_states;
    }
    [[nodiscard]] const RetraceThreshold& retrace_dT() const noexcept {
        return _retrace_dT;
    }
    [[nodiscard]] bool write_back() const noexcept { return _write_back; }
    [[nodiscard]] const RetracerStatistics& statistics() const noexcept {
        return _statistics;
    }

    // solvers::Retracer
    void begin_solve() override;
    void update() override;
    [[nodiscard]] bool accepts_convergence(double tolerance) const override;
    [[nodiscard]] std::span<const double> internal_load() const override;
    [[nodiscard]] const SparseMatrix& overflow() const override {
        return _overflow;
    }
    [[nodiscard]] solvers::RetracerCounters counters() const override;
    bool end_solve() override;

  private:
    // The IR properties of one material: its numbers and, when its
    // emissivity is a table, the table's view (resolved by initialize()).
    struct Material {
        std::array<float, 6> properties{};
        std::string emissivity_table;
        std::optional<pycanha::detail::TableView> view;
        // Where the table is lowest and highest [K].
        double low_temperature = 0.0;
        double high_temperature = 0.0;
    };

    // The rows of one optical state; `temperature` is a face's temperature.
    enum class RowState : std::uint8_t { LOW, MID, HIGH, CURRENT };
    [[nodiscard]] static RowState row_state(PatternState state);

    void check_model();
    void resolve_tables();
    void build_checks();
    void fill_rows(RowState state);
    // Traces with the current rows and reduces the result to nodes.
    [[nodiscard]] SparseMatrix trace_nodes();
    // Traces at the current temperatures and writes every coupling.
    void retrace();
    void write_couplings(const SparseMatrix& couplings);
    [[nodiscard]] bool retrace_due() const;
    void store_trace_temperatures();
    void refresh_load();
    [[nodiscard]] double face_temperature(std::size_t face) const;

    std::shared_ptr<ThermalMathematicalModel> _tmm;
    NodeNum _space_node;
    TraceSettings _trace;
    AccumConfig _accum;
    std::set<PatternState> _pattern_states;
    RetraceThreshold _retrace_dT;
    bool _write_back;

    // What the retracer keeps of the geometry model, per face.
    struct Geometry {
        std::vector<NodeNum> face_nodes;
        // Row of `materials`, or -1 without a material.
        std::vector<int> face_materials;
        std::vector<bool> face_active;
        std::vector<Material> materials;
        // One property row per face (face_material[f] = f, or -1 without a
        // material), so that every face can take its own temperature.
        MaterialTable rows;
    };
    [[nodiscard]] static Geometry read_geometry(const gmm::GeometryModel& gmm);

    // Declared before the scene, which is built from its rows.
    Geometry _geometry;
    RadiativeScene _scene;
    ExchangeAccumulator _accumulator;

    // Resolved by initialize().
    bool _initialized = false;
    std::vector<Index> _face_index;
    Index _space_index = -1;
    std::vector<Index> _checked;
    std::vector<double> _thresholds;
    std::vector<double> _trace_temperatures;
    SparseMatrix _last_couplings;
    SparseMatrix _overflow;
    std::vector<double> _load;
    StructureVersions _structure;
    std::uint64_t _parameters_structure = 0;
    int _solve_traces = 0;
    RetracerStatistics _statistics;
};

}  // namespace pycanha::radiative
