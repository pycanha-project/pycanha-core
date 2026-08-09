#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace pycanha::conduction {

/// Knobs of the gmm -> tmm conduction build.
struct TmmBuildOptions {
    /// Temperature written to every generated node.
    double initial_temperature = 0.0;
    /// Generate the in-plane conductors of each primitive's own cell grid.
    bool intra_primitive_conductors = true;
    /// Generate the side-1 <-> side-2 conductors of each face pair.
    bool through_thickness_conductors = true;
    /// Close the conductor ring between the last and the first angular cell of
    /// a primitive that spans a full revolution.
    bool close_full_revolution = true;
    /// Generated conductors at or below this value are dropped. The default
    /// keeps everything except exact zeros.
    double min_conductance = 0.0;
};

enum class DiagnosticCode : std::uint8_t {
    /// Geometry inside a boolean-cut group: its cell grid no longer exists, so
    /// the parametric integrals do not apply.
    CutGeometrySkipped,
    /// The primitive produces no faces at all (Cube is cutter-only).
    UnmeshedPrimitive,
    /// A side carrying node numbers that one of the active-side selectors
    /// excludes: either it takes part in neither physics, and so contributes
    /// nothing at all, or it is radiative only, and so defines nodes with
    /// capacitance but no conductors.
    InactiveSideSkipped,
    /// No bulk material on an active side: the side still defines nodes, but
    /// with no capacitance and no conductors.
    MissingBulk,
    /// Zero thickness on an active side.
    ZeroThickness,
    /// Zero conductivity on a conductively active side.
    ZeroConductivity,
    /// The two sides mapped to one node carry different bulk materials; the
    /// contributions are summed anyway.
    MixedBulkOnNode,
    /// A triangle's fan parametrisation is not orthogonal, so its conductors
    /// come from the discrete shared-edge fallback rather than a closed form.
    TriangleApproximated,
    /// The item has no node numbers assigned on any active side.
    NoNodeNumbers,
    /// A cell with zero parametric extent, which carries no conductance.
    DegenerateCell,
    /// A cell band reaching the axis of revolution (a disc down to r = 0, a
    /// cone or paraboloid apex, a sphere pole). The around-the-axis
    /// conductance of that band uses the near-axis form -- meridian length
    /// over the reference radius -- because the temperature difference between
    /// neighbouring angular cells vanishes at the axis instead of staying
    /// constant across the band.
    AxisSingularity,
};

[[nodiscard]] std::string_view to_string(DiagnosticCode code) noexcept;

struct BuildDiagnostic {
    DiagnosticCode code = DiagnosticCode::CutGeometrySkipped;
    std::string geometry_name;
    std::string message;
};

struct TmmBuildReport {
    std::size_t items_processed = 0;
    std::size_t items_skipped = 0;
    std::size_t nodes_created = 0;
    /// Node-pair level, after aggregation.
    std::size_t conductors_created = 0;
    /// Face level, before aggregation.
    std::size_t cell_links_computed = 0;
    std::vector<BuildDiagnostic> diagnostics;
};

}  // namespace pycanha::conduction
