#pragma once

#include <string>

#include "pycanha-core/conduction/options.hpp"

namespace pycanha {
class ThermalMathematicalModel;
}  // namespace pycanha

namespace pycanha::conduction::detail {

// Whether a diagnostic describes expected behaviour rather than something the
// build had to drop or assume.
[[nodiscard]] bool is_benign(DiagnosticCode code) noexcept;

// Records a diagnostic in the report; logging happens once the parts are
// committed, in part order, so a parallel build logs deterministically.
void report_diagnostic(TmmBuildReport& report, DiagnosticCode code,
                       std::string geometry_name, std::string message);

void log_diagnostic(const BuildDiagnostic& diagnostic);

// The builder writes into an empty tmm only: there is no merge semantics.
void require_empty_tmm(const ThermalMathematicalModel& tmm);

}  // namespace pycanha::conduction::detail
