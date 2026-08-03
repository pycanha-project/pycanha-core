#pragma once

// Umbrella header of the conduction module: it turns a GeometryModel into the
// nodes and conductive couplings of a ThermalMathematicalModel. It sits
// between gmm and tmm the same way pycanha::radiative does, so gmm stays free
// of tmm knowledge and tmm free of gmm knowledge.

#include "pycanha-core/conduction/builder.hpp"  // IWYU pragma: export
#include "pycanha-core/conduction/links.hpp"    // IWYU pragma: export
#include "pycanha-core/conduction/options.hpp"  // IWYU pragma: export
#include "pycanha-core/conduction/profile.hpp"  // IWYU pragma: export
