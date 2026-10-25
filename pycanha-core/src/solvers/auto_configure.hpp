#pragma once

#include <string_view>
#include <vector>

#include "pycanha-core/solvers/settings.hpp"

namespace pycanha::solvers::detail {

/// One setting an auto_configure() candidate sets, with the reason.
struct AutoField {
    std::string_view name;
    SettingValue value;
    std::string_view reason;
};

/// One configuration auto_configure() may choose.
struct AutoCandidate {
    std::vector<AutoField> fields;
};

/// The candidates for a solver type, preferred first; the later ones are
/// tried only when the earlier ones do not fit max_memory.
[[nodiscard]] std::vector<AutoCandidate> auto_candidates(
    SolverType type, bool pardiso_available);

/// The settings auto_configure() owns: backend, factorization,
/// reuse_factors, threads, pardiso.ordering, pardiso.level, pardiso.iparm
/// and eigen.ordering. Unsets them.
void clear_auto_fields(Settings& settings);

}  // namespace pycanha::solvers::detail
