#include "auto_configure.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "pycanha-core/solvers/settings.hpp"

namespace pycanha::solvers::detail {

namespace {

AutoField text(std::string_view name, std::string_view value,
               std::string_view reason) {
    return {.name = name, .value = std::string(value), .reason = reason};
}

AutoField flag(std::string_view name, bool value, std::string_view reason) {
    return {.name = name, .value = value, .reason = reason};
}

constexpr std::string_view pardiso_reason =
    "this build has MKL: PARDISO factorizes in parallel";
constexpr std::string_view auto_reason =
    "Cholesky where the matrix is symmetric, LU otherwise";
constexpr std::string_view threads_reason = "MKL's own thread setting";
constexpr std::string_view safe_reason =
    "the two-level factorization with nested dissection does not livelock";
constexpr std::string_view fallback_reason =
    "fallback, tried when the preferred configuration does not fit "
    "max_memory";

// PARDISO with the two-level factorization, the safe configuration.
AutoCandidate pardiso_two_level(bool with_reuse_field,
                                std::string_view reuse_reason) {
    AutoCandidate candidate{
        {text("backend", "PARDISO", pardiso_reason),
         text("factorization", "AUTO", auto_reason),
         {.name = "threads",
          .value = std::int64_t{0},
          .reason = threads_reason},
         text("pardiso.level", "TWO_LEVEL", safe_reason),
         text("pardiso.ordering", "NESTED_DISSECTION", safe_reason)}};
    if (with_reuse_field) {
        candidate.fields.push_back(
            flag("reuse_factors", /*value=*/false, reuse_reason));
    }
    return candidate;
}

// PARDISO one-level with minimum degree, reusing factors or as a fallback.
struct MinDegreeReasons {
    std::string_view reuse;
    std::string_view level;
    std::string_view ordering;
};

constexpr MinDegreeReasons reuse_reasons{
    .reuse =
        "the matrix changes little between time steps, so iterating on "
        "the previous factors avoids most factorizations; the minimum "
        "degree analysis this needs is done once, and can take long on "
        "models where a few nodes are coupled to many",
    .level = "reuse_factors needs the one-level factorization",
    .ordering =
        "nested dissection can livelock with the one-level "
        "factorization"};

constexpr MinDegreeReasons fallback_reasons{.reuse = fallback_reason,
                                            .level = fallback_reason,
                                            .ordering = fallback_reason};

AutoCandidate pardiso_min_degree(bool with_reuse_field, bool reuse,
                                 const MinDegreeReasons& reasons) {
    AutoCandidate candidate{
        {text("backend", "PARDISO", pardiso_reason),
         text("factorization", "AUTO", auto_reason),
         {.name = "threads",
          .value = std::int64_t{0},
          .reason = threads_reason},
         text("pardiso.level", "ONE_LEVEL", reasons.level),
         text("pardiso.ordering", "MIN_DEGREE", reasons.ordering)}};
    if (with_reuse_field) {
        candidate.fields.push_back(flag("reuse_factors", reuse, reasons.reuse));
    }
    return candidate;
}

AutoCandidate eigen(std::string_view ordering, std::string_view reason) {
    return {{text("backend", "EIGEN", "this build has no MKL"),
             text("factorization", "AUTO", auto_reason),
             text("eigen.ordering", ordering, reason)}};
}

}  // namespace

std::vector<AutoCandidate> auto_candidates(SolverType type,
                                           bool pardiso_available) {
    if (!pardiso_available) {
        return {eigen("COLAMD",
                      "the ordering with the least fill on most "
                      "models"),
                eigen("AMD", fallback_reason)};
    }
    if (type == SolverType::STEADY_STATE) {
        return {pardiso_two_level(/*with_reuse_field=*/true,
                                  "a steady state factorizes again only when "
                                  "the matrix changes"),
                pardiso_min_degree(/*with_reuse_field=*/true, /*reuse=*/false,
                                   fallback_reasons)};
    }
    if (type == SolverType::CRANK_NICOLSON) {
        return {pardiso_min_degree(/*with_reuse_field=*/true, /*reuse=*/true,
                                   reuse_reasons),
                pardiso_two_level(/*with_reuse_field=*/true, fallback_reason)};
    }
    // The sensitivity solve needs the factors of the current matrix, so
    // reuse_factors does not apply.
    return {pardiso_two_level(/*with_reuse_field=*/false, {}),
            pardiso_min_degree(/*with_reuse_field=*/false, /*reuse=*/false,
                               fallback_reasons)};
}

void clear_auto_fields(Settings& settings) {
    settings.backend.reset();
    settings.factorization.reset();
    settings.reuse_factors.reset();
    settings.threads.reset();
    settings.pardiso.ordering.reset();
    settings.pardiso.level.reset();
    settings.pardiso.iparm.reset();
    settings.eigen.ordering.reset();
}

}  // namespace pycanha::solvers::detail
