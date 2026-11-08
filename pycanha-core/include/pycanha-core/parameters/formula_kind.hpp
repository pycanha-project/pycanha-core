#pragma once

#include <cstdint>

namespace pycanha {

// What a formula reads. It follows from the formula's inputs and decides how
// the formula is evaluated and whether it has derivatives.
enum class FormulaKind : std::uint8_t {
    // Reads nothing: only numbers.
    Constant,
    // Reads parameters only (including the time, the values of tables
    // evaluated at a parameter, and table calls). These are the formulas
    // with derivatives.
    Parameters,
    // Reads at least one entity (temperature, capacity, load or coupling).
    Entities,
    // Runs a C++ callable, which may read anything: it counts as reading
    // every entity.
    Callable,
};

}  // namespace pycanha
