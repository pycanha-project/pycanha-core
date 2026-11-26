#pragma once

#include <cstdint>

#include "pycanha-core/gmm/primitives/primitive.hpp"

namespace pycanha::gmm {

// What a cutter keeps of the geometry it cuts.
enum class CutSense : std::uint8_t {
    KeepOutside,  // the part inside the cutter is removed: a hole
    KeepInside,   // the part outside is removed: only what it encloses is left
};

namespace cutting {

// A cutter as the cut backend takes it: the solid, in the frame the cut is
// made in, and what it keeps of the target.
struct Cutter {
    Primitive solid;
    CutSense sense = CutSense::KeepOutside;
};

}  // namespace cutting
}  // namespace pycanha::gmm
