#pragma once

#include <cstdint>
#include <string_view>

namespace pycanha {

// Who wrote a formula or a table. A generator owns what it wrote under its
// origin: a regeneration removes or overwrites it, then writes it again.
// Anything added one at a time is a user formula or table.
enum class Origin : std::uint8_t {
    user,
    // The conduction builder that turns the geometry into the network.
    conduction_builder,
    // The reader of geometry models in the .erg / .etms format.
    etms_reader,
    // The reader of thermal model input files (.d).
    esatan_tmm_reader,
    // The reader of geometry models in the STEP-TAS format.
    steptas_gmm_reader,
};

// The enumerator name, e.g. "conduction_builder".
[[nodiscard]] std::string_view to_string(Origin origin) noexcept;

}  // namespace pycanha
