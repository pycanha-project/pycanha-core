# Script mode (cmake -P): converts a SPIR-V binary into a C++ header with an
# inline constexpr uint32_t array (little-endian words, as vkCreateShaderModule
# expects pCode). Arguments:
#   -DSPV_INPUT=<file.spv> -DHEADER_OUTPUT=<file.h> -DVARIABLE_NAME=<ident>

if(NOT SPV_INPUT OR NOT HEADER_OUTPUT OR NOT VARIABLE_NAME)
    message(FATAL_ERROR "EmbedSpirv.cmake needs SPV_INPUT, HEADER_OUTPUT and VARIABLE_NAME")
endif()

file(READ "${SPV_INPUT}" _hex HEX)
string(LENGTH "${_hex}" _hex_len)
math(EXPR _rem "${_hex_len} % 8")
if(NOT _rem EQUAL 0)
    message(FATAL_ERROR "${SPV_INPUT} is not a whole number of 32-bit words")
endif()

# Bytes b0b1b2b3 (file order) -> little-endian word 0xb3b2b1b0.
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1," _words "${_hex}")

# 8 words per line to keep the generated file readable/diffable.
string(REGEX REPLACE "(0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,)" "\\1\n    " _words "${_words}")

get_filename_component(_spv_name "${SPV_INPUT}" NAME)
file(WRITE "${HEADER_OUTPUT}" "\
// Generated from ${_spv_name} by EmbedSpirv.cmake — do not edit.
#pragma once

#include <cstdint>

namespace pycanha::radiative::kernels {

inline constexpr std::uint32_t ${VARIABLE_NAME}[] = {
    ${_words}
};

}  // namespace pycanha::radiative::kernels
")
