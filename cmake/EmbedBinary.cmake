# Script mode (cmake -P): converts an arbitrary binary blob into a C++ header
# with an inline constexpr uint8_t array. Used for Metal .metallib libraries,
# which are byte streams (unlike SPIR-V, which is a stream of 32-bit words and
# has its own EmbedSpirv.cmake). Arguments:
#   -DBINARY_INPUT=<file> -DHEADER_OUTPUT=<file.h> -DVARIABLE_NAME=<ident>

if(NOT BINARY_INPUT OR NOT HEADER_OUTPUT OR NOT VARIABLE_NAME)
    message(FATAL_ERROR "EmbedBinary.cmake needs BINARY_INPUT, HEADER_OUTPUT and VARIABLE_NAME")
endif()

file(READ "${BINARY_INPUT}" _hex HEX)
string(LENGTH "${_hex}" _hex_len)
if(_hex_len EQUAL 0)
    message(FATAL_ERROR "${BINARY_INPUT} is empty")
endif()

string(REGEX REPLACE "(..)" "0x\\1," _bytes "${_hex}")

# 16 bytes per line to keep the generated file readable/diffable and to avoid
# emitting one multi-hundred-kilobyte source line. Spelled out because CMake's
# regex flavour has no {n} bounded repetition (same reason EmbedSpirv.cmake
# writes its 8 words out longhand).
string(REGEX REPLACE
    "(0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,)"
    "\\1\n    " _bytes "${_bytes}")

# The array length is needed by dispatch_data_create / newLibraryWithData:, and
# deriving it from sizeof() at the use site is easy to get wrong once the
# variable is passed around, so emit it explicitly.
math(EXPR _size "${_hex_len} / 2")

get_filename_component(_binary_name "${BINARY_INPUT}" NAME)
file(WRITE "${HEADER_OUTPUT}" "\
// Generated from ${_binary_name} by EmbedBinary.cmake — do not edit.
#pragma once

#include <cstddef>
#include <cstdint>

namespace pycanha::radiative::kernels {

inline constexpr std::size_t ${VARIABLE_NAME}_size = ${_size};

inline constexpr std::uint8_t ${VARIABLE_NAME}[] = {
    ${_bytes}
};

}  // namespace pycanha::radiative::kernels
")
