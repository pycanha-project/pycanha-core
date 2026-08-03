# Script mode (cmake -P): converts an arbitrary binary file into a C++ header
# with an inline constexpr uint8_t array plus its size. Sibling of
# EmbedSpirv.cmake, which embeds 32-bit words instead; a .metallib is an opaque
# byte stream and the size is needed explicitly at the use site (deriving it
# from sizeof() there is easy to get wrong). Arguments:
#   -DBINARY_INPUT=<file> -DHEADER_OUTPUT=<file.h> -DVARIABLE_NAME=<ident>

if(NOT BINARY_INPUT OR NOT HEADER_OUTPUT OR NOT VARIABLE_NAME)
    message(FATAL_ERROR
        "EmbedBinary.cmake needs BINARY_INPUT, HEADER_OUTPUT and VARIABLE_NAME")
endif()

file(READ "${BINARY_INPUT}" _hex HEX)
string(LENGTH "${_hex}" _hex_len)
math(EXPR _size "${_hex_len} / 2")
if(_size EQUAL 0)
    message(FATAL_ERROR "${BINARY_INPUT} is empty")
endif()

string(REGEX REPLACE "(..)" "0x\\1," _bytes "${_hex}")

# 16 bytes per line to keep the generated file readable/diffable. CMake's regex
# flavour has no {n} bounded repetition, so the group is spelled out longhand —
# otherwise this is one multi-hundred-kilobyte source line.
set(_b "0x[0-9a-f][0-9a-f],")
string(REGEX REPLACE
    "(${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b})"
    "\\1\n    " _bytes "${_bytes}")

get_filename_component(_input_name "${BINARY_INPUT}" NAME)
file(WRITE "${HEADER_OUTPUT}" "\
// Generated from ${_input_name} by EmbedBinary.cmake — do not edit.
#pragma once

#include <cstddef>
#include <cstdint>

namespace pycanha::radiative::kernels {

inline constexpr std::uint8_t ${VARIABLE_NAME}[] = {
    ${_bytes}
};

inline constexpr std::size_t ${VARIABLE_NAME}_size = ${_size};

}  // namespace pycanha::radiative::kernels
")
