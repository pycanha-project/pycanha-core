# Script mode (cmake -P): turns a slangc -reflection-json dump into a C++
# header of Metal buffer indices. Arguments:
#   -DREFLECT_INPUT=<file.json> -DHEADER_OUTPUT=<file.h> -DKERNEL_NAME=<ident>
#
# Why this exists: slangc IGNORES [[vk::binding(n, set)]] when targeting Metal
# and assigns buffer indices in declaration order, kernel-local parameters
# first. So the indices differ per kernel — `tlas` is buffer 2 in vf/exchange
# but buffer 4 in solar, which declares three accumulator buffers instead of
# one — and the flat "set 0, bindings 0-9 shared" layout of the Vulkan
# descriptor set does not carry over. Generating the table from the reflection
# dump keeps it correct across kernel edits and slang version bumps; a
# hand-written table would silently rot into wrong-results territory.

if(NOT REFLECT_INPUT OR NOT HEADER_OUTPUT OR NOT KERNEL_NAME)
    message(FATAL_ERROR "MetalBindings.cmake needs REFLECT_INPUT, HEADER_OUTPUT and KERNEL_NAME")
endif()

file(READ "${REFLECT_INPUT}" _json)

string(JSON _count ERROR_VARIABLE _err LENGTH "${_json}" parameters)
if(_err OR NOT _count GREATER 0)
    message(FATAL_ERROR "${REFLECT_INPUT} has no 'parameters' array: ${_err}")
endif()

set(_constants "")
math(EXPR _last "${_count} - 1")
foreach(_i RANGE ${_last})
    string(JSON _name GET "${_json}" parameters ${_i} name)
    string(JSON _index ERROR_VARIABLE _index_err
           GET "${_json}" parameters ${_i} binding index)
    if(_index_err)
        # A parameter with no buffer index is not bindable from the host (e.g.
        # a pure specialization constant); skipping it is correct, but say so
        # rather than emitting a silently incomplete table.
        message(STATUS "MetalBindings: ${KERNEL_NAME}.${_name} has no binding index, skipped")
        continue()
    endif()
    string(APPEND _constants
        "inline constexpr std::uint32_t ${KERNEL_NAME}_binding_${_name} = ${_index};\n")
endforeach()

get_filename_component(_reflect_name "${REFLECT_INPUT}" NAME)
file(WRITE "${HEADER_OUTPUT}" "\
// Generated from ${_reflect_name} by MetalBindings.cmake — do not edit.
#pragma once

#include <cstdint>

namespace pycanha::radiative::kernels {

${_constants}
}  // namespace pycanha::radiative::kernels
")
