# Script mode (cmake -P): turns slangc's -reflection-json dump into a C++
# header of Metal buffer indices.
#
# This has to be generated rather than written by hand: Metal ignores the
# [[vk::binding]] attributes entirely and slangc numbers buffers in declaration
# order with the kernel's own parameters first, so the mapping DIFFERS between
# kernels and silently changes whenever a buffer is added or a compiler version
# moves. A wrong index binds a valid buffer to the wrong slot, which produces
# plausible but wrong results instead of an error.
#
# Arguments:
#   -DREFLECTION_INPUT=<kernel.reflect.json> -DHEADER_OUTPUT=<file.h>
#   -DKERNEL_NAME=<ident>
# Emits pycanha::radiative::kernels::<kernel>_binding_<parameter>.

if(NOT REFLECTION_INPUT OR NOT HEADER_OUTPUT OR NOT KERNEL_NAME)
    message(FATAL_ERROR
        "MetalBindings.cmake needs REFLECTION_INPUT, HEADER_OUTPUT and KERNEL_NAME")
endif()

file(READ "${REFLECTION_INPUT}" _json)
string(JSON _count ERROR_VARIABLE _err LENGTH "${_json}" parameters)
if(_err)
    message(FATAL_ERROR "${REFLECTION_INPUT} has no parameters array: ${_err}")
endif()
if(_count EQUAL 0)
    message(FATAL_ERROR "${REFLECTION_INPUT} declares no parameters")
endif()

set(_constants "")
math(EXPR _last "${_count} - 1")
foreach(_i RANGE ${_last})
    string(JSON _name GET "${_json}" parameters ${_i} name)
    string(JSON _index ERROR_VARIABLE _index_err
           GET "${_json}" parameters ${_i} binding index)
    if(_index_err)
        # A parameter with no buffer index (an inline uniform, say) has nothing
        # to bind; the host never asks for it.
        continue()
    endif()
    string(APPEND _constants
        "inline constexpr std::uint32_t ${KERNEL_NAME}_binding_${_name} = ${_index}U;\n")
endforeach()

get_filename_component(_input_name "${REFLECTION_INPUT}" NAME)
file(WRITE "${HEADER_OUTPUT}" "\
// Generated from ${_input_name} by MetalBindings.cmake — do not edit.
#pragma once

#include <cstdint>

namespace pycanha::radiative::kernels {

${_constants}
}  // namespace pycanha::radiative::kernels
")
