# Metal backend spike (PYCANHA_OPTION_METAL_SPIKE, Apple only).
#
# Transitional: this target exists to validate that a given Mac can run the
# radiative engine BEFORE the Metal backend itself (mtl_device / mtl_scene /
# mtl_accum) is written. It links only Metal and Foundation — not pycanha-core
# — so its result is a statement about the machine and the shader toolchain,
# not about the library. Delete it once the Metal backend lands and the real
# radiative tests cover this ground.

include(Slang)
pycanha_fetch_slang()

# The three production kernels, compiled to Metal through the same rule the
# library will use. Guarded because once the Metal backend exists the library
# itself defines these targets and we must reuse them rather than collide.
foreach(_kernel vf exchange solar)
    if(NOT TARGET pycanha_kernel_${_kernel})
        pycanha_add_slang_kernel(${_kernel}
            SOURCE ${PROJECT_SOURCE_DIR}/pycanha-core/src/radiative/kernels/${_kernel}.slang
            TARGET metal
            DEPENDS ${PROJECT_SOURCE_DIR}/pycanha-core/src/radiative/kernels/common.slang)
    endif()
endforeach()

# The self-contained runtime probe.
pycanha_add_slang_kernel(probe
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe.slang
    TARGET metal)

add_executable(metal_spike ${CMAKE_CURRENT_SOURCE_DIR}/main.mm)
add_dependencies(metal_spike
    pycanha_kernel_vf pycanha_kernel_exchange pycanha_kernel_solar
    pycanha_kernel_probe)
target_include_directories(metal_spike PRIVATE ${PYCANHA_KERNEL_EMBED_DIR})
target_compile_options(metal_spike PRIVATE -fobjc-arc)
target_link_libraries(metal_spike PRIVATE
    "-framework Metal"
    "-framework Foundation")

# Registered with CTest so `ctest -R metal_spike` is the one command to run on
# a new machine. It fails (non-zero exit) on any unmet gate.
add_test(NAME metal_spike COMMAND metal_spike)
