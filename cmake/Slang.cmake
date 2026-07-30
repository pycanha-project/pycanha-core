# Slang toolchain acquisition + kernel build rules. No Conan recipe exists
# for slang, so a pinned prebuilt GitHub release is fetched at configure
# time and verified by SHA256. Shaders are inputs to the BUILD; neither
# .slang sources, .spv binaries nor slangc are ever shipped.

include(FetchContent)

# Fetches the pinned slangc for the host platform and sets PYCANHA_SLANGC.
# PYCANHA_OPTION_SLANG_VERSION comes from conanfile.py (DEPENDENCY_VERSIONS);
# the per-version SHA256 pins live here — extend the table to bump versions.
function(pycanha_fetch_slang)
    if(NOT PYCANHA_OPTION_SLANG_VERSION)
        message(FATAL_ERROR "PYCANHA_OPTION_SLANG_VERSION is not set (configure through Conan)")
    endif()
    set(_version "${PYCANHA_OPTION_SLANG_VERSION}")

    if(NOT _version STREQUAL "2026.12.2")
        message(FATAL_ERROR
            "No SHA256 pins for slang ${_version} in cmake/Slang.cmake; "
            "download the release archives for each platform, compute their "
            "sha256 and extend the table below")
    endif()
    if(WIN32)
        set(_archive "slang-${_version}-windows-x86_64.zip")
        set(_sha256 "e44a29e4ba766e892db19e7f491b0c1fc21f548a0380a1b2931039569bf747e7")
        set(_slangc_name "slangc.exe")
    elseif(APPLE)
        # Apple Silicon only: the Metal backend requires GPU family Apple9
        # (M3/M4/A17 Pro) for its 64-bit buffer atomics, so an x86_64 macOS
        # host could never run the kernels anyway.
        if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            message(FATAL_ERROR
                "The Metal kernels require Apple Silicon (found "
                "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
        endif()
        set(_archive "slang-${_version}-macos-aarch64.zip")
        set(_sha256 "d1b6aac3f10b0031ef3ba7b0432ab678180aee745ff49d6d77b74271ced7bbbc")
        set(_slangc_name "slangc")
    elseif(UNIX)
        set(_archive "slang-${_version}-linux-x86_64.tar.gz")
        set(_sha256 "5533415953112ddeb0a935755bdd2da5de530e6528a560a32ad809c9d9faf29c")
        set(_slangc_name "slangc")
    else()
        message(FATAL_ERROR "No pinned slang archive for this platform")
    endif()

    FetchContent_Declare(
        slang_toolchain
        URL "https://github.com/shader-slang/slang/releases/download/v${_version}/${_archive}"
        URL_HASH "SHA256=${_sha256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(slang_toolchain)

    set(_slangc "${slang_toolchain_SOURCE_DIR}/bin/${_slangc_name}")
    if(NOT EXISTS "${_slangc}")
        message(FATAL_ERROR "slangc not found in the fetched archive: ${_slangc}")
    endif()
    set(PYCANHA_SLANGC "${_slangc}" PARENT_SCOPE)
    message(STATUS "Slang ${_version}: ${_slangc}")

    # spirv-val catches compiler/driver SPIR-V mismatches at build time
    # instead of first-run time. Optional — the Conan spirv-tools
    # tool-require puts it on the build env PATH; skip quietly when absent.
    find_program(PYCANHA_SPIRV_VAL spirv-val)
    if(PYCANHA_SPIRV_VAL)
        message(STATUS "spirv-val: ${PYCANHA_SPIRV_VAL}")
    else()
        message(STATUS "spirv-val not found — SPIR-V validation step skipped")
    endif()

    # The Metal path shells out to the Xcode toolchain (metal / metallib) via
    # xcrun. Required, not optional: unlike spirv-val this is the compile step
    # itself, and it doubles as the shader syntax check.
    if(APPLE)
        find_program(PYCANHA_XCRUN xcrun REQUIRED)
        message(STATUS "xcrun: ${PYCANHA_XCRUN}")

        # Since Xcode 26 the Metal compiler is an unbundled component that is
        # NOT installed with Xcode itself, and it is installed per-user. Probing
        # here turns a confusing mid-build failure into an actionable configure
        # error.
        execute_process(
            COMMAND "${PYCANHA_XCRUN}" metal --version
            RESULT_VARIABLE _metal_result
            OUTPUT_VARIABLE _metal_version
            ERROR_VARIABLE _metal_error)
        if(NOT _metal_result EQUAL 0)
            message(FATAL_ERROR
                "The Metal compiler is not usable:\n${_metal_error}\n"
                "Install it with:  xcodebuild -downloadComponent MetalToolchain\n"
                "(it is installed per-user, so run it as the account that "
                "builds)")
        endif()
        string(REGEX REPLACE "\n.*" "" _metal_version "${_metal_version}")
        message(STATUS "metal: ${_metal_version}")
    endif()
endfunction()

# pycanha_add_slang_kernel(<name>
#     SOURCE <kernel.slang>
#     TARGET <spirv|metal>            # default: spirv
#     DEPENDS <module.slang> ...)     # Slang emits no depfile: list imports
#
# TARGET spirv (Vulkan backend): compiles to SPIR-V (Vulkan 1.3, scalar block
# layout, explicit ray-query + 64-bit-atomics capabilities), validates it when
# spirv-val is available, and generates
# ${PYCANHA_KERNEL_EMBED_DIR}/pycanha-core/radiative/kernels/<name>_spv.h
# defining pycanha::radiative::kernels::<name>_spv.
#
# TARGET metal (Metal backend): compiles to MSL, then through the Xcode
# toolchain to a .metallib, and generates <name>_metallib.h (the library bytes,
# for newLibraryWithData:) plus <name>_bindings.h (the buffer indices, which
# slangc assigns itself — see cmake/MetalBindings.cmake for why they cannot be
# hand-written). The MSL entry point keeps its Slang name, csMain, unlike the
# SPIR-V path where slangc renames it to main.
#
# Either way the shaders are inputs to the BUILD — no .slang source, .spv,
# .metal or .metallib is ever shipped — and compilation IS the syntax check.
# Creates the custom target pycanha_kernel_<name>; the caller wires it in.
function(pycanha_add_slang_kernel _name)
    cmake_parse_arguments(PARSE_ARGV 1 _kernel "" "SOURCE;TARGET" "DEPENDS")
    if(NOT _kernel_SOURCE)
        message(FATAL_ERROR "pycanha_add_slang_kernel: SOURCE is required")
    endif()
    if(NOT PYCANHA_SLANGC)
        message(FATAL_ERROR "pycanha_add_slang_kernel: call pycanha_fetch_slang() first")
    endif()
    if(NOT _kernel_TARGET)
        set(_kernel_TARGET spirv)
    endif()

    set(_embed_root "${CMAKE_BINARY_DIR}/kernel_embed")
    set(_embed_dir "${_embed_root}/pycanha-core/radiative/kernels")
    set(_work "${CMAKE_CURRENT_BINARY_DIR}/kernels")
    get_filename_component(_kernel_dir "${_kernel_SOURCE}" DIRECTORY)

    set(_commands
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_work}" "${_embed_dir}")

    if(_kernel_TARGET STREQUAL "spirv")
        set(_spv "${_work}/${_name}.spv")
        set(_outputs "${_spv}" "${_embed_dir}/${_name}_spv.h")
        set(_scripts "${PROJECT_SOURCE_DIR}/cmake/EmbedSpirv.cmake")
        list(APPEND _commands
             COMMAND "${PYCANHA_SLANGC}" "${_kernel_SOURCE}"
                     -I "${_kernel_dir}"
                     -profile glsl_460
                     -capability spvRayQueryKHR -capability spvInt64Atomics
                     -target spirv -O2 -fvk-use-scalar-layout
                     -entry csMain -o "${_spv}")
        if(PYCANHA_SPIRV_VAL)
            list(APPEND _commands
                 COMMAND "${PYCANHA_SPIRV_VAL}" --target-env vulkan1.3 "${_spv}")
        endif()
        list(APPEND _commands
             COMMAND "${CMAKE_COMMAND}"
                     -DSPV_INPUT=${_spv}
                     -DHEADER_OUTPUT=${_embed_dir}/${_name}_spv.h
                     -DVARIABLE_NAME=${_name}_spv
                     -P "${PROJECT_SOURCE_DIR}/cmake/EmbedSpirv.cmake")
    elseif(_kernel_TARGET STREQUAL "metal")
        if(NOT PYCANHA_XCRUN)
            message(FATAL_ERROR
                "pycanha_add_slang_kernel: TARGET metal needs xcrun (Apple only)")
        endif()
        set(_msl "${_work}/${_name}.metal")
        set(_reflect "${_work}/${_name}.reflect.json")
        set(_air "${_work}/${_name}.air")
        set(_metallib "${_work}/${_name}.metallib")
        set(_outputs "${_metallib}"
                     "${_embed_dir}/${_name}_metallib.h"
                     "${_embed_dir}/${_name}_bindings.h")
        set(_scripts "${PROJECT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
                     "${PROJECT_SOURCE_DIR}/cmake/MetalBindings.cmake")
        list(APPEND _commands
             # No -fvk-use-scalar-layout here: it is a Vulkan-only flag. The
             # Push/InstanceData structs are declared so that MSL's natural
             # layout and Vulkan's scalar layout agree byte for byte (see the
             # sun_dir comment in kernels/common.slang), which is what lets one
             # C++ mirror struct serve both backends.
             COMMAND "${PYCANHA_SLANGC}" "${_kernel_SOURCE}"
                     -I "${_kernel_dir}"
                     -target metal -O2
                     -entry csMain
                     -reflection-json "${_reflect}"
                     -o "${_msl}"
             # -Wno-unused-variable: slangc assigns the result of every
             # InterlockedAdd and of bool-returning helpers to a temporary it
             # then ignores. The warnings are about GENERATED code we cannot fix
             # at the source, and left on they bury the real diagnostics.
             COMMAND "${PYCANHA_XCRUN}" metal -Wno-unused-variable
                     -c "${_msl}" -o "${_air}"
             COMMAND "${PYCANHA_XCRUN}" metallib "${_air}" -o "${_metallib}"
             COMMAND "${CMAKE_COMMAND}"
                     -DBINARY_INPUT=${_metallib}
                     -DHEADER_OUTPUT=${_embed_dir}/${_name}_metallib.h
                     -DVARIABLE_NAME=${_name}_metallib
                     -P "${PROJECT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
             COMMAND "${CMAKE_COMMAND}"
                     -DREFLECT_INPUT=${_reflect}
                     -DHEADER_OUTPUT=${_embed_dir}/${_name}_bindings.h
                     -DKERNEL_NAME=${_name}
                     -P "${PROJECT_SOURCE_DIR}/cmake/MetalBindings.cmake")
    else()
        message(FATAL_ERROR
            "pycanha_add_slang_kernel: unknown TARGET '${_kernel_TARGET}' "
            "(expected spirv or metal)")
    endif()

    add_custom_command(
        OUTPUT ${_outputs}
        ${_commands}
        DEPENDS "${_kernel_SOURCE}" ${_kernel_DEPENDS} ${_scripts}
        COMMENT "Compiling Slang kernel ${_name} (${_kernel_TARGET})"
        VERBATIM)

    add_custom_target(pycanha_kernel_${_name} DEPENDS ${_outputs})
    set(PYCANHA_KERNEL_EMBED_DIR "${_embed_root}" PARENT_SCOPE)
endfunction()
