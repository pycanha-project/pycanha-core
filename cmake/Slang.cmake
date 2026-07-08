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
    elseif(UNIX AND NOT APPLE)
        set(_archive "slang-${_version}-linux-x86_64.tar.gz")
        set(_sha256 "5533415953112ddeb0a935755bdd2da5de530e6528a560a32ad809c9d9faf29c")
        set(_slangc_name "slangc")
    else()
        # macOS builds with PYCANHA_OPTION_RAYTRACING=False (stub) and never
        # reaches this. TODO(radiative): a Metal backend adds the
        # macos-aarch64 archive here (slangc also emits Metal).
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
endfunction()

# pycanha_add_slang_kernel(<name>
#     SOURCE <kernel.slang>
#     DEPENDS <module.slang> ...)      # Slang emits no depfile: list imports
#
# Compiles the kernel to SPIR-V (Vulkan 1.3, scalar block layout, explicit
# ray-query + 64-bit-atomics capabilities), validates the SPIR-V when
# spirv-val is available, and generates
# ${PYCANHA_KERNEL_EMBED_DIR}/pycanha-core/radiative/kernels/<name>_spv.h
# defining pycanha::radiative::kernels::<name>_spv. Creates the custom target
# pycanha_kernel_<name>; the caller wires it into the library.
function(pycanha_add_slang_kernel _name)
    cmake_parse_arguments(PARSE_ARGV 1 _kernel "" "SOURCE" "DEPENDS")
    if(NOT _kernel_SOURCE)
        message(FATAL_ERROR "pycanha_add_slang_kernel: SOURCE is required")
    endif()
    if(NOT PYCANHA_SLANGC)
        message(FATAL_ERROR "pycanha_add_slang_kernel: call pycanha_fetch_slang() first")
    endif()

    set(_embed_root "${CMAKE_BINARY_DIR}/kernel_embed")
    set(_embed_dir "${_embed_root}/pycanha-core/radiative/kernels")
    set(_spv "${CMAKE_CURRENT_BINARY_DIR}/kernels/${_name}.spv")
    set(_header "${_embed_dir}/${_name}_spv.h")
    get_filename_component(_kernel_dir "${_kernel_SOURCE}" DIRECTORY)

    set(_commands
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${CMAKE_CURRENT_BINARY_DIR}/kernels" "${_embed_dir}"
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
                 -DHEADER_OUTPUT=${_header}
                 -DVARIABLE_NAME=${_name}_spv
                 -P "${PROJECT_SOURCE_DIR}/cmake/EmbedSpirv.cmake")

    add_custom_command(
        OUTPUT "${_spv}" "${_header}"
        ${_commands}
        DEPENDS "${_kernel_SOURCE}" ${_kernel_DEPENDS}
                "${PROJECT_SOURCE_DIR}/cmake/EmbedSpirv.cmake"
        COMMENT "Compiling Slang kernel ${_name}"
        VERBATIM)

    add_custom_target(pycanha_kernel_${_name} DEPENDS "${_header}")
    set(PYCANHA_KERNEL_EMBED_DIR "${_embed_root}" PARENT_SCOPE)
endfunction()
