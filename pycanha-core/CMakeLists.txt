# Sources and Headers
#set(LIBRARY_SOURCES 
#        ""
#)

#set(LIBRARY_HEADERS 
#        "${CMAKE_CURRENT_SOURCE_DIR}/my_lib.hpp"
#        "${CMAKE_CURRENT_SOURCE_DIR}/core/sum.hpp"
#        "${CMAKE_CURRENT_SOURCE_DIR}/core/multiply.hpp"
#        "${CMAKE_CURRENT_SOURCE_DIR}/utils/package_info.hpp"
#)

# Configure the header "config.hpp" to pass the library name and version
set(LIB_VERSION ${PROJECT_VERSION})
set(PYCANHA_USE_MKL_INT 0)

#set(LIBRARY_INCLUDES "./" "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/configured_files/include")
#set(LIBRARY_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")

# External dependencies
#-----------------------------

# Eigen 5 from Conan currently exports Eigen3 CMake config/target names.
find_package(Eigen3 REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)
find_package(HDF5 REQUIRED COMPONENTS CXX)


# CDT: https://github.com/artem-ogre/CDT
# CDT is a header-only library, so we can use FetchContent
include(FetchContent)
if(NOT DEFINED CACHE{PYCANHA_OPTION_CDT_VERSION})
    message(FATAL_ERROR
        "PYCANHA_OPTION_CDT_VERSION is not defined. Configure with Conan so "
        "all dependency versions are provided from conanfile.py.")
endif()
FetchContent_Declare(
    cdt
    GIT_REPOSITORY
    https://github.com/artem-ogre/CDT.git
    GIT_TAG
    ${PYCANHA_OPTION_CDT_VERSION}
) 
FetchContent_GetProperties(cdt)
if(NOT cdt_POPULATED)
  FetchContent_Populate(cdt)
  add_subdirectory(${cdt_SOURCE_DIR}/CDT ${cdt_BINARY_DIR}/CDT)
endif()

#find_package(CDT REQUIRED CONFIG) # Use this if managed by conan


# MyLib Library
## For a compiled library:
#add_library(${LIB_NAME} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
#target_include_directories(${LIB_NAME} PUBLIC ${LIBRARY_INCLUDES})
#target_link_libraries( ${LIB_NAME} PRIVATE 
#    Eigen3::Eigen
#)

# Change from INTERFACE to STATIC or SHARED, depending on your needs
# Add all source files after the library name
add_library(${LIB_NAME} STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/triangulation.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/esatan.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/parameters/parameters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/parameters/formulas.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/coupling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/couplings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/couplingmatrices.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/conductivecouplings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/radiativecouplings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/thermalnetwork.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/nodes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/thermalmathematicalmodel.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/SparseUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/ss.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/sslu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/ts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/tscn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/tscnrl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/tscnrlds.cpp

)
target_include_directories(${LIB_NAME} 
    PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    SYSTEM PRIVATE
    $<BUILD_INTERFACE:${cdt_SOURCE_DIR}> # This adds the parent directory of CDT. So  #include <CDT/include/CDT.h> will work
)

target_link_libraries( ${LIB_NAME} 
    PUBLIC 
    Eigen3::Eigen
    spdlog::spdlog
    PRIVATE
    HDF5::HDF5
    CDT::CDT
)

# ============================================================================
# Intel MKL Configuration
# ============================================================================
if(PYCANHA_OPTION_USE_MKL)
    # Set MKL configuration before find_package
    # See MKLConfig.cmake for all available options
    set(MKL_INTERFACE "lp64")                          # LP64 = 32-bit integers
    set(MKL_THREADING "intel_thread")                  # Intel OpenMP
    set(MKL_LINK "dynamic")                            # Simplified: only dynamic mode
    # If static mode is needed again:
    #  1) restore PYCANHA_OPTION_MKL_LINK in conan/cmake options,
    #  2) pass it here as MKL_LINK,
    #  3) restore static Windows library names in conanfile.py:_mkl_system_libs().

    # MKL paths are resolved in Conan (pip install + discovery) and exposed via
    # MKL_DIR/MKL_ROOT to CMake. Keep CMake side minimal, similar to Eigen.
    if(NOT DEFINED CACHE{MKL_DIR})
        message(FATAL_ERROR
            "MKL_DIR is not defined. Configure with Conan so pip-installed MKL "
            "is discovered and exported to CMake.")
    endif()

    # NO_DEFAULT_PATH ensures CMake ONLY looks in MKL_DIR — cannot find system MKL
    find_package(MKL CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${MKL_DIR}")

    message(STATUS "Intel MKL found: ${MKL_VERSION}")
    message(STATUS "  Interface: ${MKL_INTERFACE}, Threading: ${MKL_THREADING}, Link: ${MKL_LINK}")
    message(STATUS "  MKL imported targets: ${MKL_IMPORTED_TARGETS}")

    # Link MKL using the official target (includes headers, compile options, and libraries)
    target_link_libraries(${LIB_NAME} PUBLIC MKL::MKL)

    # Configure Eigen to use MKL BLAS/LAPACK
    target_compile_definitions(${LIB_NAME} PUBLIC 
        EIGEN_USE_MKL_ALL      # Enable MKL for all Eigen operations
    )

    # Configure pycanha to use MKL
    target_compile_definitions(${LIB_NAME} PUBLIC PYCANHA_USE_MKL=1)
    set(PYCANHA_USE_MKL_INT 1)
    # MKL_VERSION is set by find_package(MKL) above
else()
    # MKL disabled - use Eigen native solvers
    set(PYCANHA_USE_MKL_INT 0)
    set(MKL_VERSION "N/A")
    target_compile_definitions(${LIB_NAME} PUBLIC PYCANHA_USE_MKL=0)
    message(STATUS "Intel MKL disabled - using Eigen native solvers")
endif()
# ============================================================================

# ============================================================================
# Logging and Profiling Configuration
# ============================================================================
# SPDLOG_ACTIVE_LEVEL controls compile-time log-level stripping.
if(PYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE)
    set(_pycanha_spdlog_active_level_define "SPDLOG_LEVEL_TRACE")
else()
    set(_pycanha_spdlog_active_level_define
        "$<IF:$<CONFIG:Debug>,SPDLOG_LEVEL_TRACE,SPDLOG_LEVEL_INFO>")
endif()

target_compile_definitions(${LIB_NAME} PUBLIC
    "SPDLOG_ACTIVE_LEVEL=${_pycanha_spdlog_active_level_define}")
message(STATUS "PYCANHA SPDLOG_ACTIVE_LEVEL=${_pycanha_spdlog_active_level_define}")

# PYCANHA_PROFILING gates the PYCANHA_PROFILE_SCOPE() macro and the
# spdlog profiling logger.
if(PYCANHA_OPTION_PROFILING)
    target_compile_definitions(${LIB_NAME} PUBLIC PYCANHA_PROFILING)
    message(STATUS "Profiling logging enabled (PYCANHA_PROFILING)")
else()
    message(STATUS "Profiling logging disabled")
endif()
# ============================================================================

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/${LIB_NAME}/config.hpp.in" 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/${LIB_NAME}/config.hpp" 
)

# For a header-only library:
#add_library(${LIB_NAME} INTERFACE)
#target_include_directories(${LIB_NAME} 
#    INTERFACE 
#    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#    $<INSTALL_INTERFACE:include>
#)

#target_link_libraries( ${LIB_NAME} INTERFACE 
#    Eigen3::Eigen
#    CDT::CDT
#)

target_set_warnings(
    TARGET
    ${LIB_NAME}
    ENABLE
    ${PYCANHA_OPTION_WARNINGS}
    AS_ERRORS
    ${PYCANHA_OPTION_WARNINGS_AS_ERRORS})


if(${PYCANHA_OPTION_LTO})
    target_enable_lto(${LIB_NAME} optimized)
endif()

# Add clang-tidy to the library
if (PYCANHA_OPTION_CLANG_TIDY)
  add_clang_tidy_to_target(${LIB_NAME})
endif()

