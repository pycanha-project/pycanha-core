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

# Eigen3
find_package(Eigen3 REQUIRED) # Managed by conan


# CDT: https://github.com/artem-ogre/CDT
# CDT is a header-only library, so we can use FetchContent
include(FetchContent)
FetchContent_Declare(
    cdt
    GIT_REPOSITORY
    https://github.com/artem-ogre/CDT.git
    GIT_TAG
    1.4.4
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
    PRIVATE
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
    set(MKL_LINK "${PYCANHA_OPTION_MKL_LINK}")        # "static" or "dynamic"

    # --- Always use pip-installed MKL, never system oneAPI ---
    # Clear env vars that could leak to system MKL installations
    set(ENV{MKLROOT} "")

    # If MKL_DIR is not already set (e.g., by Conan), discover pip MKL
    if(NOT DEFINED CACHE{MKL_DIR})
        find_program(_PYCANHA_PYTHON python3 python)
        if(_PYCANHA_PYTHON)
            execute_process(
                COMMAND "${_PYCANHA_PYTHON}" -c
                    "import importlib.metadata as m; d=m.distribution('mkl-devel'); print(next(str(d.locate_file(f).resolve().parent) for f in d.files if f.name=='MKLConfig.cmake'))"
                OUTPUT_VARIABLE _PIP_MKL_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _PIP_MKL_RESULT
                ERROR_QUIET
            )
            if(_PIP_MKL_RESULT EQUAL 0 AND _PIP_MKL_DIR)
                set(MKL_DIR "${_PIP_MKL_DIR}" CACHE PATH "MKL cmake config (discovered from pip)" FORCE)
                message(STATUS "Discovered pip-installed MKL: MKL_DIR=${MKL_DIR}")
            endif()
        endif()
        if(NOT DEFINED CACHE{MKL_DIR})
            message(FATAL_ERROR
                "MKL_DIR not set and pip MKL not found. "
                "Install MKL via: pip install mkl-devel")
        endif()
    endif()

    # NO_DEFAULT_PATH ensures CMake ONLY looks in MKL_DIR — cannot find system MKL
    find_package(MKL CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${MKL_DIR}")

    # Promote MKL_ROOT to cache so sibling subdirectories (test/) can see it
    set(MKL_ROOT "${MKL_ROOT}" CACHE PATH "MKL root directory (from pip)" FORCE)

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

