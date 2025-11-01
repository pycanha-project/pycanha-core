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

if(PYCANHA_OPTION_USE_MKL)
    find_package(MKL CONFIG
        PATHS
            "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl"
            "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/cmake/mkl"
        NO_DEFAULT_PATH QUIET)
    if(NOT TARGET MKL::MKL)
        message(FATAL_ERROR "Intel MKL requested (PYCANHA_OPTION_USE_MKL=ON) but MKL package targets were not found in the predefined lookup paths.")
    endif()

    target_link_libraries(${LIB_NAME} PUBLIC MKL::MKL)
    set(PYCANHA_USE_MKL_INT 1)
    target_compile_definitions(${LIB_NAME} PUBLIC PYCANHA_USE_MKL=1)
else()
    set(PYCANHA_USE_MKL_INT 0)
    target_compile_definitions(${LIB_NAME} PUBLIC PYCANHA_USE_MKL=0)
endif()

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

