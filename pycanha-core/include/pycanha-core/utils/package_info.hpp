#pragma once
#include <print>

#include "pycanha-core/config.hpp"

namespace pycanha {

inline void print_package_info() {
    // PACKAGE INFO
    std::println("Printing package info:");
    std::println("---------------------");
    std::println("{} v{}", LIB_NAME, LIB_VERSION);
    std::println("Build type: {}", LIB_BUILD_TYPE);
    std::println("Compiler: {} v{}", LIB_COMPILER_INFO, LIB_COMPILER_VERSION);
    std::println("C++ Standard: {}", LIB_CPP_STANDARD);
#ifdef PYCANHA_USE_MKL
    std::println("MKL Enabled: {}", MKL_VERSION);
#else
    std::println("MKL Disabled");
#endif
    std::println("---------------------");
}
}  // namespace pycanha
