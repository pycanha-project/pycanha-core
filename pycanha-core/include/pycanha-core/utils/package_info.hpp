#pragma once
#include <iostream>

#include "pycanha-core/config.hpp"

namespace pycanha {

inline void print_package_info() {
    // PACKAGE INFO
    std::cout << "Printing package info:\n";
    std::cout << "---------------------\n";
    std::cout << LIB_NAME << " v" << LIB_VERSION << "\n";
    std::cout << "Build type: " << LIB_BUILD_TYPE << "\n";
    std::cout << "Compiler: " << LIB_COMPILER_INFO << " v"
              << LIB_COMPILER_VERSION << "\n";
    std::cout << "C++ Standard: " << LIB_CPP_STANDARD << "\n";
    #ifdef PYCANHA_USE_MKL
    std::cout << "MKL Enabled\n";
    #endif
    std::cout << "---------------------\n";
}
}  // namespace pycanha
