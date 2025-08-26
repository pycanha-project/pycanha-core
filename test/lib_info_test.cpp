#include "pycanha-core/pycanha-core.hpp"  // NOLINT (misc-include-cleaner)

using namespace pycanha;

// Supress clang-tidy warning that main should'n throw exceptions
//  NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    print_package_info();  // NOLINT (misc-include-cleaner)

    return 0;
}
