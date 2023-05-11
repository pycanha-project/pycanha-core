#include "pycanha-core/pycanha-core.hpp"

// Supress clang-tidy warning that main should'n throw exceptions
//  NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    print_package_info();
    return 0;
}
