# Intel MKL Configuration

## Overview

`pycanha-core` supports Intel MKL acceleration through `PYCANHA_OPTION_USE_MKL`.

## Build Options

- `PYCANHA_OPTION_USE_MKL` (`ON`/`OFF`, default `ON`)
- `PYCANHA_OPTION_MKL_VERSION` (string, default `2025.3.0`)

`pycanha-core` now supports **dynamic MKL linking only**.

## Current Dependency Flow

### 1. Conan installs MKL from pip

When `PYCANHA_OPTION_USE_MKL=True`, `conanfile.py`:
- runs `pip install mkl-devel==<PYCANHA_OPTION_MKL_VERSION>` using the current Python interpreter,
- locates `MKLConfig.cmake` from pip package metadata,
- derives:
  - `MKL_ROOT`
  - `include` directory
  - `lib` directory
  - runtime `bin` directory (Windows) or `lib` (Linux/macOS).

Conan then exports CMake cache entries:
- `MKL_ROOT`
- `MKL_DIR` (`<mklroot>/lib/cmake/mkl`)

and runtime environment for tests/consumers:
- Windows: prepends MKL `bin` to `PATH`
- Linux/macOS: prepends MKL `lib` to `LD_LIBRARY_PATH`

### 2. CMake only consumes `MKL_DIR`

In `pycanha-core/CMakeLists.txt`, MKL handling is intentionally small:

```cmake
if(PYCANHA_OPTION_USE_MKL)
  set(MKL_INTERFACE "lp64")
  set(MKL_THREADING "intel_thread")
  set(MKL_LINK "dynamic")

  if(NOT DEFINED CACHE{MKL_DIR})
    message(FATAL_ERROR "MKL_DIR is not defined. Configure with Conan.")
  endif()

  find_package(MKL CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${MKL_DIR}")
  target_link_libraries(${LIB_NAME} PUBLIC MKL::MKL)
  target_compile_definitions(${LIB_NAME} PUBLIC EIGEN_USE_MKL_ALL PYCANHA_USE_MKL=1)
else()
  target_compile_definitions(${LIB_NAME} PUBLIC PYCANHA_USE_MKL=0)
endif()
```

This is now aligned with other dependencies such as Eigen from the CMake perspective:
- CMake only does `find_package(...)` + `target_link_libraries(...)`.
- Dependency acquisition/discovery happens in Conan.

## Typical Commands

```bash
# Default: MKL enabled, dynamic link mode
conan create . --build=missing

# Disable MKL
conan create . --build=missing -o PYCANHA_OPTION_USE_MKL=False

# Pin a specific pip MKL version
conan create . --build=missing -o PYCANHA_OPTION_MKL_VERSION=2025.3.0
```

To re-enable static MKL in the future, you must restore the removed
`PYCANHA_OPTION_MKL_LINK` option in Conan/CMake and the static Windows MKL
library list in `conanfile.py`.

## Runtime Notes

- Unit tests on Windows copy MKL DLLs next to the test executable (`test/CMakeLists.txt`) to avoid runtime loader issues during `catch_discover_tests`.
- Conan also exports runtime paths (`PATH`/`LD_LIBRARY_PATH`) so downstream executions can resolve MKL shared libraries.

## Troubleshooting

### `MKL_DIR is not defined`

Configure/build through Conan. Direct CMake configure without Conan will not install/discover pip MKL.

### Linker cannot find MKL libraries (Linux)

`conanfile.py` attempts to create unversioned `.so` symlinks when pip ships only versioned files (for example `.so.2`). Re-run the Conan build in an environment where symlink creation is allowed.

### Runtime error loading MKL shared libraries

- Windows: ensure MKL `bin` directory is available in `PATH`.
- Linux/macOS: ensure MKL `lib` is available in `LD_LIBRARY_PATH`.

When using Conan-generated environments, this is set automatically.
