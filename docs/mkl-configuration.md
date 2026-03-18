# Intel MKL Configuration

## Overview

pycanha-core uses Intel Math Kernel Library (MKL) for accelerated linear algebra operations. MKL is installed automatically via `pip` during the Conan build. No Intel oneAPI toolkit installation or `setvars` script is required.

## Configuration Options

### PYCANHA_OPTION_USE_MKL

- **Type**: Boolean
- **Default**: ON
- **Description**: Enables MKL for accelerated computations through Eigen

### PYCANHA_OPTION_MKL_VERSION

- **Type**: String
- **Default**: `"2025.3.0"`
- **Description**: The version of `mkl-devel` to install via pip

### PYCANHA_OPTION_MKL_LINK

- **Type**: String (`"static"` or `"dynamic"`)
- **Default**: `"dynamic"`
- **Description**: Controls MKL linkage type

## Installation

### Prerequisites

A working Python environment with `pip` available. This can be a virtual environment (venv), a conda environment, or the system Python — the Conan recipe handles everything else.

### Build with Conan

```bash
# Default: dynamic MKL, version 2025.3.0
conan create . --build=missing

# Static MKL linkage
conan create . --build=missing -o PYCANHA_OPTION_MKL_LINK=static

# Specific MKL version
conan create . --build=missing -o PYCANHA_OPTION_MKL_VERSION=2024.2.0

# Disable MKL entirely
conan create . --build=missing -o PYCANHA_OPTION_USE_MKL=False
```

## How It Works

### Conan Recipe (`conanfile.py`)

When `PYCANHA_OPTION_USE_MKL=True`, the `generate()` method:

1. Runs `pip install mkl-devel=={version}` in the current Python environment.
2. Discovers the path to `MKLConfig.cmake` via `importlib.metadata` in a subprocess (to pick up freshly-installed packages).
3. Derives all MKL directories from the location of `MKLConfig.cmake`:
   - **Windows pip layout**: `<prefix>/Library/lib/cmake/mkl/MKLConfig.cmake`
   - **Linux pip layout**: `<prefix>/lib/cmake/mkl/MKLConfig.cmake`
   - From the cmake config file, going up 3 directories gives the MKL root (`<prefix>/Library` on Windows, `<prefix>` on Linux).
4. Passes `MKL_DIR` (path to `MKLConfig.cmake`'s directory) to CMake as a cache variable.
5. Exports `MKLROOT`, `PATH` (Windows) / `LD_LIBRARY_PATH` (Linux) environment scripts so tests can find shared libraries at runtime.

The `package_info()` method exports MKL library names and directories so that downstream consumers (e.g. pycanha-core-python) correctly link against MKL without any additional setup:

- On **Windows**, MKL import libraries go into `cpp_info.libs` (Conan validates `.lib` files).
- On **Linux**, MKL libraries go into `cpp_info.system_libs` (pip ships versioned SONAMEs like `libmkl_intel_lp64.so.2` which Conan can't validate as regular libs).

### CMake (`pycanha-core/CMakeLists.txt`)

```cmake
if(PYCANHA_OPTION_USE_MKL)
    set(MKL_INTERFACE "lp64")
    set(MKL_THREADING "intel_thread")
    set(MKL_LINK "${PYCANHA_OPTION_MKL_LINK}")

    find_package(MKL CONFIG REQUIRED)
    target_link_libraries(${LIB_NAME} PUBLIC MKL::MKL)
    target_compile_definitions(${LIB_NAME} PUBLIC
        EIGEN_USE_MKL_ALL
        PYCANHA_USE_MKL=1
    )
endif()
```

CMake finds MKL via the `MKL_DIR` variable set by Conan, which points directly to the pip-installed `MKLConfig.cmake`. No environment variables are needed for CMake discovery.

- **MKL_INTERFACE**: `lp64` (32-bit integers)
- **MKL_THREADING**: `intel_thread` (Intel OpenMP)
- **MKL_LINK**: `static` or `dynamic`

## Troubleshooting

### Error: MKL not found by CMake

- Check that `pip install mkl-devel` succeeded (look for the Conan output line `Installing mkl-devel==...`).
- Verify with `python -c "from importlib.metadata import distribution; print(distribution('mkl-devel').locate_file('.'))"` that the package is installed in the expected environment.
- If using a virtual environment, make sure it is activated before running Conan.

### Error: DLL/shared library not found at runtime

- The Conan recipe generates `mkl_run_env` scripts that set `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux). These are sourced automatically by Conan during test execution.
- For manual testing, source the generated script: `source build/generators/mkl_run_env.sh` (Linux) or `.\build\generators\mkl_run_env.bat` (Windows).
- When packaging applications, ensure MKL runtime redistributables are deployed alongside your executable.

### Error: Unresolved MKL symbols in a consumer (e.g. pycanha-core-python)

- The consumer's Python environment must also have MKL pip packages installed. The `package_info()` method discovers MKL directories from the consumer's environment.
- Verify that `mkl-devel` is installed in the environment where `conan install` runs for the consumer project.
