# Intel MKL Configuration

## Overview

pycanha-core uses Intel Math Kernel Library (MKL) for accelerated linear algebra operations. MKL is installed via pip in a Python virtual environment.

## Configuration Options

### PYCANHA_OPTION_USE_MKL

- **Type**: Boolean
- **Default**: ON
- **Description**: Enables MKL for accelerated computations through Eigen

### PYCANHA_OPTION_MKL_LINK

- **Type**: String ("static" or "dynamic")
- **Default**: "dynamic"
- **Description**: Controls MKL linkage type

## Installation

### Prerequisites

MKL is installed via pip (version 2025.3) inside a Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install mkl-devel==2025.3
```

### Build with Conan

Conan automatically installs MKL when building:

```bash
# Activate virtual environment first
.venv\Scripts\activate

# Build with dynamic MKL (default)
conan create . --build=missing

# Build with static MKL
conan create . --build=missing -o PYCANHA_OPTION_MKL_LINK=static

# Build without MKL
conan create . --build=missing -o PYCANHA_OPTION_USE_MKL=False
```

**Note**: Conan will fail if `VIRTUAL_ENV` is not set.

## How It Works

### Conan Recipe (`conanfile.py`)

The `_mkl_paths_from_pip_venv()` method:
1. Checks if `PYCANHA_OPTION_USE_MKL=True`
2. Verifies `VIRTUAL_ENV` is set
3. Installs `mkl-devel==2025.3` via pip
4. Sets `MKLROOT` environment variable
5. Configures runtime PATH/LD_LIBRARY_PATH

### CMake (`pycanha-core/CMakeLists.txt`)

```cmake
if(PYCANHA_OPTION_USE_MKL)
    set(MKL_INTERFACE "lp64")
    set(MKL_THREADING "intel_thread")
    set(MKL_LINK "${PYCANHA_OPTION_MKL_LINK}")
    
    find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
    target_link_libraries(${LIB_NAME} PUBLIC MKL::MKL)
    target_compile_definitions(${LIB_NAME} PUBLIC 
        EIGEN_USE_MKL_ALL
        PYCANHA_USE_MKL=1
    )
endif()
```

- **MKL_INTERFACE**: `lp64` (32-bit integers)
- **MKL_THREADING**: `intel_thread` (Intel OpenMP)
- **MKL_LINK**: `static` or `dynamic`

## Troubleshooting

### Error: VIRTUAL_ENV not set

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
```

### Error: MKL not found by CMake

```bash
# Install MKL
pip install mkl-devel==2025.3

# Set MKLROOT
# Windows:
$env:MKLROOT = "$env:VIRTUAL_ENV\Library"
# Linux/macOS:
export MKLROOT="$VIRTUAL_ENV"
```

### Error: DLL/shared library not found at runtime

**Windows:**
```powershell
$env:PATH = "$env:VIRTUAL_ENV\Library\bin;$env:PATH"
```

**Linux:**
```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"
```

(This is automatic when using Conan)
