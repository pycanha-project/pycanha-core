
# Using the tools

Here you can find a more detailed description of each tool and some tips on how to use them.
If you want practical examples on how to use them, check the tool documentation and the [CI workflows](cicd.md).

## Code formatting tools
As the name suggests, these tools format the code to follow pre-configured standard rules, so there is no need to worry about spacing, line length, new lines, etc.
They perform very basic tests that should not alter the code that the compiler sees.
So, for example, these tools would not reorder the `#include` statements.
Because of that, ideally you should configure your IDE to apply the format upon saving.



### clang-format
The format rules are configured in the root file `.clang-format`. Currently, the Google style is being used.

Most IDEs will dected the `.clang-format` file and automatically format the code.
In `VS Code` this works well with the [clangd extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd), which uses the repository `.clang-format` file when formatting C and C++ files.
You can also run manually the command `Format Document` from the command palette to format the file.

## Static analysis tools
Static analysis tools perform more complex checks than formatting tools, but they also take longer to analyze and can yield many false positives.
I would recomment to run the analysis everytime upon saving or opening the files if your pc can handle it.

### clang-tidy
The checks are specified in the `.clang-tidy` file. Right now basically checking for everything except some specific rules for other projects
or annoying checks like using auto return trailing types. The `.clang-tidy` is also configured to enforce the naming conventions I prefer.
In `VS Code`, the preferred setup is the [clangd extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd). It is usually faster than the Microsoft C++ language service for day-to-day editing, and it can surface `clang-tidy` diagnostics directly when started with the `--clang-tidy` argument. Because this repository generates `compile_commands.json`, `clangd` can pick up the project compilation flags and apply the checks from `.clang-tidy` consistently.

There is also a custom option in CMake, configurable through Conan, to create dedicated `clang-tidy` targets. Those targets are intentionally disabled for MSVC builds; on non-MSVC configurations they are added through `Tools.cmake` and exercised in the `code-checks.yml` workflow.
Because in the end it is another CMake target, it can be invoked directly from the CMake VS Code extension, and the code will be linted automatically. 

### cpplint
cpplint is in between a formatting and a static analysis tool. The rules are configured in the `CPPLINT.cfg` file.
In `VS Code` you can use it just by installing the [cpp-check-lint extension](https://marketplace.visualstudio.com/items?itemName=QiuMingGe.cpp-check-lint)
The extension will check and lint the code automatically when saving, but it can be configured. The extension also includes the `cppcheck` tool.

### cppcheck

cppcheck is a static analysis tool that detects errors and vulnerabilities in C and C++ code. It can identify memory leaks, buffer overruns, and other issues that might cause crashes or undefined behavior. In VS Code, you can use it by installing the [cpp-check-lint extension](https://marketplace.visualstudio.com/items?itemName=QiuMingGe.cpp-check-lint). The extension will check the code automatically when saving, but it can be configured. The extension also includes the cpplint tool.

Similar as with `clang-tidy` there is a `cppcheck` target for CMake to check all the relevant files of the library. The target is configured in the `CMakeLists.txt` file under the `test` directory. If you look at the target configuration, you can see that some files are excluded to avoid false positives. If you use `cppcheck` manually or through VS Code, you might want to exclude also those files.

### CodeQL
CodeQL is a code analysis engine developed by GitHub that can analyze source code to identify security vulnerabilities, bugs, and maintainability issues. CodeQL supports multiple languages, including C and C++. 
This repository does not currently define a dedicated CodeQL workflow. If a CodeQL check is reintroduced later, it should be treated as a repository-level analysis step rather than a required local development task.

## Testing tools

### Catch2
C++ library for unit testing. It integrates easily with CTest (CMake own test coordinator tool). In this template, everything related to the tests is under the `./test` folder. When the folder is included from the root `CMakeLists` using the `add_subdirectory` command, a `tests` target (and the clang-tidy and cppcheck targets if the options are activated) is made available. 

### Code coverage
Code coverage is a metric used in software testing to measure the extent to which the source code of a program has been executed by a test suite. It is expressed as a percentage, representing the ratio of the number of lines, branches, functions, or statements in the code that have been tested to the total number of lines, branches, functions, or statements.

Here, we use `gcov`, `lcov` and [Codecov](https://codecov.io/). Check the `coverage` workflow and the `coverage` CMake target for more info.


## Documentation

### Doxygen + graphviz + Doxygen Awesome CSS

Doxygen is a popular documentation generation tool for C++ code and it generates documentation from source code comments in a variety of formats, including HTML, LaTeX, and more. You can use it with `Graphviz` to also generate relationship diagrams of the classes. The default theme of doxygen is a bit outdated, so you can use CSS templates such as [Doxygen Awesome](https://github.com/jothepro/doxygen-awesome-css) to make it much better. 


## Package managers: Conan
This repository uses `Conan` as the package manager and build entry point. In practice, Conan is responsible for four things:

- resolving the C++ dependencies used by the library and tests;
- generating the CMake toolchain and dependency files used for configuration;
- carrying package options into CMake cache variables;
- defining the package version and build layout used in CI and local builds.

The options related to the package itself and its dependencies are defined in `conanfile.py`. Compiler, standard library, architecture, and build-type selections are carried through Conan profiles.

### conanfile.py
The recipe is the authoritative description of the package. Important pieces of the current recipe are:

- `version`: the package version consumed by both Conan and CMake.
- `package_type`: currently `static-library`.
- `requires` and `test_requires`: runtime and test dependencies such as Eigen, HDF5, SymEngine, spdlog, and Catch2.
- `options` and `default_options`: feature toggles passed through to CMake, including documentation, warnings, sanitizers, clang-tidy, cppcheck, coverage, profiling, and MKL selection.
- `validate()`: compiler and language-level constraints. The current package requires C++23, GCC 14 or newer, Clang 18 or newer, or MSVC 2022.
- `generate()`: generation of `CMakeDeps`, `CMakeToolchain`, preset files, and the MKL runtime environment.

For more details, check the `conanfile.py` file in the root directory and the [Conan documentation](https://docs.conan.io/2/reference/conanfile.html).

### Conan profiles
There are several profiles under `.conan/profiles` used by the GitHub workflows and local builds. The repository currently uses three profile groups:

1. Profiles starting with the `os` name
    For example, `windows-msvc2022-amd64`, `ubuntu-gcc14-amd64`, or `macos-clang-arm64`. These profiles select the target operating system, compiler, architecture, and C++ standard, and they set `[buildenv]` so CMake uses the intended compiler rather than silently falling back to another one.

2. Profiles starting with `build-`
    These profiles set the build type during the configuration and build stages. For example, `build-release` produces the `conan-release` preset and `build-debug` produces the `conan-debug` preset.

3. Profiles starting with `options-`
    These profiles hold repository-specific option sets such as `options-ci`, `options-doc`, `options-coverage`, or `options-lib-warns`.

When calling `conan install` or `conan create`, several profiles can be combined. The CI workflows do that explicitly so the compiler, build type, and option set remain readable and reproducible.

For local development, there are two reasonable approaches:

- reuse the repository profiles directly, following the same pattern as CI;
- generate and maintain a local `default` profile with `conan profile detect --force`, then add the package options you want to that profile.

Typical CI-style commands in this repository look like this:

```bash
conan config install .conan
conan install . --build=missing \
  -pr:h=ubuntu-gcc14-amd64 -pr:h=build-release -pr:h=options-doc \
  -pr:b=ubuntu-gcc14-amd64 -pr:b=build-release -pr:b=options-doc
cmake --preset=conan-release
```

For more information about profile settings and composition, see the [Conan profile documentation](https://docs.conan.io/2/reference/config_files/profiles.html).

### Current repository usage
In this repository, Conan is not an optional wrapper around CMake. It is the supported way to prepare dependencies, presets, and option sets for both CI and local builds.

That choice is independent of any specific binding library. If the C++ core is consumed from another C++ project or from Python bindings built with nanobind, the expectation remains the same: Conan prepares the dependency and toolchain layer, and CMake consumes the generated configuration.


