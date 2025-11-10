from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps
from conan.tools.build import check_max_cppstd, check_min_cppstd, can_run
from conan.tools.files import copy
from conan.tools.env import VirtualBuildEnv, Environment
from conan.errors import ConanInvalidConfiguration
from conan.tools.scm import Version

import os
import sys
import subprocess
from pathlib import Path
from importlib import metadata


class Recipe_pycanha_core(ConanFile):
    # Package info
    name = "pycanha-core"  # Migh be passed to CMake if required

    # This is the version used everywhere. Right now is set manually,
    # but it could be set automatically from the git tag for example.
    version = "0.5"

    # I've followed the instructions from https://docs.conan.io/2/tutorial/creating_packages/other_types_of_packages/header_only_packages.html
    # but without adding the "header-only" keyword to the recipe, it doesn't work. The use of the "header-only" is from here:
    # https://github.com/hobbeshunter/json2cpp/tree/conan-2-export-tool-and-lib
    # package_type = "header-library"

    # Now we are using a static library, because we build some functions separately.
    package_type = "static-library"

    # Optional metadata
    license = "MIT"
    author = "Javier Piqueras"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "C++ template library"
    topics = ("C++", "Template", "conan", "doxygen")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    DEFAULT_MKL_PIP_VERSION = "2025.3.0"
    options = {
        "fPIC": [True, False],
        "PYCANHA_OPTION_LIBRARY": [True, False],
        "PYCANHA_OPTION_USE_MKL": [True, False],
        "PYCANHA_OPTION_MKL_VERSION": ["ANY"],
        "PYCANHA_OPTION_MKL_LINK": ["static", "dynamic"],
        "PYCANHA_OPTION_LTO": [True, False],
        "PYCANHA_OPTION_DOCS": [True, False],
        "PYCANHA_OPTION_WARNINGS": [True, False],
        "PYCANHA_OPTION_WARNINGS_AS_ERRORS": [True, False],
        "PYCANHA_OPTION_COVERAGE": [True, False],
        "PYCANHA_OPTION_INCLUDE_WHAT_YOU_USE": [True, False],
        "PYCANHA_OPTION_CLANG_TIDY": [True, False],
        "PYCANHA_OPTION_CPPCHECK": [True, False],
        "PYCANHA_OPTION_SANITIZE_ADDR": [True, False],
        "PYCANHA_OPTION_SANITIZE_UNDEF": [True, False],
    }

    default_options = {
        "fPIC": True,
        "PYCANHA_OPTION_LIBRARY": True,
        "PYCANHA_OPTION_USE_MKL": True,
        "PYCANHA_OPTION_MKL_VERSION": DEFAULT_MKL_PIP_VERSION,
        "PYCANHA_OPTION_MKL_LINK": "dynamic",
        "PYCANHA_OPTION_LTO": True,
        "PYCANHA_OPTION_DOCS": False,
        "PYCANHA_OPTION_WARNINGS": False,
        "PYCANHA_OPTION_WARNINGS_AS_ERRORS": False,
        "PYCANHA_OPTION_COVERAGE": False,
        "PYCANHA_OPTION_INCLUDE_WHAT_YOU_USE": False,
        "PYCANHA_OPTION_CLANG_TIDY": False,
        "PYCANHA_OPTION_CPPCHECK": False,
        "PYCANHA_OPTION_SANITIZE_ADDR": False,
        "PYCANHA_OPTION_SANITIZE_UNDEF": False,
    }

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports = "LICENSE", "README.md"
    exports_sources = "CMakeLists.txt", "pycanha-core/*", "cmake/*", "test/*"
    # no_copy_source = True # Not needed

    def requirements(self):
        # Dependencies can be defined also with a version range.
        # For now, hard-coding an specific version, so we know exactly what version is used in the build.

        # Library dependencies
        self.requires("eigen/3.4.0", transitive_headers=True)
        # transitive_headers=True is used when the dependencies of the library are headers needed by the consumer.

        # self.requires("cdt/1.3.0") # This is a header-only library. No need for complicated build. Can be fetched by CMake directly.

        # Test dependencies
        self.test_requires("catch2/3.3.2")

        default_version = self.DEFAULT_MKL_PIP_VERSION
        opt_version = self.options.get_safe("PYCANHA_OPTION_MKL_VERSION")
        self.MKL_PIP_VERSION = str(opt_version) if opt_version else default_version

        # Conditional dependencies. Depending on the option selected
        if self.options.PYCANHA_OPTION_DOCS:
            # This mean build doxygen!! It takes to long. Install it with other tool.
            # self.requires("doxygen/1.9.4")
            pass

    def validate(self):
        # Raise an error for non-supported configurations.
        check_min_cppstd(self, "23")

        # Check compiler support for C++23
        if self.settings.compiler == "gcc":
            if Version(str(self.settings.compiler.version)) < "11":
                raise ConanInvalidConfiguration("GCC >= 11 required for C++23 support")
        elif self.settings.compiler == "clang":
            if Version(str(self.settings.compiler.version)) < "12":
                raise ConanInvalidConfiguration(
                    "Clang >= 12 required for C++23 support"
                )
        elif self.settings.compiler == "msvc":
            if Version(str(self.settings.compiler.version)) < "193":
                raise ConanInvalidConfiguration(
                    "MSVC >= 193 (Visual Studio 2022) required for C++23 support"
                )

        if self.options.PYCANHA_OPTION_INCLUDE_WHAT_YOU_USE:
            raise ConanInvalidConfiguration(
                "IWYU (Include what you use) is broken right now. Set to OFF."
            )

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        # For static libraries, propagate fPIC to dependencies
        if self.package_type == "static-library":
            # Only propagate fPIC if it exists (it's removed on Windows)
            if self.options.get_safe("fPIC") is not None:
                self.options["*"].fPIC = self.options.fPIC

    def layout(self):
        # Here, it is defined where the build files are placed.
        # Instead of defining it manually, we can use a predefined layout by conan for cmake.
        # See: https://docs.conan.io/2/tutorial/consuming_packages/the_flexibility_of_conanfile_py.html#use-the-layout-method
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)

        # Variables passed to CMake. Conan has "variables" and "cache_variables".
        # See: https://docs.conan.io/2/reference/tools/cmake/cmaketoolchain.html#cache-variables
        # Pass the options to CMake (skip fPIC as it's handled by CMakeToolchain)
        for option, val in self.options.items():
            if option != "fPIC":
                tc.cache_variables[option] = val

        # Not sure if needed, this will pass -DCMAKE_BUILD_TYPE=<build_type> to CMake
        # Multi-config generators (like Visual Studio or Ninja Multi-Config) set the build type at build time, not at generation time.
        # But just in case, we set it here also. And the build_type is controlled by conan (through the profile or the command line)
        build_type = self.settings.get_safe("build_type", default="Release")
        tc.cache_variables["CMAKE_BUILD_TYPE"] = build_type
        tc.cache_variables["CONAN_PROJECT_VERSION"] = self.version

        # Enable compile commands export for Debug builds (useful for IDE integration)
        if self.settings.build_type == "Debug":
            tc.cache_variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"

        # Configure MKL from pip-based venv (strict, single approach)
        mkl_data = None
        self._mkl_package_data = (
            None  # Cache MKL discovery so package_info can reuse locations
        )
        try:
            mkl_data = self._mkl_paths_from_pip_venv()
        except ConanInvalidConfiguration as e:
            # If MKL is enabled, we want to fail early (before CMake configure)
            raise

        if mkl_data:
            mklroot, include, libdir, bindir = mkl_data

            # Help CMake find MKL if your CMakeLists uses MKLROOT
            tc.cache_variables["MKLROOT"] = str(mklroot)
            tc.cache_variables["MKL_ROOT"] = str(mklroot)
            self._mkl_package_data = (
                mkl_data  # Store full MKL paths for installation metadata
            )

        tc.generate()

        # Export env for MKL runtime (so tests can find dll/so)
        if mkl_data:
            mklroot, include, libdir, bindir = mkl_data
            env = Environment()
            env.define("MKLROOT", str(mklroot))
            if self.settings.os == "Windows":
                env.prepend_path("PATH", str(bindir))
            else:
                env.prepend_path("LD_LIBRARY_PATH", str(bindir))

            build_env = env.vars(self, scope="build")
            build_env.save_script("mkl_build_env")

            run_env = env.vars(self, scope="run")
            run_env.save_script("mkl_run_env")

        # With this, we tell conan to export the environment variables set in the profiles (eg: linux-gcc12-....)
        # In those variables we force CMake to use the specific version of the compiler, otherwise it chooses whatever he likes.
        ms = VirtualBuildEnv(self)
        ms.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

        # Running the tests (not the package_test, which is run separately after this using the test method)
        # See: https://docs.conan.io/2/tutorial/creating_packages/build_packages.html
        if can_run(self):
            if not self.conf.get("tools.build:skip_test", default=False):
                # Run tests with verbose output to see all test results
                self.output.info("Running unit tests...")
                # --verbose shows test execution output from ctest
                # Uncomment to see the tests, otherwise only in case of failure info is shown
                # cmake.test(cli_args=["--verbose"])

    def package(self):
        # Manual file copying approach (until CMake install rules are defined)
        # Copy headers
        copy(
            self,
            pattern="*.hpp",
            src=os.path.join(self.source_folder, "pycanha-core/include"),
            dst=os.path.join(self.package_folder, "include"),
        )

        # Copy static library files (.lib for MSVC, .a for GCC/Clang)
        copy(
            self,
            pattern="*.lib",
            src=self.build_folder,
            dst=os.path.join(self.package_folder, "lib"),
            keep_path=False,
        )
        copy(
            self,
            pattern="*.a",
            src=self.build_folder,
            dst=os.path.join(self.package_folder, "lib"),
            keep_path=False,
        )

        # Copy license files
        copy(
            self,
            pattern="LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
            keep_path=False,
        )

    def package_info(self):
        # Set CMake module/config file names and target names for better consumer experience
        self.cpp_info.set_property("cmake_file_name", "pycanha-core")
        self.cpp_info.set_property("cmake_target_name", "pycanha-core::pycanha-core")

        # This line is necessary because the headers are not in the "include" folder. By default, conan will look for the headers in the "include" folder
        # See: https://docs.conan.io/2/tutorial/creating_packages/define_package_information.html?highlight=also%20copy%20include%20folder#define-information-for-consumers-the-package-info-method
        # self.cpp_info.includedirs = ["pycanha-core/include"]
        # -- Now this is not necessary. Because I'm using CMake install() to copy pycanha-core/include to the package/include folder
        #    So now the headers are in the include folder, and conan will find them automatically.

        self.cpp_info.libs = ["pycanha-core"]

        if self.options.PYCANHA_OPTION_USE_MKL:
            self.cpp_info.defines.append("PYCANHA_USE_MKL=1")
            self.cpp_info.defines.append("EIGEN_USE_MKL_ALL")
        else:
            self.cpp_info.defines.append("PYCANHA_USE_MKL=0")

        # Surface MKL include/lib/bin locations so consumers can find headers and shared libs
        if self.options.PYCANHA_OPTION_USE_MKL:
            # Try to get from cached data first (when building), otherwise discover (when consuming)
            mkl_data = None
            if hasattr(self, "_mkl_package_data") and self._mkl_package_data:
                mkl_data = self._mkl_package_data
            else:
                # When consuming, discover MKL paths
                mkl_data = self._discover_mkl_paths()

            if mkl_data:
                _, include_dir, lib_dir, bin_dir = mkl_data

                include_path = str(include_dir)
                lib_path = str(lib_dir)
                bin_path = str(bin_dir)

                if include_path not in self.cpp_info.includedirs:
                    self.cpp_info.includedirs.append(include_path)
                if lib_path not in self.cpp_info.libdirs:
                    self.cpp_info.libdirs.append(lib_path)
                if bin_path not in self.cpp_info.bindirs:
                    self.cpp_info.bindirs.append(bin_path)

            # Provide MKL import/static libs so consumers resolve symbols at link time
            link_libs, system_libs = self._mkl_link_components()
            # On Windows, the MKL import libraries follow the expected naming
            # convention and CMakeDeps can resolve them via cpp_info.libs. On
            # Unix-like platforms the pip packages often ship only versioned
            # SONAMEs (libmkl_*.so.N), so we surface them as system libs instead
            # to avoid configure-time validation errors while still linking.
            link_target = (
                self.cpp_info.libs
                if self.settings.os == "Windows"
                else self.cpp_info.system_libs
            )
            for lib in link_libs:
                if lib not in link_target:
                    link_target.append(lib)
            for syslib in system_libs:
                if syslib not in self.cpp_info.system_libs:
                    self.cpp_info.system_libs.append(syslib)

        # Without adding the link flags, the sanitizers libraries are not linked (for the consumer).
        if self.options.PYCANHA_OPTION_SANITIZE_ADDR:
            self.cpp_info.sharedlinkflags.append("-fsanitize=address")
            self.cpp_info.exelinkflags.append("-fsanitize=address")
        if self.options.PYCANHA_OPTION_SANITIZE_UNDEF:
            self.cpp_info.sharedlinkflags.append("-fsanitize=undefined")
            self.cpp_info.exelinkflags.append("-fsanitize=undefined")

    # ============================================================================
    # MKL Installation and Discovery
    # ============================================================================
    # This section handles installing MKL from pip and discovering its paths.
    #
    # The challenge: After pip installs packages, Python's importlib.metadata
    # cache prevents the current process from seeing newly installed packages.
    #
    # The solution: Run a minimal subprocess that lists MKL package files in a
    # fresh Python context (with proper environment setup). Then process those
    # file paths in the main process to find include and lib directories.
    #
    # Environment setup: We unset PYTHONPATH and PYTHONHOME to ensure pip and
    # metadata discovery work with the correct virtual environment, not any
    # parent or system Python installation.
    #
    # Path discovery: We walk up from actual .h and library files to find their
    # parent 'include' or 'lib'/'bin' directories. This works cross-platform
    # without hardcoding paths (Windows uses Library/include and Library/bin,
    # Linux/Mac use include and lib).
    # ============================================================================

    def _walk_to_marker(self, start: Path, markers: set[str]) -> Path | None:
        """Walk up from `start` until a parent dir name is in `markers`."""
        d = start
        while d.parent != d:
            if d.name.lower() in markers:
                return d
            d = d.parent
        return None

    def _discover_mkl_paths(self):
        """Discover MKL paths from installed packages (without installing)."""
        # Setup environment to avoid conflicts
        env_reset = Environment()
        env_reset.unset("PYTHONPATH")
        env_reset.unset("PYTHONHOME")

        # Get file list from subprocess
        get_files_script = """
from importlib import metadata
for pkg in ("mkl-static", "mkl-devel", "mkl", "mkl-include"):
    try:
        dist = metadata.distribution(pkg)
        for f in dist.files or []:
            print(dist.locate_file(f).resolve())
    except: pass
"""

        with env_reset.vars(self).apply():
            result = subprocess.run(
                [sys.executable, "-c", get_files_script],
                capture_output=True,
                text=True,
                check=False,
            )

        if result.returncode != 0:
            return None

        # Process the files
        include_dir = lib_dir = bin_dir = None
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            p = Path(line.strip())
            suffix = p.suffix.lower()
            name_lower = p.name.lower()

            if not include_dir and suffix == ".h":
                include_dir = self._walk_to_marker(p.parent, {"include"})

            if self.settings.os == "Windows":
                if not lib_dir and suffix == ".lib" and "mkl" in name_lower:
                    lib_dir = self._walk_to_marker(p.parent, {"lib"})
                if not bin_dir and suffix == ".dll" and "mkl" in name_lower:
                    bin_dir = self._walk_to_marker(p.parent, {"bin"})
            else:
                if not lib_dir and suffix in (".so", ".dylib"):
                    lib_dir = self._walk_to_marker(p.parent, {"lib"})
                if not bin_dir and suffix == ".dll" and "mkl" in name_lower:
                    bin_dir = self._walk_to_marker(p.parent, {"bin"})

            if include_dir and lib_dir and (bin_dir or self.settings.os != "Windows"):
                break

        candidate_root = None
        if include_dir:
            candidate_root = include_dir.parent
        elif lib_dir:
            candidate_root = lib_dir.parent
        elif bin_dir:
            candidate_root = bin_dir.parent

        if not include_dir and candidate_root:
            maybe_include = candidate_root / "include"
            if maybe_include.is_dir():
                include_dir = maybe_include

        if not lib_dir and candidate_root:
            maybe_lib = candidate_root / "lib"
            if maybe_lib.is_dir():
                lib_dir = maybe_lib

        if not bin_dir:
            if self.settings.os == "Windows" and candidate_root:
                maybe_bin = candidate_root / "bin"
                if maybe_bin.is_dir():
                    bin_dir = maybe_bin
            elif lib_dir:
                bin_dir = lib_dir

        if not include_dir or not lib_dir:
            return None

        mklroot = include_dir.parent
        if bin_dir is None:
            bin_dir = lib_dir

        return mklroot, include_dir, lib_dir, bin_dir

    def _mkl_link_components(self) -> tuple[list[str], list[str]]:
        """Return MKL libraries and system libs required for the current platform."""
        link_preference = str(
            self.options.get_safe("PYCANHA_OPTION_MKL_LINK", "dynamic")
        ).lower()

        link_libs: list[str] = []
        system_libs: list[str] = []

        if self.settings.os == "Windows":
            if link_preference == "static":
                link_libs.extend(["mkl_intel_lp64", "mkl_intel_thread", "mkl_core"])
            else:
                link_libs.extend(
                    ["mkl_intel_lp64_dll", "mkl_intel_thread_dll", "mkl_core_dll"]
                )
            link_libs.append("libiomp5md")
        else:
            link_libs.extend(["mkl_intel_lp64", "mkl_intel_thread", "mkl_core"])
            # Intel ships iomp5 regardless of static/dynamic preference in pip packages
            link_libs.append("iomp5")

            if str(self.settings.os).lower() in {"linux", "freebsd"}:
                system_libs.extend(["pthread", "m", "dl"])

        return link_libs, system_libs

    def _mkl_paths_from_pip_venv(self):
        """Install mkl-devel and find MKL paths."""
        if not bool(self.options.get_safe("PYCANHA_OPTION_USE_MKL", False)):
            return None

        # Setup environment to avoid conflicts
        env_reset = Environment()
        env_reset.unset("PYTHONPATH")
        env_reset.unset("PYTHONHOME")

        # Install mkl-devel
        expected_version = str(self.MKL_PIP_VERSION)
        with env_reset.vars(self).apply():
            self.output.info(f"Installing mkl-devel=={expected_version}")
            self.run(
                f'"{sys.executable}" -m pip install "mkl-devel=={expected_version}"'
            )

        # Discover MKL paths
        mkl_data = self._discover_mkl_paths()
        if not mkl_data:
            raise ConanInvalidConfiguration(
                f"MKL directories not found after installing mkl-devel=={expected_version}"
            )

        return mkl_data
