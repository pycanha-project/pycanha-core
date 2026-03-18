from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps
from conan.tools.build import check_max_cppstd, check_min_cppstd, can_run
from conan.tools.files import copy
from conan.tools.env import VirtualBuildEnv, Environment
from conan.errors import ConanInvalidConfiguration
from conan.tools.scm import Version

import json
import os
import subprocess
import sys


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
            if option not in ("fPIC", "PYCANHA_OPTION_MKL_VERSION"):
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

        # Install MKL via pip and tell CMake where to find MKLConfig.cmake
        self._mkl_package_data = None
        if self.options.PYCANHA_OPTION_USE_MKL:
            mkl_data = self._install_and_find_mkl()
            if mkl_data:
                mkl_dir, mklroot = mkl_data[0], mkl_data[1]
                tc.cache_variables["MKL_DIR"] = mkl_dir
                tc.cache_variables["MKLROOT"] = mklroot
                self._mkl_package_data = mkl_data

        tc.generate()

        # Set MKL runtime environment so tests can find shared libraries
        if self._mkl_package_data:
            _, mklroot, _, lib_dir, bin_dir = self._mkl_package_data
            env = Environment()
            env.define("MKLROOT", mklroot)
            if self.settings.os == "Windows":
                env.prepend_path("PATH", bin_dir)
            else:
                env.prepend_path("LD_LIBRARY_PATH", lib_dir)

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

        # Export MKL paths and library names for consumers
        if self.options.PYCANHA_OPTION_USE_MKL:
            link_libs, system_libs = self._mkl_link_components()
            # On Windows, import libs (.lib) work in cpp_info.libs.
            # On Linux, pip ships versioned SONAMEs (.so.N); use system_libs to
            # avoid Conan validation errors while still passing -l flags.
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

            # Discover MKL directories from pip-installed packages
            mkl_data = None
            if hasattr(self, "_mkl_package_data") and self._mkl_package_data:
                mkl_data = self._mkl_package_data
            else:
                mkl_data = self._find_mkl_paths()

            if mkl_data:
                _, _, include_dir, lib_dir, bin_dir = mkl_data
                if (
                    os.path.isdir(include_dir)
                    and include_dir not in self.cpp_info.includedirs
                ):
                    self.cpp_info.includedirs.append(include_dir)
                if os.path.isdir(lib_dir) and lib_dir not in self.cpp_info.libdirs:
                    self.cpp_info.libdirs.append(lib_dir)
                if os.path.isdir(bin_dir) and bin_dir not in self.cpp_info.bindirs:
                    self.cpp_info.bindirs.append(bin_dir)

        # Without adding the link flags, the sanitizers libraries are not linked (for the consumer).
        compiler_name = str(self.settings.compiler)
        if compiler_name not in {"msvc", "intel-cc"}:
            if self.options.PYCANHA_OPTION_SANITIZE_ADDR:
                self.cpp_info.sharedlinkflags.append("-fsanitize=address")
                self.cpp_info.exelinkflags.append("-fsanitize=address")
            if self.options.PYCANHA_OPTION_SANITIZE_UNDEF:
                self.cpp_info.sharedlinkflags.append("-fsanitize=undefined")
                self.cpp_info.exelinkflags.append("-fsanitize=undefined")

    # ==========================================================================
    # MKL Installation and Discovery (pip-based)
    # ==========================================================================
    # MKL is installed via pip (mkl-devel) into the current Python environment.
    # We find MKLConfig.cmake via importlib.metadata in a subprocess (to pick
    # up freshly-installed packages) and derive all other paths from it.
    #
    # Layout of pip-installed MKL:
    #   Windows: <prefix>/Library/lib/cmake/mkl/MKLConfig.cmake
    #   Linux:   <prefix>/lib/cmake/mkl/MKLConfig.cmake
    #
    # From MKLConfig.cmake, going up 4 directories gives the MKL root:
    #   root/include  -> headers
    #   root/lib      -> libraries (.lib on Windows, .so on Linux)
    #   root/bin      -> DLLs (Windows only; on Linux bin_dir == lib_dir)
    # ==========================================================================

    _MKL_FIND_SCRIPT = """
import json
from importlib import metadata
from pathlib import Path

result = {}
for pkg in ("mkl-devel", "mkl", "mkl-include"):
    try:
        dist = metadata.distribution(pkg)
        for f in dist.files or []:
            p = dist.locate_file(f).resolve()
            if p.name == "MKLConfig.cmake":
                mkl_dir = p.parent
                root = mkl_dir.parent.parent.parent  # cmake/mkl -> cmake -> lib -> root
                result["mkl_dir"] = str(mkl_dir)
                result["root"] = str(root)
                result["include_dir"] = str(root / "include")
                result["lib_dir"] = str(root / "lib")
                bin_candidate = root / "bin"
                result["bin_dir"] = str(bin_candidate) if bin_candidate.is_dir() else str(root / "lib")
                break
    except Exception:
        continue
    if result:
        break

print(json.dumps(result))
"""

    def _find_mkl_paths(self):
        """Discover MKL paths from pip-installed packages via a subprocess."""
        env_reset = Environment()
        env_reset.unset("PYTHONPATH")
        env_reset.unset("PYTHONHOME")

        with env_reset.vars(self).apply():
            result = subprocess.run(
                [sys.executable, "-c", self._MKL_FIND_SCRIPT],
                capture_output=True,
                text=True,
                check=False,
            )

        if result.returncode != 0 or not result.stdout.strip():
            return None

        data = json.loads(result.stdout.strip())
        if not data.get("mkl_dir"):
            return None

        return (
            data["mkl_dir"],
            data["root"],
            data["include_dir"],
            data["lib_dir"],
            data["bin_dir"],
        )

    def _install_and_find_mkl(self):
        """Install mkl-devel via pip and return discovered MKL paths."""
        if not bool(self.options.get_safe("PYCANHA_OPTION_USE_MKL", False)):
            return None

        env_reset = Environment()
        env_reset.unset("PYTHONPATH")
        env_reset.unset("PYTHONHOME")

        version = str(self.MKL_PIP_VERSION)
        with env_reset.vars(self).apply():
            self.output.info(f"Installing mkl-devel=={version}")
            self.run(f'"{sys.executable}" -m pip install "mkl-devel=={version}"')

        mkl_data = self._find_mkl_paths()
        if not mkl_data:
            raise ConanInvalidConfiguration(
                f"MKL directories not found after installing mkl-devel=={version}"
            )
        return mkl_data

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
