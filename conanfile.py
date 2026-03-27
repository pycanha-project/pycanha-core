import os
import subprocess
import sys
from pathlib import Path

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import can_run, check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.env import Environment, VirtualBuildEnv
from conan.tools.files import copy
from conan.tools.scm import Version


class Recipe_pycanha_core(ConanFile):
    # Package info
    name = "pycanha-core"  # Migh be passed to CMake if required

    # This is the version used everywhere. Right now is set manually,
    # but it could be set automatically from the git tag for example.
    version = "0.7"

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
    # Single source of truth for pinned versions used by this release.
    # Includes both Conan dependencies and tooling hints used by CMake/docs.
    DEPENDENCY_VERSIONS = {
        "eigen": "5.0.1",
        "cdt": "1.4.4",
        "mkl": "2025.3.1",
        "catch2": "3.13.0",
        "doxygen": "1.9.4",  # Tested version, but this is just a hint for CMake
        "doxygen_awesome_css": "v2.2.0",
    }
    options = {
        "fPIC": [True, False],
        "PYCANHA_OPTION_LIBRARY": [True, False],
        "PYCANHA_OPTION_USE_MKL": [True, False],
        "PYCANHA_OPTION_MKL_VERSION": ["ANY"],
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
        "PYCANHA_OPTION_MKL_VERSION": "default",
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
        versions = self.DEPENDENCY_VERSIONS

        # Library dependencies
        self.requires(f"eigen/{versions['eigen']}", transitive_headers=True)
        # transitive_headers=True is used when the dependencies of the library are headers needed by the consumer.

        # self.requires(f"cdt/{versions['cdt']}")
        # CDT is currently fetched in CMake with FetchContent. We still keep
        # its version centralized here and pass it to CMake in generate().

        # Test dependencies
        self.test_requires(f"catch2/{versions['catch2']}")

        default_version = versions["mkl"]
        opt_version = self.options.get_safe("PYCANHA_OPTION_MKL_VERSION")
        if opt_version and str(opt_version).lower() not in {"", "default", "auto"}:
            self.MKL_PIP_VERSION = str(opt_version)
        else:
            self.MKL_PIP_VERSION = default_version

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
            if Version(str(self.settings.compiler.version)) < "14":
                raise ConanInvalidConfiguration("GCC >= 14 required for C++23 support")
        elif self.settings.compiler == "clang":
            if Version(str(self.settings.compiler.version)) < "18":
                raise ConanInvalidConfiguration(
                    "Clang >= 18 required for C++23 support"
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

        # Export centrally managed version hints for CMake and docs tooling.
        tc.cache_variables["PYCANHA_OPTION_CDT_VERSION"] = self.DEPENDENCY_VERSIONS[
            "cdt"
        ]
        tc.cache_variables["PYCANHA_OPTION_DOXYGEN_VERSION"] = self.DEPENDENCY_VERSIONS[
            "doxygen"
        ]
        tc.cache_variables["PYCANHA_OPTION_DOXYGEN_AWESOME_CSS_VERSION"] = (
            self.DEPENDENCY_VERSIONS["doxygen_awesome_css"]
        )

        # Enable compile commands export for Debug builds (useful for IDE integration)
        if self.settings.build_type == "Debug":
            tc.cache_variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"

        # Configure MKL from pip
        self._mkl_package_data = None
        mkl_data = self._install_mkl()

        if mkl_data:
            mklroot, include_dir, lib_dir, bin_dir = mkl_data
            self._mkl_package_data = mkl_data
            # Help CMake find MKL (never use MKLROOT — it leaks to system oneAPI)
            tc.cache_variables["MKL_ROOT"] = str(mklroot)
            tc.cache_variables["MKL_DIR"] = str(lib_dir / "cmake" / "mkl")

        tc.generate()

        # Export env for MKL runtime (so tests can find dll/so)
        if mkl_data:
            mklroot, include_dir, lib_dir, bin_dir = mkl_data
            env = Environment()
            if self.settings.os == "Windows":
                env.prepend_path("PATH", str(bin_dir))
            else:
                env.prepend_path("LD_LIBRARY_PATH", str(lib_dir))

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
                cmake.test(cli_args=["--verbose"])

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

            # Discover MKL paths (from cache during build, or rediscover for consumers)
            mkl_data = None
            if hasattr(self, "_mkl_package_data") and self._mkl_package_data:
                mkl_data = self._mkl_package_data
            else:
                mkl_data = self._find_mkl_paths()

            if mkl_data:
                mklroot, include_dir, lib_dir, bin_dir = mkl_data

                self.cpp_info.includedirs.append(str(include_dir))
                self.cpp_info.libdirs.append(str(lib_dir))
                if str(bin_dir) != str(lib_dir):
                    self.cpp_info.bindirs.append(str(bin_dir))

                # Add MKL library names so consumers can link
                self._add_mkl_link_libs(lib_dir)

                # Propagate runtime/build environment to consumers
                self.buildenv_info.define("MKL_DIR", str(lib_dir / "cmake" / "mkl"))
                if self.settings.os == "Windows":
                    self.runenv_info.prepend_path("PATH", str(bin_dir))
                    self.buildenv_info.prepend_path("PATH", str(bin_dir))
                else:
                    self.runenv_info.prepend_path("LD_LIBRARY_PATH", str(lib_dir))
                    self.buildenv_info.prepend_path("LD_LIBRARY_PATH", str(lib_dir))
        else:
            self.cpp_info.defines.append("PYCANHA_USE_MKL=0")

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
    # Strategy: pip install mkl-devel, then find MKLConfig.cmake via
    # importlib.metadata in a subprocess (avoids metadata cache issues).
    # From MKLConfig.cmake location, derive all paths:
    #   <mklroot>/lib/cmake/mkl/MKLConfig.cmake
    # ============================================================================

    def _find_mkl_paths(self):
        """Find MKL paths by locating MKLConfig.cmake via pip package metadata.

        Returns (mklroot, include_dir, lib_dir, bin_dir) or None.
        """
        env_reset = Environment()
        env_reset.unset("PYTHONPATH")
        env_reset.unset("PYTHONHOME")

        script = (
            "import importlib.metadata, sys\n"
            "for pkg in ('mkl-devel', 'mkl'):\n"
            "    try:\n"
            "        dist = importlib.metadata.distribution(pkg)\n"
            "        for f in dist.files or []:\n"
            "            r = dist.locate_file(f).resolve()\n"
            "            if r.name == 'MKLConfig.cmake':\n"
            "                print(r); sys.exit(0)\n"
            "    except importlib.metadata.PackageNotFoundError: pass\n"
            "sys.exit(1)\n"
        )

        with env_reset.vars(self).apply():
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=False,
            )

        if result.returncode != 0:
            self.output.warning("Could not find MKLConfig.cmake via pip metadata")
            return None

        config_path = Path(result.stdout.strip())
        # Layout: <mklroot>/lib/cmake/mkl/MKLConfig.cmake
        lib_dir = config_path.parent.parent.parent  # mkl/ -> cmake/ -> lib/
        mklroot = lib_dir.parent
        include_dir = mklroot / "include"
        # On Windows, DLLs are in <mklroot>/bin; on Linux, .so are in lib/
        bin_dir = (mklroot / "bin") if self.settings.os == "Windows" else lib_dir

        self.output.info(f"MKL found: root={mklroot}, lib={lib_dir}")

        # Debug: log what MKL library files exist
        for name in ["mkl_intel_lp64", "mkl_intel_thread", "mkl_core", "iomp5"]:
            files = sorted(lib_dir.glob(f"lib{name}*"))
            files += sorted(lib_dir.glob(f"{name}*"))  # Windows: no lib prefix
            if files:
                self.output.info(f"  {name}: {sorted(set(f.name for f in files))}")
            else:
                self.output.warning(f"  {name}: NOT FOUND in {lib_dir}")

        # On Linux, pip MKL may only ship versioned .so.2 files without
        # the unversioned .so symlinks that the linker needs for -l flags.
        # Create them if missing.
        self._ensure_mkl_dev_symlinks(lib_dir)

        return mklroot, include_dir, lib_dir, bin_dir

    def _install_mkl(self):
        """Install mkl-devel via pip and return MKL paths."""
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
                f"MKL not found after installing mkl-devel=={version}"
            )
        return mkl_data

    def _add_mkl_link_libs(self, lib_dir):
        """Add MKL libraries to cpp_info for consumer linking.

        On Windows, uses system_libs (always works with .lib files).
        On Linux, tries system_libs (-l flags) if .so symlinks exist,
        otherwise falls back to full file paths as linker flags.
        """
        if self.settings.os == "Windows":
            self.cpp_info.system_libs.extend(self._mkl_system_libs())
            return

        # Linux/macOS: check each MKL lib individually
        mkl_names = ["mkl_intel_lp64", "mkl_intel_thread", "mkl_core", "iomp5"]
        for name in mkl_names:
            so_file = lib_dir / f"lib{name}.so"
            if so_file.exists() or so_file.is_symlink():
                self.cpp_info.system_libs.append(name)
            else:
                # No unversioned .so: link by full path to versioned file
                versioned = sorted(lib_dir.glob(f"lib{name}.so.*"))
                if versioned:
                    self.output.info(f"Using full path for {name}: {versioned[-1]}")
                    self.cpp_info.exelinkflags.append(str(versioned[-1]))
                    self.cpp_info.sharedlinkflags.append(str(versioned[-1]))
                else:
                    self.output.warning(f"MKL lib not found: {name}")

        if self.settings.os == "Linux":
            self.cpp_info.system_libs.extend(["pthread", "m", "dl"])

    def _ensure_mkl_dev_symlinks(self, lib_dir):
        """Create unversioned .so symlinks if missing.

        pip MKL packages may only ship versioned .so.2 files. The linker
        needs unversioned .so symlinks to resolve -l flags.
        """
        if self.settings.os == "Windows":
            return
        for name in ["mkl_intel_lp64", "mkl_intel_thread", "mkl_core", "iomp5"]:
            so_link = lib_dir / f"lib{name}.so"
            if so_link.exists() or so_link.is_symlink():
                continue
            versioned = sorted(lib_dir.glob(f"lib{name}.so.*"))
            if versioned:
                try:
                    so_link.symlink_to(versioned[-1].name)
                    self.output.info(
                        f"Created symlink: {so_link.name} -> {versioned[-1].name}"
                    )
                except OSError as e:
                    self.output.warning(f"Cannot create {so_link.name} symlink: {e}")

    def _mkl_system_libs(self):
        """Return MKL library names for linking (dynamic-link only)."""
        # If static MKL support is needed again, restore the removed
        # PYCANHA_OPTION_MKL_LINK option and add the static library names here.
        if self.settings.os == "Windows":
            return [
                "mkl_intel_lp64_dll",
                "mkl_intel_thread_dll",
                "mkl_core_dll",
                "libiomp5md",
            ]
        else:
            # Linux/macOS
            libs = ["mkl_intel_lp64", "mkl_intel_thread", "mkl_core", "iomp5"]
            if self.settings.os == "Linux":
                libs.extend(["pthread", "m", "dl"])
            return libs
