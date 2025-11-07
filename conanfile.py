from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps
from conan.tools.build import check_max_cppstd, check_min_cppstd, can_run
from conan.tools.files import copy
from conan.tools.env import VirtualBuildEnv
from conan.errors import ConanInvalidConfiguration
from conan.tools.scm import Version

import os


class Recipe_pycanha_core(ConanFile):
    # Package info
    name = "pycanha-core"  # Migh be passed to CMake if required

    # This is the version used everywhere. Right now is set manually,
    # but it could be set automatically from the git tag for example.
    version = "0.3"

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
    options = {
        "fPIC": [True, False],
        "PYCANHA_OPTION_LIBRARY": [True, False],
        "PYCANHA_OPTION_USE_MKL": [True, False],
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

        tc.generate()

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

        # Without adding the link flags, the sanitizers libraries are not linked (for the consumer).
        if self.options.PYCANHA_OPTION_SANITIZE_ADDR:
            self.cpp_info.sharedlinkflags.append("-fsanitize=address")
            self.cpp_info.exelinkflags.append("-fsanitize=address")
        if self.options.PYCANHA_OPTION_SANITIZE_UNDEF:
            self.cpp_info.sharedlinkflags.append("-fsanitize=undefined")
            self.cpp_info.exelinkflags.append("-fsanitize=undefined")
