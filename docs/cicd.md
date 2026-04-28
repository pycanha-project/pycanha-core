# CI/CD

The repository currently uses five GitHub Actions workflows under `.github/workflows`. They fall into two groups: continuous validation on `main` and pull requests, and documentation publication on release.

## Continuous validation on `main` and pull requests

These workflows verify the repository state. They do not rewrite source files or publish artifacts.

### Code Checks

Defined in `code-checks.yml`.

This workflow runs repository-wide style and static-analysis checks:

- `cpplint`
- `clang-format`
- `clang-tidy`
- `cppcheck`

The static-analysis job configures the project through Conan and CMake before invoking the analysis targets.

### CI

Defined in `ci.yml`.

This is the main build-and-test matrix. It currently runs `conan create` for both `Debug` and `Release` with the following host configurations:

- Ubuntu latest with GCC 14 and MKL enabled
- Windows latest with MSVC 2022 and MKL enabled
- macOS latest with Clang and MKL disabled

The purpose of this workflow is to validate that the static library, its tests, and its Conan packaging path remain consistent across the supported configurations.

### Coverage

Defined in `coverage.yml`.

This workflow performs a debug build on Ubuntu with GCC 14, enables the coverage configuration, builds the `coverage` target, and uploads the resulting report to Codecov.

### Documentation

Defined in `docs.yml`.

This workflow validates that the documentation can be configured and built on Ubuntu. It installs the documentation toolchain, configures the project through Conan with the documentation options profile, and builds the `doc` target.

## Release-time publication

### Deploy

Defined in `deploy.yml`.

This workflow runs only when a GitHub release is published. It rebuilds the documentation and deploys the generated HTML to GitHub Pages.

The repository now builds a static library rather than a header-only package. The current deploy workflow is documentation-only: it publishes the docs site, but it does not publish separate binary artifacts.
