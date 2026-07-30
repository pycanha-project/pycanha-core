# Metal capability spike — running it on a Mac

This target answers one question: **can this Mac run the pycanha-core radiative
engine?** It is the M0 step B/C gate from `roadmap/21-metal-spike-report.md`, and
it exists because nothing about Metal execution can be tested in CI — GitHub's
macOS runners are virtual machines exposing an *Apple Paravirtual* GPU (family
Apple5) with no ray-tracing support and no 64-bit atomics.

It links only Metal and Foundation, not `pycanha-core`, so a failure here is a
statement about the machine and the shader toolchain rather than about the
library. It is **transitional**: delete it once the Metal backend (`mtl_device`
/ `mtl_scene` / `mtl_accum`) lands and the real radiative tests cover this
ground.

**Requires Apple Silicon with GPU family Apple9 — M3, M4 or newer.** That is
where Metal gained the full set of 64-bit buffer atomics the fixed-point energy
deposits rely on (Apple8/M2 has 64-bit min/max only). On an M1 or M2 the spike
correctly reports gate 1 as FAIL.

## What it checks

| Gate | What it proves |
|---|---|
| 1 | A device reports both `supportsRaytracing` and `supportsFamily:Apple9` |
| 2 | The three production kernels (`vf`, `exchange`, `solar`) survive Slang → MSL → `.metallib` → `MTLComputePipelineState`. Pipeline creation is the real check: it is where Metal validates the entry point and its buffer bindings |
| 3 | The runtime executes inline ray tracing (hit/miss + primitive index) and 64-bit atomic adds correctly, via the self-contained `probe.slang` kernel |

Exit code is 0 only if every gate passes.

## Setting up a store-fresh Mac mini M4

Four things are needed. Nothing else.

**1. Xcode** (not just the Command Line Tools — the Metal compiler ships only
with Xcode). Install from the App Store, then point the toolchain at it and
accept the licence:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept
```

**2. The Metal toolchain.** Since Xcode 26 the Metal compiler is an *unbundled*
component that is **not** installed with Xcode:

```bash
xcodebuild -downloadComponent MetalToolchain
```

It installs **per-user**, so run it as the account that will build. Verify:

```bash
xcrun metal --version
```

If that fails, the CMake configure step stops with the same advice — the build
will not get halfway through and then die.

**3. A recent Python** (for Conan and CMake, which live in the repo's own
virtualenv). The `python3` that comes with the Command Line Tools is old; use
Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.13
```

**4. The repo virtualenv.** Same convention as the Windows and WSL copies — the
`conan` and `cmake` used for the build MUST be the ones in this repo's `.venv`,
never a global install:

```bash
cd <path to>/pycanha-core
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install conan cmake
conan config install .conan          # installs the repo's conan profiles
```

Not needed on macOS: MKL (the build passes `PYCANHA_OPTION_USE_MKL=False`), the
Vulkan SDK, and any Slang install — `cmake/Slang.cmake` fetches the pinned
`slangc` release itself and verifies its SHA256.

## Building and running

```bash
cd <path to>/pycanha-core
source .venv/bin/activate

conan install . --build=missing \
  -pr:h=macos-clang21-arm64 -pr:h=build-release -pr:h=options-ci \
  -pr:b=macos-clang21-arm64 -pr:b=build-release -pr:b=options-ci \
  -o PYCANHA_OPTION_USE_MKL=False \
  -o PYCANHA_OPTION_METAL_SPIKE=True

cmake --preset=conan-release
cmake --build build/Release --target metal_spike -j"$(sysctl -n hw.ncpu)"
./build/Release/test/metal_spike/metal_spike
```

The first `conan install` builds several dependencies from source (symengine,
hdf5, manifold) and takes a while; later runs are cached. `PYCANHA_OPTION_RAYTRACING`
stays forced off on macOS, so the library itself still compiles the stub — the
spike deliberately does not depend on it.

Expected output on an M4:

```
Gate 1 — device capability
  [PASS] an Apple9 device with ray tracing — Apple M4
Gate 2 — production kernels through the toolchain
  [PASS] vf: metallib -> pipeline state — max threads/threadgroup = ...
  [PASS] exchange: metallib -> pipeline state — ...
  [PASS] solar: metallib -> pipeline state — ...
Gate 3 — runtime ray tracing and 64-bit atomics
  [PASS] acceleration structure build — BLAS ... B, TLAS ... B
  [PASS] inline ray tracing (hit/miss + primitive index) — 128 rays correct
  [PASS] 64-bit atomic add — got 549755813888, expected 549755813888

RESULT: PASS
```

## Interpreting a failure

- **Gate 1 FAIL** — the Mac is M1/M2, or the GPU is virtualised. Not a code bug.
- **`xcrun metal` errors during the build** — the Metal toolchain component is
  missing or was installed under a different user account (see step 2).
- **Gate 2 FAIL on a specific kernel** — the Metal front end rejected something
  slangc emitted. The most likely culprit is the `(packed_float3 device*)`
  casts from raw device addresses in `kernels/common.slang`; that is gate G3 in
  the spike report, and its fallback (replacing the `uint64_t *_addr` fields
  with element offsets into scene-wide buffers) is described there. Keep the
  `.metal` and `.air` intermediates from `build/Release/test/metal_spike/kernels/`
  — they are the evidence.
- **Gate 3 ray tracing FAIL, atomics PASS** — suspect the `useResource:` call in
  `main.mm`. A TLAS does not make the BLASes it references resident, and Metal
  reports misses rather than faulting, which is exactly the silent-wrong-results
  failure mode the real backend has to guard against.
- **Gate 3 atomics FAIL** — the 64-bit atomic add is not behaving. Check that
  gate 1 really reported `apple9=yes`; a partial sum that is a multiple of 2^32
  short points at lost adds rather than a 32-bit truncation.
