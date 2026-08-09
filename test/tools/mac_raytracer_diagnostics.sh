#!/usr/bin/env bash
# =============================================================================
# pycanha-core 0.18 — macOS / Metal ray-tracing backend diagnostic run
# =============================================================================
#
# WHY THIS EXISTS
# ---------------
# The Metal backend (`mtl_device`, `mtl_scene`, `mtl_accum` + the Metal branch
# of the Slang kernels) has never been compiled or executed on real hardware.
# It was written and reviewed on a Windows/WSL machine; GitHub's macOS runners
# are Apple Virtualization Framework VMs whose "Apple Paravirtual device" is
# GPU family Apple5 — no ray tracing, no 64-bit atomics — so CI is COMPILE
# COVERAGE ONLY and every [gpu] test case skips there.
#
# The person who owns this code has no Mac. Every round trip to an Apple9 Mac
# (M3 / M4 / A17 Pro) is therefore expensive. This script exists to make ONE
# pass produce everything an engineer or an agent could need to diagnose a
# failure without asking a follow-up question:
#
#   * it never stops at the first failure — every stage runs, and each stage's
#     verdict is recorded independently;
#   * it isolates the risky checks into standalone probes that do NOT need the
#     repository to build, so a compile error in the library still yields a
#     full hardware capability report;
#   * every shader in those probes is compiled AT RUNTIME from a string, with
#     several fallback variants, so a wrong guess about Metal Shading Language
#     costs a logged error message instead of the whole run;
#   * it retries a failed build once with -Werror off, so a single Mac-only
#     warning cannot cost a whole trip;
#   * it captures generated intermediates (MSL, reflection JSON, .metallib,
#     binding headers) — the artifacts a fixing agent cannot regenerate
#     remotely;
#   * it self-checks determinism by running the fixed-seed dump twice and
#     diffing, which detects a race / missing barrier / residency mistake with
#     NO reference machine involved.
#
# THE TRAP THIS SCRIPT IS BUILT AROUND
# ------------------------------------
# Every [gpu] test in this repository begins with
#
#     if (!rad::is_available()) { SUCCEED("no RT-capable GPU device"); return; }
#
# so a fully green `ctest` on a machine the backend does not recognise proves
# NOTHING. A green run is only meaningful together with "an RT device was
# found and the GPU cases really executed". This script reports those as
# separate verdicts and marks the overall result INCONCLUSIVE, never PASS,
# when the GPU cases skipped.
#
# REQUIREMENTS
# ------------
#   * macOS 14+ on Apple Silicon; an Apple9 GPU (M3/M4/A17 Pro) for anything
#     beyond compile coverage.
#   * Xcode, plus the unbundled Metal toolchain:  xcodebuild -downloadComponent MetalToolchain
#   * A Python venv at the repo root with conan + cmake (see CLAUDE.md), or
#     conan + cmake anywhere on PATH.
#   * Network access on the first run: the build fetches a pinned slangc
#     release and the Conan dependencies.
#
# USAGE
# -----
#   test/tools/mac_raytracer_diagnostics.sh [options]
#
#   --probe-only          Stages 0-3 only (no Conan, no build). ~30 seconds;
#                         run this FIRST to confirm the machine is Apple9.
#   --skip-build          Reuse an existing build tree (see --build-dir).
#   --build-type TYPE     release (default) or debug.
#   --build-dir DIR       Default <repo>/build/Release (or /Debug).
#   --fast                Drop LTO + sanitizers (faster build, NOT CI parity).
#   --ci-parity           Additionally run the exact CI `conan create` command.
#   --reference-dir DIR   Directory holding bitdiff_vf.txt / bitdiff_exchange.txt
#                         produced by a Vulkan machine at the SAME commit.
#   --expect-tests N      Expected ctest count (Linux/Windows count + 1; the
#                         extra one is the Metal instance-transform test).
#   --python              Also probe an installed `pycanha_core` wheel.
#   --jobs N              Build parallelism (default: physical cores).
#   --no-conan-config     Skip `conan config install .conan`.
#   --out DIR             Output directory (default ./mac-rt-diag-<timestamp>).
#
# OUTPUT
# ------
# Everything lands in one directory plus a .tar.gz of it. Send the tarball
# back; it is self-describing. 00-SUMMARY.txt is the human entry point and
# summary.json the machine-readable one.
# =============================================================================

set -u

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
PROBE_ONLY=0
SKIP_BUILD=0
BUILD_TYPE="release"
BUILD_DIR=""
FAST=0
CI_PARITY=0
REFERENCE_DIR=""
EXPECT_TESTS=""
DO_PYTHON=0
JOBS=""
CONAN_CONFIG=1
OUT_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --probe-only)      PROBE_ONLY=1 ;;
        --skip-build)      SKIP_BUILD=1 ;;
        --build-type)      BUILD_TYPE="$2"; shift ;;
        --build-dir)       BUILD_DIR="$2"; shift ;;
        --fast)            FAST=1 ;;
        --ci-parity)       CI_PARITY=1 ;;
        --reference-dir)   REFERENCE_DIR="$2"; shift ;;
        --expect-tests)    EXPECT_TESTS="$2"; shift ;;
        --python)          DO_PYTHON=1 ;;
        --jobs)            JOBS="$2"; shift ;;
        --no-conan-config) CONAN_CONFIG=0 ;;
        --out)             OUT_DIR="$2"; shift ;;
        -h|--help)         sed -n '1,95p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

case "$BUILD_TYPE" in
    release) CMAKE_BUILD_TYPE="Release"; CONAN_PRESET="conan-release" ;;
    debug)   CMAKE_BUILD_TYPE="Debug";   CONAN_PRESET="conan-debug" ;;
    *) echo "--build-type must be release or debug" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Paths and output directory
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
[ -n "$OUT_DIR" ] || OUT_DIR="$PWD/mac-rt-diag-$STAMP"
mkdir -p "$OUT_DIR" || { echo "cannot create $OUT_DIR" >&2; exit 1; }
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
[ -n "$BUILD_DIR" ] || BUILD_DIR="$REPO_ROOT/build/$CMAKE_BUILD_TYPE"
[ -n "$JOBS" ] || JOBS="$(sysctl -n hw.physicalcpu 2>/dev/null || echo 4)"

WORK="$OUT_DIR/work"
KERNEL_DIR="$OUT_DIR/13-kernels"
mkdir -p "$WORK" "$KERNEL_DIR"

MAIN_LOG="$OUT_DIR/00-console.log"
FACTS="$WORK/facts.txt"          # key<TAB>value, one per line -> summary.json
: > "$FACTS"

RUN_START_EPOCH="$(date +%s)"
# Marker for "files created during this run". BSD find has no reliable
# -newermt "@epoch", but -newer <file> is portable.
touch "$WORK/.run-start"

# ---------------------------------------------------------------------------
# Helpers  (bash 3.2 — macOS ships no newer bash; no associative arrays,
#           no mapfile, no ${var^^})
# ---------------------------------------------------------------------------
log()      { echo "$*" | tee -a "$MAIN_LOG"; }
logfile()  { echo "$*" >> "$MAIN_LOG"; }

section() {
    log ""
    log "==============================================================================="
    log "== $*"
    log "==============================================================================="
}

fact() {  # fact <key> <value...>
    key="$1"; shift
    printf '%s\t%s\n' "$key" "$*" >> "$FACTS"
}

# run_logged <logfile> <label> <command...>
# Records the exact command, its output, wall time and exit code. Never aborts.
run_logged() {
    _lf="$1"; shift
    _label="$1"; shift
    {
        echo "### $_label"
        echo "### cwd: $PWD"
        echo "### cmd: $*"
        echo "### started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "-------------------------------------------------------------------------------"
    } >> "$_lf"
    _t0="$(date +%s)"
    "$@" >> "$_lf" 2>&1
    _rc=$?
    _t1="$(date +%s)"
    {
        echo "-------------------------------------------------------------------------------"
        echo "### exit code: $_rc   (${_t1}s - ${_t0}s = $((_t1 - _t0))s)"
        echo ""
    } >> "$_lf"
    log "    [$_label] exit=$_rc  $((_t1 - _t0))s  -> $(basename "$_lf")"
    return $_rc
}

# grep -c prints "0" AND exits 1 when nothing matches, so the usual
# `$(grep -c ... || echo 0)` yields the two-line string "0\n0" and every later
# numeric test on it is a syntax error. Always count through this.
count_matches() {  # count_matches <grep-args...> -- always prints one integer
    _n="$(grep -c "$@" 2>/dev/null)"
    case "$_n" in
        ''|*[!0-9]*) echo 0 ;;
        *) echo "$_n" ;;
    esac
}

# Same, but the command's stdout also goes to the console.
run_logged_tee() {
    _lf="$1"; shift
    _label="$1"; shift
    {
        echo "### $_label"
        echo "### cmd: $*"
        echo "-------------------------------------------------------------------------------"
    } >> "$_lf"
    "$@" 2>&1 | tee -a "$_lf"
    _rc=${PIPESTATUS[0]}
    echo "### exit code: $_rc" >> "$_lf"
    return $_rc
}

have() { command -v "$1" > /dev/null 2>&1; }

# Verdict bookkeeping. Values: PASS / FAIL / SKIP / WARN / UNKNOWN
VERDICT_KEYS=""
verdict() {  # verdict <key> <PASS|FAIL|SKIP|WARN|UNKNOWN> <one-line explanation>
    _k="$1"; _v="$2"; shift 2
    VERDICT_KEYS="$VERDICT_KEYS $_k"
    printf '%s\t%s\t%s\n' "$_k" "$_v" "$*" >> "$WORK/verdicts.txt"
    log "  ==> $_k: $_v — $*"
}
: > "$WORK/verdicts.txt"

verdict_of() {  # echoes the recorded value for a key, or UNKNOWN
    awk -F'\t' -v k="$1" '$1==k{print $2; found=1} END{if(!found) print "UNKNOWN"}' \
        "$WORK/verdicts.txt" | tail -1
}

json_escape() { sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/	/ /g'; }

log "pycanha-core macOS / Metal ray-tracing diagnostics"
log "  script     : $0"
log "  repo root  : $REPO_ROOT"
log "  output dir : $OUT_DIR"
log "  build type : $CMAKE_BUILD_TYPE"
log "  build dir  : $BUILD_DIR"
log "  started    : $(date '+%Y-%m-%d %H:%M:%S %z')"
fact run_started "$(date '+%Y-%m-%dT%H:%M:%S%z')"
fact build_type "$CMAKE_BUILD_TYPE"
fact probe_only "$PROBE_ONLY"

# ===========================================================================
section "STAGE 0 — machine, toolchain and repository state"
# ===========================================================================
ENV_LOG="$OUT_DIR/01-environment.txt"
: > "$ENV_LOG"

{
    echo "=== date ==="; date '+%Y-%m-%d %H:%M:%S %z'
    echo; echo "=== sw_vers ==="; sw_vers 2>&1
    echo; echo "=== uname -a ==="; uname -a 2>&1
    echo; echo "=== hardware ==="
    for k in hw.model hw.machine machdep.cpu.brand_string hw.physicalcpu \
             hw.logicalcpu hw.memsize hw.perflevel0.physicalcpu \
             hw.perflevel1.physicalcpu; do
        printf '%-34s %s\n' "$k" "$(sysctl -n "$k" 2>/dev/null || echo '<n/a>')"
    done
    echo; echo "=== GPU (system_profiler SPDisplaysDataType) ==="
    system_profiler SPDisplaysDataType 2>&1 | head -60
    echo; echo "=== Xcode / SDK ==="
    echo "xcode-select -p        : $(xcode-select -p 2>&1)"
    echo "xcodebuild -version    : $(xcodebuild -version 2>&1 | tr '\n' ' ')"
    echo "xcrun --show-sdk-path  : $(xcrun --show-sdk-path 2>&1)"
    echo "xcrun --show-sdk-version: $(xcrun --show-sdk-version 2>&1)"
    echo "clang --version        : $(clang --version 2>&1 | head -1)"
    echo "clang++ resource dir   : $(clang++ -print-resource-dir 2>&1)"
    echo; echo "=== Metal shader compiler ==="
    echo "xcrun metal --version:"
    xcrun metal --version 2>&1
    echo "xcrun metallib --version:"
    xcrun metallib --version 2>&1 | head -5
    echo "installed Metal toolchain components:"
    xcodebuild -showComponent MetalToolchain 2>&1 | head -20
    echo; echo "=== python / conan / cmake ==="
    echo "which python3 : $(command -v python3 2>&1)"
    echo "python3 -V    : $(python3 -V 2>&1)"
    echo "which conan   : $(command -v conan 2>&1)"
    echo "conan --version: $(conan --version 2>&1)"
    echo "which cmake   : $(command -v cmake 2>&1)"
    echo "cmake --version: $(cmake --version 2>&1 | head -1)"
    echo "which ninja   : $(command -v ninja 2>&1)"
    echo "VIRTUAL_ENV   : ${VIRTUAL_ENV:-<unset>}"
    echo; echo "=== repository ==="
    echo "root: $REPO_ROOT"
    ( cd "$REPO_ROOT" && \
      echo "branch  : $(git rev-parse --abbrev-ref HEAD 2>&1)" && \
      echo "commit  : $(git rev-parse HEAD 2>&1)" && \
      echo "describe: $(git describe --tags --always --dirty 2>&1)" && \
      echo "--- git status --short ---" && git status --short 2>&1 | head -50 && \
      echo "--- git log -5 --oneline ---" && git log -5 --oneline 2>&1 )
    echo "conanfile version: $(grep -m1 '^    version' "$REPO_ROOT/conanfile.py" 2>&1)"
    echo; echo "=== environment variables of interest ==="
    env | grep -Ei '^(CC|CXX|SDKROOT|MACOSX|MTL_|ASAN|UBSAN|CONAN|CMAKE|DEVELOPER_DIR)' \
        | sort
} >> "$ENV_LOG" 2>&1

log "  wrote $(basename "$ENV_LOG")"
fact hw_model "$(sysctl -n hw.model 2>/dev/null || echo unknown)"
fact cpu_brand "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
fact macos_version "$(sw_vers -productVersion 2>/dev/null || echo unknown)"
fact git_commit "$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)"
fact git_branch "$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
fact git_dirty "$(cd "$REPO_ROOT" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"

if ! have xcrun; then
    verdict metal_toolchain FAIL "xcrun not found — is this macOS with Xcode installed?"
elif xcrun metal --version > /dev/null 2>&1; then
    verdict metal_toolchain PASS "$(xcrun metal --version 2>&1 | head -1)"
else
    verdict metal_toolchain FAIL "xcrun metal unavailable — run: xcodebuild -downloadComponent MetalToolchain"
fi

# ===========================================================================
section "STAGE 1 — standalone Metal device capability probe (no repo build)"
# ===========================================================================
# Answers, without compiling one line of pycanha: does this machine expose a
# device that pycanha's own check would accept? mtl_device.mm computes
#     info.ray_tracing = [dev supportsRaytracing] && [dev supportsFamily:Apple9]
# and this probe reproduces exactly that expression, so its verdict IS
# pycanha's verdict.
#
# GPU family enumerators are passed as raw integers rather than the SDK
# symbols: an older SDK would not declare MTLGPUFamilyApple9, and
# supportsFamily: safely answers NO for a value it does not know, so the
# numeric form is both SDK-proof and future-proof (Apple10 shows up by
# itself).
PROBE_DEV_SRC="$WORK/probe_device.mm"
PROBE_DEV_BIN="$WORK/probe_device"
DEVICE_LOG="$OUT_DIR/02-metal-device-probe.txt"

cat > "$PROBE_DEV_SRC" <<'PROBE_DEVICE_EOF'
// Standalone Metal capability + split-counter probe for pycanha-core.
// Host code only: every shader is compiled at runtime from a string, so a
// wrong guess about MSL is a logged error, not a build failure.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

const char* yes_no(BOOL value) { return value ? "yes" : "no"; }

struct FamilyEntry {
    NSInteger value;
    const char* name;
};

// Raw MTLGPUFamily values: Apple1..10 = 1001.., Mac2 = 2002,
// Common1..3 = 3001.., Metal3/4 = 5001/5002.
const FamilyEntry kFamilies[] = {
    {1001, "Apple1"},  {1002, "Apple2"},  {1003, "Apple3"},
    {1004, "Apple4"},  {1005, "Apple5"},  {1006, "Apple6"},
    {1007, "Apple7"},  {1008, "Apple8"},  {1009, "Apple9"},
    {1010, "Apple10"}, {2002, "Mac2"},    {3001, "Common1"},
    {3002, "Common2"}, {3003, "Common3"}, {5001, "Metal3"},
    {5002, "Metal4"},
};

const char* location_name(MTLDeviceLocation location) {
    switch (location) {
        case MTLDeviceLocationBuiltIn: return "BuiltIn";
        case MTLDeviceLocationSlot: return "Slot";
        case MTLDeviceLocationExternal: return "External";
        case MTLDeviceLocationUnspecified: return "Unspecified";
        default: return "?";
    }
}

// The exact expression mtl_device.mm uses for DeviceInfo::ray_tracing.
BOOL pycanha_rt_verdict(id<MTLDevice> device) {
    return [device supportsRaytracing] &&
           [device supportsFamily:(MTLGPUFamily)1009];
}

void dump_device(id<MTLDevice> device, unsigned index) {
    printf("\n--- device[%u] ---------------------------------------------\n",
           index);
    printf("  name                          : %s\n",
           [[device name] UTF8String]);
    printf("  registryID                    : %llu\n",
           (unsigned long long)[device registryID]);
    printf("  location                      : %s (locationNumber=%lu)\n",
           location_name([device location]),
           (unsigned long)[device locationNumber]);
    printf("  headless / lowPower / removable: %s / %s / %s\n",
           yes_no([device isHeadless]), yes_no([device isLowPower]),
           yes_no([device isRemovable]));
    printf("  hasUnifiedMemory              : %s\n",
           yes_no([device hasUnifiedMemory]));
    printf("  recommendedMaxWorkingSetSize  : %llu bytes (%.2f GiB)\n",
           (unsigned long long)[device recommendedMaxWorkingSetSize],
           (double)[device recommendedMaxWorkingSetSize] / (1024.0 * 1024.0 * 1024.0));
    printf("  currentAllocatedSize          : %llu bytes\n",
           (unsigned long long)[device currentAllocatedSize]);
    printf("  maxBufferLength               : %llu bytes (%.2f GiB)\n",
           (unsigned long long)[device maxBufferLength],
           (double)[device maxBufferLength] / (1024.0 * 1024.0 * 1024.0));
    MTLSize tg = [device maxThreadsPerThreadgroup];
    printf("  maxThreadsPerThreadgroup      : %lu x %lu x %lu\n",
           (unsigned long)tg.width, (unsigned long)tg.height,
           (unsigned long)tg.depth);
    printf("  maxThreadgroupMemoryLength    : %lu bytes\n",
           (unsigned long)[device maxThreadgroupMemoryLength]);
    printf("  argumentBuffersSupport        : tier%ld\n",
           (long)[device argumentBuffersSupport] + 1);
    printf("  supportsRaytracing            : %s   <== gate 1\n",
           yes_no([device supportsRaytracing]));
    printf("  supportsRaytracingFromRender  : %s\n",
           yes_no([device supportsRaytracingFromRender]));
    printf("  supportsPrimitiveMotionBlur   : %s\n",
           yes_no([device supportsPrimitiveMotionBlur]));
    printf("  supportsFunctionPointers      : %s\n",
           yes_no([device supportsFunctionPointers]));
    printf("  supports32BitFloatFiltering   : %s\n",
           yes_no([device supports32BitFloatFiltering]));
    printf("  GPU families:\n");
    for (const FamilyEntry& entry : kFamilies) {
        BOOL supported = [device supportsFamily:(MTLGPUFamily)entry.value];
        printf("      %-9s (%4ld) : %s%s\n", entry.name, (long)entry.value,
               yes_no(supported),
               (entry.value == 1009 && supported) ? "   <== gate 2" : "");
    }
    printf("  PYCANHA VERDICT (supportsRaytracing && Apple9) : %s\n",
           pycanha_rt_verdict(device) ? "RT-CAPABLE" : "NOT RT-CAPABLE");
}

// --- runtime shader compilation --------------------------------------------
// Every kernel below is compiled here, against the real device, so the probe
// never needs the offline Metal toolchain and never fails to build because of
// a shader mistake.
id<MTLLibrary> compile_source(id<MTLDevice> device, const char* label,
                              const char* source, bool verbose) {
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    id<MTLLibrary> library =
        [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                             options:options
                               error:&error];
    if (library == nil) {
        printf("  [%s] COMPILE FAILED\n", label);
        if (verbose && error != nil) {
            printf("      %s\n",
                   [[error localizedDescription] UTF8String]);
        }
        return nil;
    }
    printf("  [%s] compiles OK\n", label);
    return library;
}

// The split 64-bit counter from kernels/common.slang (fp_add_raw, Metal
// branch): two 32-bit atomics at an 8-byte stride whose little-endian image
// is the u64. This runs it for real, with a value chosen so the low half
// wraps on nearly every thread — adding a whole multiple of 2^32 would pass
// even with the carry logic deleted.
const char* kSplitAddSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void split_add(device atomic_uint* cell   [[buffer(0)]],
                      constant ulong&     value  [[buffer(1)]],
                      uint                tid    [[thread_position_in_grid]]) {
    uint lo = uint(value & 0xFFFFFFFFul);
    uint hi = uint(value >> 32);
    uint old_lo = atomic_fetch_add_explicit(&cell[0], lo, memory_order_relaxed);
    uint carry = (old_lo + lo) < old_lo ? 1u : 0u;
    atomic_fetch_add_explicit(&cell[1], hi + carry, memory_order_relaxed);
}
)MSL";

// If this ever compiles AND runs, the split path in common.slang can collapse
// back to a single native atomic. It does not compile as of Metal 4.0.
const char* kNativeU64Source = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void native_add(device ulong* cell [[buffer(0)]],
                       uint tid [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit((device atomic_ulong*)cell, 1ul,
                              memory_order_relaxed);
}
)MSL";

int run_split_probe(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- split 64-bit counter: does it actually accumulate? ------\n");
    id<MTLLibrary> library =
        compile_source(device, "split_add", kSplitAddSource, true);
    if (library == nil) {
        printf("  RESULT: FAIL (kernel did not compile)\n");
        return 1;
    }
    NSError* error = nil;
    id<MTLFunction> function =
        [library newFunctionWithName:@"split_add"];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        printf("  pipeline creation FAILED: %s\n",
               error ? [[error localizedDescription] UTF8String] : "?");
        printf("  RESULT: FAIL\n");
        return 1;
    }
    printf("  pipeline: maxTotalThreadsPerThreadgroup=%lu "
           "threadExecutionWidth=%lu\n",
           (unsigned long)[pipeline maxTotalThreadsPerThreadgroup],
           (unsigned long)[pipeline threadExecutionWidth]);

    const uint32_t thread_count = 4096;
    const uint64_t value = 0x0000000180000007ull;  // low half wraps constantly
    id<MTLBuffer> cell = [device newBufferWithLength:8
                                             options:MTLResourceStorageModeShared];
    std::memset([cell contents], 0, 8);
    id<MTLBuffer> value_buffer =
        [device newBufferWithBytes:&value
                            length:sizeof(value)
                           options:MTLResourceStorageModeShared];

    NSUInteger group = [pipeline maxTotalThreadsPerThreadgroup];
    if (group > 256) { group = 256; }
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:cell offset:0 atIndex:0];
    [encoder setBuffer:value_buffer offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
        printf("  command buffer status=%ld error=%s\n",
               (long)[command_buffer status],
               [command_buffer error]
                   ? [[[command_buffer error] localizedDescription] UTF8String]
                   : "none");
        printf("  RESULT: FAIL (command buffer did not complete)\n");
        return 1;
    }

    uint32_t halves[2] = {0, 0};
    std::memcpy(halves, [cell contents], 8);
    const uint64_t got =
        (static_cast<uint64_t>(halves[1]) << 32) | static_cast<uint64_t>(halves[0]);
    const uint64_t expected = static_cast<uint64_t>(thread_count) * value;
    printf("  threads=%u  addend=0x%016llx\n", thread_count,
           (unsigned long long)value);
    printf("  lo=0x%08x hi=0x%08x -> 0x%016llx\n", halves[0], halves[1],
           (unsigned long long)got);
    printf("  expected                         0x%016llx\n",
           (unsigned long long)expected);
    if (got == expected) {
        printf("  RESULT: PASS (carry folding is exact under contention)\n");
        return 0;
    }
    printf("  RESULT: FAIL — the split counter loses carries on this GPU. "
           "This is fp_add_raw in kernels/common.slang.\n");
    return 1;
}

}  // namespace

int main() {
    @autoreleasepool {
        printf("=== NSProcessInfo ===\n");
        printf("  os version : %s\n",
               [[[NSProcessInfo processInfo] operatingSystemVersionString]
                   UTF8String]);
        printf("  physical memory : %llu bytes\n",
               (unsigned long long)[[NSProcessInfo processInfo] physicalMemory]);

        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        id<MTLDevice> system_default = MTLCreateSystemDefaultDevice();
        printf("\n=== MTLCopyAllDevices: %lu device(s) ===\n",
               (unsigned long)[devices count]);
        printf("  MTLCreateSystemDefaultDevice: %s\n",
               system_default ? [[system_default name] UTF8String] : "<null>");

        if ([devices count] == 0) {
            printf("\nNo Metal device at all. Nothing further can run.\n");
            printf("\nOVERALL: NO_DEVICE\n");
            return 3;
        }

        unsigned index = 0;
        id<MTLDevice> chosen = nil;
        for (id<MTLDevice> device in devices) {
            dump_device(device, index);
            if (chosen == nil && pycanha_rt_verdict(device)) { chosen = device; }
            ++index;
        }

        if (chosen == nil) {
            printf("\nNo device passes pycanha's RT gate "
                   "(supportsRaytracing && GPU family Apple9).\n");
            printf("Every [gpu] test in the suite will SKIP on this machine, "
                   "and a green ctest run proves nothing.\n");
            printf("\nOVERALL: NOT_RT_CAPABLE\n");
            return 2;
        }
        printf("\nSelected RT-capable device: %s\n",
               [[chosen name] UTF8String]);

        id<MTLCommandQueue> queue = [chosen newCommandQueue];
        if (queue == nil) {
            printf("newCommandQueue returned nil — this is exactly what "
                   "DeviceImpl's constructor throws on.\n");
            printf("\nOVERALL: NO_QUEUE\n");
            return 4;
        }
        printf("Command queue created OK.\n");

        printf("\n--- runtime shader compilation checks ----------------------\n");
        (void)compile_source(chosen, "native u64 atomic (expected to FAIL)",
                             kNativeU64Source, true);

        const int split_rc = run_split_probe(chosen, queue);

        printf("\nOVERALL: %s\n", split_rc == 0 ? "RT_CAPABLE_SPLIT_OK"
                                                : "RT_CAPABLE_SPLIT_BROKEN");
        return split_rc == 0 ? 0 : 5;
    }
}
PROBE_DEVICE_EOF

cp "$PROBE_DEV_SRC" "$OUT_DIR/02-metal-device-probe.mm"

: > "$DEVICE_LOG"
run_logged "$DEVICE_LOG" "compile probe_device.mm" \
    clang++ -std=c++20 -fobjc-arc -O1 -Wall \
        -framework Metal -framework Foundation \
        -o "$PROBE_DEV_BIN" "$PROBE_DEV_SRC"
PROBE_DEV_COMPILE_RC=$?

RT_CAPABLE=0
if [ $PROBE_DEV_COMPILE_RC -ne 0 ]; then
    verdict device_probe FAIL "the standalone probe itself did not compile — see 02-metal-device-probe.txt (the compiler output is at the top of that file)"
    verdict rt_device UNKNOWN "the probe that answers this did not build"
    verdict split_atomic UNKNOWN "the probe that answers this did not build"
else
    verdict device_probe PASS "probe built and ran"
    run_logged_tee "$DEVICE_LOG" "run probe_device" "$PROBE_DEV_BIN"
    PROBE_DEV_RC=$?
    fact device_probe_exit "$PROBE_DEV_RC"
    OVERALL_LINE="$(grep -m1 '^OVERALL:' "$DEVICE_LOG" | sed 's/^OVERALL: //')"
    fact device_probe_overall "${OVERALL_LINE:-UNKNOWN}"
    DEVICE_NAME="$(grep -m1 'Selected RT-capable device:' "$DEVICE_LOG" | sed 's/.*device: //')"
    fact device_name "${DEVICE_NAME:-unknown}"
    case "$PROBE_DEV_RC" in
        0) RT_CAPABLE=1
           verdict rt_device PASS "RT-capable: ${DEVICE_NAME:-?}"
           verdict split_atomic PASS "the 64-bit split counter accumulates exactly under contention" ;;
        5) RT_CAPABLE=1
           verdict rt_device PASS "RT-capable: ${DEVICE_NAME:-?}"
           verdict split_atomic FAIL "the split counter LOSES CARRIES — fp_add_raw in kernels/common.slang is wrong on this GPU" ;;
        2) verdict rt_device FAIL "no device passes supportsRaytracing && Apple9 — every [gpu] test will skip"
           verdict split_atomic SKIP "no RT device to run it on" ;;
        3) verdict rt_device FAIL "no Metal device at all"
           verdict split_atomic SKIP "no device" ;;
        4) verdict rt_device FAIL "command queue creation failed (DeviceImpl would throw here)"
           verdict split_atomic SKIP "no queue" ;;
        *) verdict rt_device UNKNOWN "probe exited $PROBE_DEV_RC — read 02-metal-device-probe.txt"
           verdict split_atomic UNKNOWN "probe exited $PROBE_DEV_RC" ;;
    esac
fi

# ===========================================================================
section "STAGE 2 — standalone ray-query probe (acceleration structure + intersection_query)"
# ===========================================================================
# The library's kernels use ray queries against a built acceleration
# structure. This probe does the same thing in ~150 lines with no pycanha
# involved, so an RT failure can be attributed to the machine/driver rather
# than to the port. It also answers a question the port had to guess at:
# whether MSL's is_committed_triangle_front_facing() agrees with the
# geometric substitute (dot(n, direction) < 0) that common.slang uses on
# Metal — §2.1 of the implementation guide.
#
# Two rays are traced at one triangle: one from +z (front) and one from -z
# (back). Both must hit at distance 1.0; the front-facing flags must come out
# opposite. Every shader variant is compiled at runtime and tried in order,
# so an MSL mistake costs a logged message.
PROBE_RT_SRC="$WORK/probe_rt.mm"
PROBE_RT_BIN="$WORK/probe_rt"
RT_LOG="$OUT_DIR/03-metal-rayquery-probe.txt"

cat > "$PROBE_RT_SRC" <<'PROBE_RT_EOF'
// Standalone Metal ray-query probe: builds a one-triangle primitive
// acceleration structure and traces two rays through intersection_query,
// mirroring what the Slang kernels do. Host code only; the shader is
// compiled at runtime, from several variants tried in order.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct RayIn {
    float origin[3];
    float direction[3];
};

// Variant 0 — the full query, including the front-facing flag.
const char* kSourceFull = R"MSL(
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct RayIn { packed_float3 origin; packed_float3 direction; };

kernel void rt_probe(primitive_acceleration_structure accel [[buffer(0)]],
                     device const RayIn* rays                [[buffer(1)]],
                     device float*       out                 [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    ray r;
    r.origin = float3(rays[tid].origin);
    r.direction = float3(rays[tid].direction);
    r.min_distance = 0.0f;
    r.max_distance = 1000.0f;

    intersection_query<triangle_data> q;
    q.reset(r, accel);
    while (q.next()) { }

    uint base = tid * 4u;
    if (q.get_committed_intersection_type() == intersection_type::triangle) {
        out[base + 0] = 1.0f;
        out[base + 1] = q.get_committed_distance();
        out[base + 2] = float(q.get_committed_primitive_id());
        out[base + 3] = q.is_committed_triangle_front_facing() ? 1.0f : 0.0f;
    } else {
        out[base + 0] = 0.0f;
        out[base + 1] = -1.0f;
        out[base + 2] = -1.0f;
        out[base + 3] = -1.0f;
    }
}
)MSL";

// Variant 1 — same, without the front-facing query (out[3] = -2 marks it).
const char* kSourceNoFrontFace = R"MSL(
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct RayIn { packed_float3 origin; packed_float3 direction; };

kernel void rt_probe(primitive_acceleration_structure accel [[buffer(0)]],
                     device const RayIn* rays                [[buffer(1)]],
                     device float*       out                 [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    ray r;
    r.origin = float3(rays[tid].origin);
    r.direction = float3(rays[tid].direction);
    r.min_distance = 0.0f;
    r.max_distance = 1000.0f;

    intersection_query<triangle_data> q;
    q.reset(r, accel);
    while (q.next()) { }

    uint base = tid * 4u;
    if (q.get_committed_intersection_type() == intersection_type::triangle) {
        out[base + 0] = 1.0f;
        out[base + 1] = q.get_committed_distance();
        out[base + 2] = float(q.get_committed_primitive_id());
        out[base + 3] = -2.0f;
    } else {
        out[base + 0] = 0.0f;
        out[base + 1] = -1.0f;
        out[base + 2] = -1.0f;
        out[base + 3] = -2.0f;
    }
}
)MSL";

// Variant 2 — no triangle_data tag at all, the most conservative form.
const char* kSourceBare = R"MSL(
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct RayIn { packed_float3 origin; packed_float3 direction; };

kernel void rt_probe(primitive_acceleration_structure accel [[buffer(0)]],
                     device const RayIn* rays                [[buffer(1)]],
                     device float*       out                 [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    ray r;
    r.origin = float3(rays[tid].origin);
    r.direction = float3(rays[tid].direction);
    r.min_distance = 0.0f;
    r.max_distance = 1000.0f;

    intersection_query<> q;
    q.reset(r, accel);
    while (q.next()) { }

    uint base = tid * 4u;
    if (q.get_committed_intersection_type() == intersection_type::triangle) {
        out[base + 0] = 1.0f;
        out[base + 1] = q.get_committed_distance();
        out[base + 2] = float(q.get_committed_primitive_id());
        out[base + 3] = -2.0f;
    } else {
        out[base + 0] = 0.0f;
        out[base + 1] = -1.0f;
        out[base + 2] = -1.0f;
        out[base + 3] = -2.0f;
    }
}
)MSL";

id<MTLLibrary> compile_first_that_works(id<MTLDevice> device, int* which) {
    const char* sources[] = {kSourceFull, kSourceNoFrontFace, kSourceBare};
    const char* names[] = {"full (with is_committed_triangle_front_facing)",
                           "no front-face query", "bare intersection_query<>"};
    for (int i = 0; i < 3; ++i) {
        NSError* error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithSource:[NSString stringWithUTF8String:sources[i]]
                                 options:[MTLCompileOptions new]
                                   error:&error];
        if (library != nil) {
            printf("  shader variant %d (%s): COMPILES\n", i, names[i]);
            *which = i;
            return library;
        }
        printf("  shader variant %d (%s): failed\n", i, names[i]);
        if (error != nil) {
            printf("      %s\n", [[error localizedDescription] UTF8String]);
        }
    }
    return nil;
}

}  // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = nil;
        for (id<MTLDevice> candidate in MTLCopyAllDevices()) {
            if ([candidate supportsRaytracing] &&
                [candidate supportsFamily:(MTLGPUFamily)1009]) {
                device = candidate;
                break;
            }
        }
        if (device == nil) {
            printf("No RT-capable device (supportsRaytracing && Apple9).\n");
            printf("\nOVERALL: NOT_RT_CAPABLE\n");
            return 2;
        }
        printf("device: %s\n", [[device name] UTF8String]);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            printf("\nOVERALL: NO_QUEUE\n");
            return 4;
        }

        printf("\n--- runtime shader compilation -----------------------------\n");
        int variant = -1;
        id<MTLLibrary> library = compile_first_that_works(device, &variant);
        if (library == nil) {
            printf("\nOVERALL: SHADER_COMPILE_FAILED\n");
            return 5;
        }

        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:@"rt_probe"];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            printf("pipeline creation FAILED: %s\n",
                   error ? [[error localizedDescription] UTF8String] : "?");
            printf("\nOVERALL: PIPELINE_FAILED\n");
            return 6;
        }
        printf("pipeline OK (maxTotalThreadsPerThreadgroup=%lu, "
               "threadExecutionWidth=%lu)\n",
               (unsigned long)[pipeline maxTotalThreadsPerThreadgroup],
               (unsigned long)[pipeline threadExecutionWidth]);

        // --- geometry: one CCW triangle in the z = 0 plane -----------------
        printf("\n--- acceleration structure ---------------------------------\n");
        const float vertices[9] = {0.0f, 0.0f, 0.0f,
                                   1.0f, 0.0f, 0.0f,
                                   0.0f, 1.0f, 0.0f};
        const uint32_t indices[3] = {0, 1, 2};
        id<MTLBuffer> vertex_buffer =
            [device newBufferWithBytes:vertices
                                length:sizeof(vertices)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> index_buffer =
            [device newBufferWithBytes:indices
                                length:sizeof(indices)
                               options:MTLResourceStorageModeShared];

        MTLAccelerationStructureTriangleGeometryDescriptor* geometry =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        geometry.vertexBuffer = vertex_buffer;
        geometry.vertexBufferOffset = 0;
        geometry.vertexStride = 3 * sizeof(float);
        geometry.indexBuffer = index_buffer;
        geometry.indexBufferOffset = 0;
        geometry.indexType = MTLIndexTypeUInt32;
        geometry.triangleCount = 1;
        geometry.opaque = YES;

        MTLPrimitiveAccelerationStructureDescriptor* descriptor =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        descriptor.geometryDescriptors = @[geometry];

        MTLAccelerationStructureSizes sizes =
            [device accelerationStructureSizesWithDescriptor:descriptor];
        printf("  sizes: structure=%lu scratch=%lu refit=%lu\n",
               (unsigned long)sizes.accelerationStructureSize,
               (unsigned long)sizes.buildScratchBufferSize,
               (unsigned long)sizes.refitScratchBufferSize);
        if (sizes.accelerationStructureSize == 0) {
            printf("  acceleration structure size is 0 — the descriptor was "
                   "rejected.\n");
            printf("\nOVERALL: AS_SIZE_ZERO\n");
            return 7;
        }

        id<MTLAccelerationStructure> accel =
            [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        id<MTLBuffer> scratch =
            [device newBufferWithLength:(sizes.buildScratchBufferSize > 0
                                             ? sizes.buildScratchBufferSize
                                             : 16)
                                options:MTLResourceStorageModePrivate];
        if (accel == nil) {
            printf("\nOVERALL: AS_ALLOC_FAILED\n");
            return 8;
        }

        id<MTLCommandBuffer> build_cb = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> build_encoder =
            [build_cb accelerationStructureCommandEncoder];
        [build_encoder buildAccelerationStructure:accel
                                       descriptor:descriptor
                                    scratchBuffer:scratch
                              scratchBufferOffset:0];
        [build_encoder endEncoding];
        [build_cb commit];
        [build_cb waitUntilCompleted];
        if ([build_cb status] != MTLCommandBufferStatusCompleted) {
            printf("  build command buffer status=%ld error=%s\n",
                   (long)[build_cb status],
                   [build_cb error]
                       ? [[[build_cb error] localizedDescription] UTF8String]
                       : "none");
            printf("\nOVERALL: AS_BUILD_FAILED\n");
            return 9;
        }
        printf("  build OK\n");

        // --- two rays: one from the front (+z), one from the back (-z) -----
        printf("\n--- trace --------------------------------------------------\n");
        RayIn rays[2];
        rays[0] = RayIn{{0.25f, 0.25f, 1.0f}, {0.0f, 0.0f, -1.0f}};
        rays[1] = RayIn{{0.25f, 0.25f, -1.0f}, {0.0f, 0.0f, 1.0f}};
        id<MTLBuffer> ray_buffer =
            [device newBufferWithBytes:rays
                                length:sizeof(rays)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer =
            [device newBufferWithLength:sizeof(float) * 8
                                options:MTLResourceStorageModeShared];
        std::memset([out_buffer contents], 0, sizeof(float) * 8);

        id<MTLCommandBuffer> trace_cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [trace_cb computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setAccelerationStructure:accel atBufferIndex:0];
        [encoder setBuffer:ray_buffer offset:0 atIndex:1];
        [encoder setBuffer:out_buffer offset:0 atIndex:2];
        // Residency: the guide flags useResource: as an untested path — it
        // reports misses silently rather than faulting, so exercise it.
        [encoder useResource:accel usage:MTLResourceUsageRead];
        [encoder dispatchThreads:MTLSizeMake(2, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(2, 1, 1)];
        [encoder endEncoding];
        [trace_cb commit];
        [trace_cb waitUntilCompleted];
        if ([trace_cb status] != MTLCommandBufferStatusCompleted) {
            printf("  trace command buffer status=%ld error=%s\n",
                   (long)[trace_cb status],
                   [trace_cb error]
                       ? [[[trace_cb error] localizedDescription] UTF8String]
                       : "none");
            printf("\nOVERALL: TRACE_FAILED\n");
            return 10;
        }

        float out[8];
        std::memcpy(out, [out_buffer contents], sizeof(out));
        const char* labels[2] = {"ray 0: from +z, direction -z (front side)",
                                 "ray 1: from -z, direction +z (back side)"};
        int failures = 0;
        for (int i = 0; i < 2; ++i) {
            printf("  %s\n", labels[i]);
            printf("      hit=%.0f distance=%.6f primitive_id=%.0f "
                   "front_facing=%s\n",
                   (double)out[i * 4 + 0], (double)out[i * 4 + 1],
                   (double)out[i * 4 + 2],
                   out[i * 4 + 3] < -1.5f
                       ? "<not queried>"
                       : (out[i * 4 + 3] > 0.5f ? "yes" : "no"));
            if (out[i * 4 + 0] < 0.5f) {
                printf("      MISS — the ray should have hit at distance 1.0\n");
                ++failures;
            } else if (out[i * 4 + 1] < 0.99f || out[i * 4 + 1] > 1.01f) {
                printf("      WRONG DISTANCE (expected 1.0)\n");
                ++failures;
            }
        }

        if (variant == 0 && out[0] > 0.5f && out[4] > 0.5f) {
            const bool front0 = out[3] > 0.5f;
            const bool front1 = out[7] > 0.5f;
            printf("\n  front-facing agreement check "
                   "(common.slang §2.1 substitutes dot(n, dir) < 0):\n");
            printf("      ray 0 (geometric front) : MSL says %s\n",
                   front0 ? "front" : "back");
            printf("      ray 1 (geometric back)  : MSL says %s\n",
                   front1 ? "front" : "back");
            if (front0 && !front1) {
                printf("      AGREES with the geometric substitute — the "
                       "Metal branch of trace_ray is consistent, and MSL's "
                       "own predicate is available if it is ever wanted.\n");
            } else {
                printf("      DISAGREES — worth investigating before trusting "
                       "side (slot) assignment on Metal.\n");
                ++failures;
            }
        }

        printf("\nOVERALL: %s\n", failures == 0 ? "RAYQUERY_OK" : "RAYQUERY_WRONG");
        return failures == 0 ? 0 : 11;
    }
}
PROBE_RT_EOF

cp "$PROBE_RT_SRC" "$OUT_DIR/03-metal-rayquery-probe.mm"

: > "$RT_LOG"
run_logged "$RT_LOG" "compile probe_rt.mm" \
    clang++ -std=c++20 -fobjc-arc -O1 -Wall \
        -framework Metal -framework Foundation \
        -o "$PROBE_RT_BIN" "$PROBE_RT_SRC"
PROBE_RT_COMPILE_RC=$?

if [ $PROBE_RT_COMPILE_RC -ne 0 ]; then
    verdict ray_query WARN "the ray-query probe did not compile (host code) — see 03-metal-rayquery-probe.txt; this is a limitation of the probe, not necessarily of the backend"
else
    run_logged_tee "$RT_LOG" "run probe_rt" "$PROBE_RT_BIN"
    PROBE_RT_RC=$?
    RT_OVERALL="$(grep -m1 '^OVERALL:' "$RT_LOG" | sed 's/^OVERALL: //')"
    fact rayquery_overall "${RT_OVERALL:-UNKNOWN}"
    case "$PROBE_RT_RC" in
        0) verdict ray_query PASS "acceleration structure build + intersection_query behave correctly" ;;
        2) verdict ray_query SKIP "no RT-capable device" ;;
        *) verdict ray_query FAIL "${RT_OVERALL:-exit $PROBE_RT_RC} — see 03-metal-rayquery-probe.txt" ;;
    esac
fi

# ===========================================================================
section "STAGE 3 — Metal 64-bit atomic compile matrix"
# ===========================================================================
ATOMICS_LOG="$OUT_DIR/04-atomics64-matrix.txt"
if [ -x "$SCRIPT_DIR/probe_metal_atomics64.sh" ]; then
    : > "$ATOMICS_LOG"
    run_logged_tee "$ATOMICS_LOG" "probe_metal_atomics64.sh" \
        "$SCRIPT_DIR/probe_metal_atomics64.sh"
    ATOMICS_RC=$?
    if [ $ATOMICS_RC -ne 0 ]; then
        verdict atomics64 FAIL "the probe exited $ATOMICS_RC (usually a missing Metal toolchain) — see 04-atomics64-matrix.txt"
    elif grep -q 'A 64-bit atomic now compiles' "$ATOMICS_LOG" 2>/dev/null; then
        verdict atomics64 WARN "a 64-bit atomic now COMPILES — the split counter in common.slang could collapse to the native path"
    else
        verdict atomics64 PASS "no native 64-bit buffer atomic (expected) — the split counter is still required"
    fi
elif [ -f "$SCRIPT_DIR/probe_metal_atomics64.sh" ]; then
    run_logged_tee "$ATOMICS_LOG" "probe_metal_atomics64.sh" \
        bash "$SCRIPT_DIR/probe_metal_atomics64.sh"
    verdict atomics64 PASS "ran via bash (script was not executable)"
else
    verdict atomics64 SKIP "test/tools/probe_metal_atomics64.sh not found"
fi

if [ "$PROBE_ONLY" -eq 1 ]; then
    log ""
    log "--probe-only: stopping before the Conan/CMake build."
    SKIP_BUILD=1
    SKIP_TESTS=1
else
    SKIP_TESTS=0
fi

# ===========================================================================
section "STAGE 4 — Conan + CMake build"
# ===========================================================================
CONAN_LOG="$OUT_DIR/10-conan-install.log"
CONFIGURE_LOG="$OUT_DIR/11-cmake-configure.log"
BUILD_LOG="$OUT_DIR/12-build.log"
BUILD_OK=0

# CI parity: macOS builds without MKL, and ASan's container-overflow check is
# a known false positive against the un-instrumented symengine dependency.
export ASAN_OPTIONS="detect_container_overflow=0${ASAN_OPTIONS:+,$ASAN_OPTIONS}"
export CTEST_OUTPUT_ON_FAILURE=1
fact asan_options "$ASAN_OPTIONS"

CONAN_OPTS="-o PYCANHA_OPTION_USE_MKL=False"
if [ "$FAST" -eq 1 ]; then
    CONAN_OPTS="$CONAN_OPTS -o PYCANHA_OPTION_LTO=False -o PYCANHA_OPTION_SANITIZE_ADDR=False -o PYCANHA_OPTION_SANITIZE_UNDEF=False"
fi
fact conan_options "$CONAN_OPTS"

if [ "$PROBE_ONLY" -eq 1 ]; then
    verdict build SKIP "--probe-only"
elif [ "$SKIP_BUILD" -eq 1 ]; then
    if [ -d "$BUILD_DIR" ]; then
        verdict build SKIP "--skip-build, reusing $BUILD_DIR"
        BUILD_OK=1
    else
        verdict build FAIL "--skip-build was given but $BUILD_DIR does not exist"
    fi
elif ! have conan || ! have cmake; then
    verdict build FAIL "conan and/or cmake not on PATH — activate the repo .venv first (see CLAUDE.md)"
else
    cd "$REPO_ROOT" || exit 1
    : > "$CONAN_LOG"

    if [ "$CONAN_CONFIG" -eq 1 ]; then
        run_logged "$CONAN_LOG" "conan config install .conan" \
            conan config install .conan
    fi
    run_logged "$CONAN_LOG" "conan profile show (host)" \
        conan profile show -pr:h=macos-clang21-arm64 -pr:h=build-$BUILD_TYPE -pr:h=options-ci

    # shellcheck disable=SC2086
    run_logged "$CONAN_LOG" "conan install" \
        conan install . --build=missing \
            -pr:h=macos-clang21-arm64 -pr:h=build-$BUILD_TYPE -pr:h=options-ci \
            -pr:b=macos-clang21-arm64 -pr:b=build-$BUILD_TYPE -pr:b=options-ci \
            $CONAN_OPTS
    CONAN_RC=$?

    if [ $CONAN_RC -ne 0 ]; then
        verdict build FAIL "conan install failed — see 10-conan-install.log"
    else
        : > "$CONFIGURE_LOG"
        run_logged "$CONFIGURE_LOG" "cmake --preset=$CONAN_PRESET" \
            cmake --preset="$CONAN_PRESET"
        CONFIGURE_RC=$?

        if [ $CONFIGURE_RC -ne 0 ]; then
            verdict configure FAIL "cmake configure failed — see 11-cmake-configure.log"
        else
            verdict configure PASS "configured with preset $CONAN_PRESET"
            : > "$BUILD_LOG"
            run_logged "$BUILD_LOG" "cmake --build (tests)" \
                cmake --build "$BUILD_DIR" --target tests --parallel "$JOBS"
            BUILD_RC=$?

            if [ $BUILD_RC -ne 0 ]; then
                # One Mac-only warning must not cost a whole round trip: retry
                # once without -Werror so the run still produces test results.
                log ""
                log "  build failed — retrying once with WARNINGS_AS_ERRORS=False"
                log "  (this is NOT CI parity; the original failure is in 12-build.log)"
                # shellcheck disable=SC2086
                run_logged "$CONAN_LOG" "conan install (retry, -Werror off)" \
                    conan install . --build=missing \
                        -pr:h=macos-clang21-arm64 -pr:h=build-$BUILD_TYPE -pr:h=options-ci \
                        -pr:b=macos-clang21-arm64 -pr:b=build-$BUILD_TYPE -pr:b=options-ci \
                        $CONAN_OPTS -o PYCANHA_OPTION_WARNINGS_AS_ERRORS=False
                run_logged "$CONFIGURE_LOG" "cmake --preset (retry)" \
                    cmake --preset="$CONAN_PRESET"
                run_logged "$BUILD_LOG" "cmake --build (retry, -Werror off)" \
                    cmake --build "$BUILD_DIR" --target tests --parallel "$JOBS"
                RETRY_RC=$?
                if [ $RETRY_RC -eq 0 ]; then
                    BUILD_OK=1
                    verdict build WARN "built only with -Werror OFF — there are Mac-only warnings to fix; both attempts are in 12-build.log"
                else
                    verdict build FAIL "build failed with and without -Werror — see 12-build.log"
                fi
            else
                BUILD_OK=1
                verdict build PASS "library + tests built (CI options, -Werror on)"
            fi
        fi
    fi
fi

# Compiler diagnostics summary — warnings are the first thing to fix and the
# easiest thing to lose in a 20 MB log.
if [ -f "$BUILD_LOG" ]; then
    {
        echo "=== error lines ==="
        grep -nE 'error:' "$BUILD_LOG" | head -200
        echo
        echo "=== warning lines (deduplicated by message) ==="
        grep -oE 'warning:.*' "$BUILD_LOG" | sort | uniq -c | sort -rn | head -100
        echo
        echo "=== objective-c++ / metal related lines ==="
        grep -nEi '\.mm|objcxx|metal|slang|metallib|framework' "$BUILD_LOG" | head -120
    } > "$OUT_DIR/12-build-diagnostics.txt" 2>&1
    fact build_errors "$(count_matches -E 'error:' "$BUILD_LOG")"
    fact build_warnings "$(count_matches -E 'warning:' "$BUILD_LOG")"
fi

# ===========================================================================
section "STAGE 5 — generated kernel artifacts"
# ===========================================================================
# These are the things that cannot be regenerated remotely: the MSL slangc
# produced, the reflection dump the buffer indices come from, the .metallib,
# and the generated binding headers. Ship them all.
KERNELS_FOUND=0
if [ -d "$BUILD_DIR" ]; then
    for pattern in '*.metal' '*.reflect.json' '*.metallib' '*.air' '*_bindings.h' '*_metallib.h'; do
        find "$BUILD_DIR" -name "$pattern" -type f 2>/dev/null | while read -r f; do
            cp "$f" "$KERNEL_DIR/" 2>/dev/null
        done
    done
    KERNELS_FOUND="$(ls -1 "$KERNEL_DIR" 2>/dev/null | wc -l | tr -d ' ')"
fi
log "  collected $KERNELS_FOUND generated kernel artifact(s) into 13-kernels/"
fact kernel_artifacts "$KERNELS_FOUND"

KERNEL_ANALYSIS="$OUT_DIR/13-kernel-analysis.txt"
{
    echo "Generated Metal kernel analysis"
    echo "==============================="
    echo
    echo "Files collected:"
    ls -la "$KERNEL_DIR" 2>&1
    echo
    echo "--- atomic operation census -------------------------------------"
    echo "The exchange kernel MUST contain atomic_fetch_add_explicit calls."
    echo "Zero of them means the deposit degraded to a non-atomic"
    echo "'buf[idx] = buf[idx] + value' read-modify-write, which compiles,"
    echo "passes pipeline creation, and silently loses ~31 of every 32"
    echo "concurrent deposits. See roadmap 22 §9."
    echo
    for msl in "$KERNEL_DIR"/*.metal; do
        [ -f "$msl" ] || continue
        name="$(basename "$msl")"
        printf '  %-22s atomic_fetch_add=%-4s atomic_*=%-4s intersection_query=%-4s lines=%s\n' \
            "$name" \
            "$(grep -c 'atomic_fetch_add' "$msl" 2>/dev/null)" \
            "$(grep -c 'atomic_' "$msl" 2>/dev/null)" \
            "$(grep -c 'intersection_query\|intersector' "$msl" 2>/dev/null)" \
            "$(wc -l < "$msl" | tr -d ' ')"
    done
    echo
    echo "--- non-atomic read-modify-write suspects -----------------------"
    for msl in "$KERNEL_DIR"/*.metal; do
        [ -f "$msl" ] || continue
        hits="$(grep -nE '^\s*\*?_?S?[A-Za-z0-9_]+ = \*?_?S?[A-Za-z0-9_]+ \+' "$msl" 2>/dev/null | head -10)"
        if [ -n "$hits" ]; then
            echo "  $(basename "$msl"):"
            echo "$hits" | sed 's/^/      /'
        fi
    done
    echo
    echo "--- generated buffer index tables --------------------------------"
    for hdr in "$KERNEL_DIR"/*_bindings.h; do
        [ -f "$hdr" ] || continue
        echo "  === $(basename "$hdr") ==="
        sed 's/^/      /' "$hdr"
    done
    echo
    echo "--- reflection JSON entry points ---------------------------------"
    for js in "$KERNEL_DIR"/*.reflect.json; do
        [ -f "$js" ] || continue
        echo "  === $(basename "$js") ($(wc -c < "$js" | tr -d ' ') bytes) ==="
        if have python3; then
            python3 - "$js" <<'PY' 2>&1 | sed 's/^/      /'
import json, sys
with open(sys.argv[1]) as fh:
    data = json.load(fh)
def walk(node, depth=0):
    if depth > 3:
        return
    if isinstance(node, dict):
        for key in ("name", "binding", "index", "stage", "kind", "type"):
            if key in node and not isinstance(node[key], (dict, list)):
                print("  " * depth + f"{key}: {node[key]}")
        for value in node.values():
            walk(value, depth + 1)
    elif isinstance(node, list):
        for item in node[:40]:
            walk(item, depth + 1)
walk(data)
PY
        else
            head -c 4000 "$js"
        fi
    done
    echo
    echo "--- metallib disassembly (best effort) ---------------------------"
    for lib in "$KERNEL_DIR"/*.metallib; do
        [ -f "$lib" ] || continue
        echo "  === $(basename "$lib") ($(wc -c < "$lib" | tr -d ' ') bytes) ==="
        if xcrun --find metal-objdump > /dev/null 2>&1; then
            xcrun metal-objdump --macho --archive-headers "$lib" 2>&1 | head -30 | sed 's/^/      /'
        elif xcrun --find metal-nm > /dev/null 2>&1; then
            xcrun metal-nm "$lib" 2>&1 | head -30 | sed 's/^/      /'
        else
            echo "      (no metal-objdump / metal-nm available)"
        fi
    done
} > "$KERNEL_ANALYSIS" 2>&1

EXCHANGE_MSL="$KERNEL_DIR/exchange.metal"
if [ -f "$EXCHANGE_MSL" ]; then
    ATOMIC_ADDS="$(count_matches 'atomic_fetch_add' "$EXCHANGE_MSL")"
    fact exchange_atomic_adds "$ATOMIC_ADDS"
    if [ "$ATOMIC_ADDS" -ge 2 ]; then
        verdict kernel_codegen PASS "exchange.metal has $ATOMIC_ADDS atomic_fetch_add calls (the split pair)"
    elif [ "$ATOMIC_ADDS" -ge 1 ]; then
        verdict kernel_codegen WARN "exchange.metal has only $ATOMIC_ADDS atomic_fetch_add call — expected 2 for the split counter"
    else
        verdict kernel_codegen FAIL "exchange.metal has NO atomic add — the deposit degraded to a non-atomic read-modify-write (roadmap 22 §9)"
    fi
else
    verdict kernel_codegen SKIP "no generated exchange.metal found (build did not get that far)"
fi

# ===========================================================================
section "STAGE 6 — test suite"
# ===========================================================================
CTEST_LOG="$OUT_DIR/20-ctest.log"
INVENTORY="$OUT_DIR/20-test-inventory.txt"
RADIATIVE_LOG="$OUT_DIR/22-tests-radiative-verbose.log"
GPU_LOG="$OUT_DIR/23-tests-gpu-verbose.log"
SKIP_ANALYSIS="$OUT_DIR/24-gpu-skip-analysis.txt"

TESTS_BIN=""
if [ -d "$BUILD_DIR" ]; then
    TESTS_BIN="$(find "$BUILD_DIR" -type f -name tests -perm -111 2>/dev/null | head -1)"
fi
fact tests_binary "${TESTS_BIN:-<not found>}"

if [ "$SKIP_TESTS" -eq 1 ]; then
    verdict ctest SKIP "--probe-only"
elif [ "$BUILD_OK" -eq 0 ]; then
    verdict ctest SKIP "no usable build"
elif [ -z "$TESTS_BIN" ]; then
    verdict ctest FAIL "the tests binary was not found under $BUILD_DIR"
else
    log "  tests binary: $TESTS_BIN"

    # Inventory first: what exists, and how much of it is GPU-gated.
    {
        echo "=== ctest -N (all) ==="
        ctest --test-dir "$BUILD_DIR" -N 2>&1
        echo
        echo "=== ctest -N -L gpu ==="
        ctest --test-dir "$BUILD_DIR" -N -L gpu 2>&1
        echo
        echo "=== ctest -N -L radiative ==="
        ctest --test-dir "$BUILD_DIR" -N -L radiative 2>&1
        echo
        echo "=== tests --list-tags ==="
        "$TESTS_BIN" --list-tags 2>&1
    } > "$INVENTORY" 2>&1

    TOTAL_TESTS="$(grep -m1 'Total Tests:' "$INVENTORY" | awk '{print $3}')"
    GPU_TESTS="$(awk '/=== ctest -N -L gpu ===/,/=== ctest -N -L radiative ===/' "$INVENTORY" | grep -m1 'Total Tests:' | awk '{print $3}')"
    fact ctest_total "${TOTAL_TESTS:-unknown}"
    fact ctest_gpu_labelled "${GPU_TESTS:-unknown}"
    log "  discovered ${TOTAL_TESTS:-?} ctest cases, ${GPU_TESTS:-?} labelled [gpu]"

    : > "$CTEST_LOG"
    run_logged "$CTEST_LOG" "ctest (full suite)" \
        ctest --test-dir "$BUILD_DIR" --output-on-failure --no-tests=error
    CTEST_RC=$?

    PASSED="$(grep -oE '[0-9]+% tests passed, [0-9]+ tests failed out of [0-9]+' "$CTEST_LOG" | tail -1)"
    FAILED_COUNT="$(echo "$PASSED" | grep -oE '[0-9]+ tests failed' | grep -oE '^[0-9]+')"
    fact ctest_result_line "${PASSED:-unknown}"
    fact ctest_exit "$CTEST_RC"

    if [ $CTEST_RC -eq 0 ]; then
        verdict ctest PASS "${PASSED:-all tests passed}"
    else
        verdict ctest FAIL "${PASSED:-ctest exited $CTEST_RC} — see 20-ctest.log"
        grep -A 40 'The following tests FAILED' "$CTEST_LOG" > "$OUT_DIR/21-ctest-failures.txt" 2>&1
    fi

    if [ -n "$EXPECT_TESTS" ] && [ -n "${TOTAL_TESTS:-}" ]; then
        if [ "$TOTAL_TESTS" = "$EXPECT_TESTS" ]; then
            verdict test_count PASS "$TOTAL_TESTS cases, matches --expect-tests"
        else
            verdict test_count FAIL "$TOTAL_TESTS cases but --expect-tests said $EXPECT_TESTS — a test file is missing from this platform's build"
        fi
    fi

    # Verbose per-assertion runs. Catch2's -s prints expanded values for
    # PASSING assertions too, which is what turns "green" into numbers an
    # agent can reason about without another trip.
    : > "$RADIATIVE_LOG"
    run_logged "$RADIATIVE_LOG" "tests [radiative] -s --durations yes" \
        "$TESTS_BIN" "[radiative]" -s --durations yes --order decl \
            --warn NoAssertions
    : > "$GPU_LOG"
    run_logged "$GPU_LOG" "tests [gpu] -s --durations yes" \
        "$TESTS_BIN" "[gpu]" -s --durations yes --order decl --warn NoAssertions

    # --- the trap: did the GPU cases actually execute? ---------------------
    SKIP_HITS="$(count_matches 'no RT-capable' "$GPU_LOG")"
    {
        echo "Did the [gpu] test cases actually run, or did they skip?"
        echo "======================================================="
        echo
        echo "Every [gpu] case in this repository starts with"
        echo "    if (!rad::is_available()) { SUCCEED(\"no RT-capable GPU device\"); return; }"
        echo "so on a machine the backend does not recognise the whole suite"
        echo "reports GREEN while executing no GPU code at all."
        echo
        echo "occurrences of 'no RT-capable' in the [gpu] run : $SKIP_HITS"
        echo "ctest cases labelled [gpu]                      : ${GPU_TESTS:-unknown}"
        echo "standalone probe verdict                        : $(verdict_of rt_device)"
        echo
        echo "--- the matching lines ---"
        grep -n 'no RT-capable' "$GPU_LOG" 2>/dev/null | head -40
        echo
        echo "--- reported GPU time (nonzero means kernels really ran) ---"
        grep -niE 'gpu_time|gpu time|duration|device' "$GPU_LOG" 2>/dev/null | head -60
    } > "$SKIP_ANALYSIS" 2>&1

    fact gpu_skip_messages "$SKIP_HITS"
    if [ "$SKIP_HITS" -gt 0 ]; then
        verdict gpu_executed FAIL "$SKIP_HITS GPU case(s) skipped for lack of an RT device — a green suite here proves nothing"
    elif [ "$RT_CAPABLE" -eq 1 ]; then
        verdict gpu_executed PASS "no skip messages: the GPU cases executed on real hardware"
    else
        verdict gpu_executed UNKNOWN "no skip messages found, but the standalone probe did not confirm an RT device"
    fi
fi

# ===========================================================================
section "STAGE 7 — determinism and cross-backend bit-diff"
# ===========================================================================
# The hidden [.][bitdiff] case dumps a fixed-seed VF matrix and a fixed-seed
# exchange matrix as raw double bit patterns. Two uses:
#
#   1. SELF-COMPARISON (no other machine needed): run it twice and diff. The
#      cells are integer atomics, so the result is deterministic BY
#      CONSTRUCTION. Any difference between two runs on the same GPU is a
#      race, a missing barrier or a residency mistake — never noise.
#   2. CROSS-BACKEND: diff against the same dump from a Vulkan machine at the
#      same commit. Must match byte for byte.
BITDIFF_DIR="$OUT_DIR/30-bitdiff"
BITDIFF_LOG="$OUT_DIR/30-bitdiff.log"
mkdir -p "$BITDIFF_DIR/run1" "$BITDIFF_DIR/run2"

if [ "$SKIP_TESTS" -eq 1 ] || [ -z "$TESTS_BIN" ]; then
    verdict determinism SKIP "no test binary"
    verdict cross_backend SKIP "no test binary"
elif [ "$RT_CAPABLE" -eq 0 ]; then
    verdict determinism SKIP "no RT device — the dump would be empty"
    verdict cross_backend SKIP "no RT device"
else
    : > "$BITDIFF_LOG"
    for run in 1 2; do
        ( cd "$BITDIFF_DIR/run$run" && \
          "$TESTS_BIN" "[bitdiff]" -s ) >> "$BITDIFF_LOG" 2>&1
        echo "### run $run exit=$?" >> "$BITDIFF_LOG"
    done

    if [ -f "$BITDIFF_DIR/run1/bitdiff_vf.txt" ] && \
       [ -f "$BITDIFF_DIR/run2/bitdiff_vf.txt" ]; then
        SELF_DIFF="$OUT_DIR/31-bitdiff-selfcompare.txt"
        {
            echo "Same machine, same seed, two consecutive runs."
            echo "Integer accumulators make this deterministic by construction:"
            echo "any difference is a race or a residency bug."
            echo
            for f in bitdiff_vf.txt bitdiff_exchange.txt; do
                echo "=== $f ==="
                if [ ! -f "$BITDIFF_DIR/run1/$f" ]; then
                    echo "  MISSING from run 1"
                elif [ ! -f "$BITDIFF_DIR/run2/$f" ]; then
                    echo "  MISSING from run 2"
                else
                    echo "  run1: $(wc -l < "$BITDIFF_DIR/run1/$f" | tr -d ' ') lines, $(wc -c < "$BITDIFF_DIR/run1/$f" | tr -d ' ') bytes"
                    echo "  run2: $(wc -l < "$BITDIFF_DIR/run2/$f" | tr -d ' ') lines, $(wc -c < "$BITDIFF_DIR/run2/$f" | tr -d ' ') bytes"
                    if diff -q "$BITDIFF_DIR/run1/$f" "$BITDIFF_DIR/run2/$f" > /dev/null 2>&1; then
                        echo "  IDENTICAL"
                    else
                        echo "  DIFFERENT — first 60 differing lines:"
                        diff "$BITDIFF_DIR/run1/$f" "$BITDIFF_DIR/run2/$f" | head -60
                    fi
                fi
                echo
            done
            echo "=== conservation error reported by the run ==="
            grep -i 'conservation error' "$BITDIFF_LOG" 2>/dev/null
            echo "(must be exactly 0; the kernel flushes every ray's balance,"
            echo " so nonzero means a broken deposit, not Monte-Carlo noise)"
        } > "$SELF_DIFF" 2>&1

        if diff -q "$BITDIFF_DIR/run1/bitdiff_vf.txt" "$BITDIFF_DIR/run2/bitdiff_vf.txt" > /dev/null 2>&1 && \
           diff -q "$BITDIFF_DIR/run1/bitdiff_exchange.txt" "$BITDIFF_DIR/run2/bitdiff_exchange.txt" > /dev/null 2>&1; then
            verdict determinism PASS "two consecutive fixed-seed runs are byte-identical"
        else
            verdict determinism FAIL "two runs of the SAME fixed seed differ — a race, a missing barrier or a residency mistake; see 31-bitdiff-selfcompare.txt"
        fi

        CONS="$(grep -i 'conservation error' "$BITDIFF_LOG" | tail -1)"
        fact conservation "${CONS:-not reported}"
        if echo "$CONS" | grep -qE 'error: 0$|error: 0[^0-9]'; then
            verdict conservation PASS "$CONS"
        elif [ -n "$CONS" ]; then
            verdict conservation FAIL "$CONS — must be exactly 0"
        else
            verdict conservation UNKNOWN "the run did not report a conservation error"
        fi
    else
        verdict determinism FAIL "the [bitdiff] case produced no dump files — see 30-bitdiff.log"
    fi

    # Cross-backend comparison against a Vulkan machine's dumps.
    if [ -n "$REFERENCE_DIR" ]; then
        XREF="$OUT_DIR/32-bitdiff-vs-reference.txt"
        xdiff_bad=0
        {
            echo "Cross-backend comparison against: $REFERENCE_DIR"
            echo "Same kernel sources + integer atomics => must match BYTE FOR BYTE."
            echo
            for f in bitdiff_vf.txt bitdiff_exchange.txt; do
                echo "=== $f ==="
                if [ ! -f "$REFERENCE_DIR/$f" ]; then
                    echo "  reference file missing"
                    continue
                fi
                if diff -q "$BITDIFF_DIR/run1/$f" "$REFERENCE_DIR/$f" > /dev/null 2>&1; then
                    echo "  IDENTICAL to the reference"
                else
                    echo "  DIFFERENT — first 80 differing lines:"
                    diff "$REFERENCE_DIR/$f" "$BITDIFF_DIR/run1/$f" | head -80
                fi
                echo
            done
        } > "$XREF" 2>&1
        if grep -q 'DIFFERENT' "$XREF"; then xdiff_bad=1; fi
        if grep -q 'reference file missing' "$XREF"; then xdiff_bad=1; fi
        if [ "$xdiff_bad" -eq 0 ]; then
            verdict cross_backend PASS "bit-identical to the Vulkan reference"
        else
            verdict cross_backend FAIL "differs from the Vulkan reference — see 32-bitdiff-vs-reference.txt"
        fi
    else
        verdict cross_backend SKIP "no --reference-dir given; copy 30-bitdiff/run1/*.txt to a Vulkan machine and diff there"
    fi
fi

# ===========================================================================
section "STAGE 8 — Metal validation layers"
# ===========================================================================
# The API validation layer and shader validation catch precisely the class of
# bug the port is most exposed to: a wrong per-kernel buffer index, a
# resource left out of useResource:, an out-of-bounds device write. They are
# far too slow for a normal run, so this is a separate, targeted pass.
VALIDATION_LOG="$OUT_DIR/40-metal-validation.log"
if [ "$SKIP_TESTS" -eq 1 ] || [ -z "$TESTS_BIN" ] || [ "$RT_CAPABLE" -eq 0 ]; then
    verdict validation SKIP "no RT device / no test binary"
else
    : > "$VALIDATION_LOG"
    {
        echo "Environment for this pass:"
        echo "  MTL_DEBUG_LAYER=1"
        echo "  MTL_DEBUG_LAYER_ERROR_MODE=nslog"
        echo "  MTL_SHADER_VALIDATION=1"
        echo "  MTL_SHADER_VALIDATION_GLOBAL_MEMORY=1"
        echo
    } >> "$VALIDATION_LOG"
    (
        export MTL_DEBUG_LAYER=1
        export MTL_DEBUG_LAYER_ERROR_MODE=nslog
        export MTL_SHADER_VALIDATION=1
        export MTL_SHADER_VALIDATION_GLOBAL_MEMORY=1
        run_logged "$VALIDATION_LOG" "tests [gpu] under Metal validation" \
            "$TESTS_BIN" "[gpu]" --durations yes
    )
    VALIDATION_RC=$?
    VAL_HITS="$(count_matches -iE 'validation|MTLDebug|assertion failed|out of bounds|not resident|hazard' "$VALIDATION_LOG")"
    fact validation_hits "$VAL_HITS"
    if [ $VALIDATION_RC -eq 0 ] && [ "$VAL_HITS" -le 4 ]; then
        verdict validation PASS "clean under API + shader validation"
    elif [ $VALIDATION_RC -eq 0 ]; then
        verdict validation WARN "tests passed but the validation layer emitted $VAL_HITS message(s) — read 40-metal-validation.log"
    else
        verdict validation FAIL "tests failed under validation (exit $VALIDATION_RC) — read 40-metal-validation.log"
    fi
fi

# ===========================================================================
section "STAGE 9 — crash reports and backtraces"
# ===========================================================================
CRASH_DIR="$OUT_DIR/50-crash-reports"
mkdir -p "$CRASH_DIR"
CRASH_COUNT=0
for d in "$HOME/Library/Logs/DiagnosticReports" "/Library/Logs/DiagnosticReports"; do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 1 -type f \( -name 'tests*' -o -name 'probe_*' \) \
        -newer "$WORK/.run-start" 2>/dev/null | while read -r report; do
        cp "$report" "$CRASH_DIR/" 2>/dev/null
    done
done
CRASH_COUNT="$(ls -1 "$CRASH_DIR" 2>/dev/null | wc -l | tr -d ' ')"
fact crash_reports "$CRASH_COUNT"
if [ "$CRASH_COUNT" -gt 0 ]; then
    verdict crashes FAIL "$CRASH_COUNT crash report(s) captured in 50-crash-reports/"
else
    verdict crashes PASS "no crash reports generated during this run"
fi

# An lldb backtrace is worth more than any log line when something aborts.
if [ -n "$TESTS_BIN" ] && [ "$SKIP_TESTS" -eq 0 ] && \
   { [ "$(verdict_of ctest)" = "FAIL" ] || [ "$CRASH_COUNT" -gt 0 ]; } && have lldb; then
    LLDB_LOG="$OUT_DIR/51-lldb-backtrace.log"
    log "  a failure was seen — re-running [gpu] under lldb for a backtrace"
    run_logged "$LLDB_LOG" "lldb tests [gpu]" \
        lldb --batch -o run -o 'thread backtrace all' -o 'register read' -o quit \
             -- "$TESTS_BIN" "[gpu]"
    fact lldb_backtrace "51-lldb-backtrace.log"
fi

# ===========================================================================
section "STAGE 10 — CI parity (conan create) [optional]"
# ===========================================================================
if [ "$CI_PARITY" -eq 1 ] && [ "$PROBE_ONLY" -eq 0 ]; then
    CREATE_LOG="$OUT_DIR/60-conan-create.log"
    : > "$CREATE_LOG"
    cd "$REPO_ROOT" || exit 1
    # shellcheck disable=SC2086
    run_logged "$CREATE_LOG" "conan create (exact CI command)" \
        conan create . --build=missing \
            -pr:h=macos-clang21-arm64 -pr:h=build-$BUILD_TYPE -pr:h=options-ci \
            -pr:b=macos-clang21-arm64 -pr:b=build-$BUILD_TYPE -pr:b=options-ci \
            -s:h "&:build_type=$CMAKE_BUILD_TYPE" -s:h "*:build_type=Release" \
            -o PYCANHA_OPTION_USE_MKL=False
    if [ $? -eq 0 ]; then
        verdict ci_parity PASS "conan create succeeded (library + tests + test_package)"
    else
        verdict ci_parity FAIL "conan create failed — see 60-conan-create.log"
    fi
else
    verdict ci_parity SKIP "not requested (--ci-parity)"
fi

# ===========================================================================
section "STAGE 11 — Python bindings smoke test [optional]"
# ===========================================================================
if [ "$DO_PYTHON" -eq 1 ]; then
    PY_LOG="$OUT_DIR/70-python.log"
    : > "$PY_LOG"
    run_logged "$PY_LOG" "python3 -c import pycanha_core" python3 - <<'PY'
import sys
print("python:", sys.version)
try:
    import pycanha_core
except Exception as exc:                      # noqa: BLE001 - diagnostic script
    print("import pycanha_core FAILED:", type(exc).__name__, exc)
    raise SystemExit(1)
print("pycanha_core:", getattr(pycanha_core, "__file__", "?"))
print("version:", getattr(pycanha_core, "__version__", "<none>"))
print("top-level names:", sorted(n for n in dir(pycanha_core) if not n.startswith("_")))
rad = getattr(pycanha_core, "radiative", None)
if rad is None:
    print("no `radiative` submodule — the wheel was built without raytracing")
    raise SystemExit(2)
print("radiative names:", sorted(n for n in dir(rad) if not n.startswith("_")))
for name in ("is_available", "enumerate_devices"):
    fn = getattr(rad, name, None)
    print(f"{name}:", "<missing>" if fn is None else fn)
if getattr(rad, "is_available", None) is not None:
    print("is_available() ->", rad.is_available())
if getattr(rad, "enumerate_devices", None) is not None:
    for dev in rad.enumerate_devices():
        print("  device:", {a: getattr(dev, a) for a in
                            ("name", "ray_tracing", "software",
                             "max_dispatch_rays", "index")
                            if hasattr(dev, a)})
PY
    PY_RC=$?
    case "$PY_RC" in
        0) verdict python PASS "pycanha_core imported and reported its devices" ;;
        2) verdict python FAIL "pycanha_core has no radiative submodule" ;;
        *) verdict python FAIL "pycanha_core import/probe failed — see 70-python.log" ;;
    esac
else
    verdict python SKIP "not requested (--python)"
fi

# ===========================================================================
section "SUMMARY"
# ===========================================================================
SUMMARY="$OUT_DIR/00-SUMMARY.txt"

# Overall verdict. "No RT device" is deliberately INCONCLUSIVE and never PASS:
# the suite goes green by skipping.
OVERALL="PASS"
if [ "$(verdict_of rt_device)" != "PASS" ]; then
    OVERALL="INCONCLUSIVE"
fi
while IFS="$(printf '\t')" read -r k v rest; do
    [ -n "$k" ] || continue
    if [ "$v" = "FAIL" ]; then OVERALL="FAIL"; fi
done < "$WORK/verdicts.txt"
if [ "$PROBE_ONLY" -eq 1 ] && [ "$OVERALL" = "PASS" ]; then
    OVERALL="PROBE_ONLY_OK"
fi

{
    echo "==============================================================================="
    echo "pycanha-core — macOS / Metal ray-tracing diagnostics"
    echo "==============================================================================="
    echo "  overall verdict : $OVERALL"
    echo "  machine         : $(sysctl -n hw.model 2>/dev/null) / $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    echo "  GPU             : $(grep -m1 'Selected RT-capable device:' "$DEVICE_LOG" 2>/dev/null | sed 's/.*device: //')"
    echo "  macOS           : $(sw_vers -productVersion 2>/dev/null) ($(sw_vers -buildVersion 2>/dev/null))"
    echo "  commit          : $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null) on $(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    echo "  dirty files     : $(cd "$REPO_ROOT" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    echo "  build type      : $CMAKE_BUILD_TYPE"
    echo "  elapsed         : $(( $(date +%s) - RUN_START_EPOCH ))s"
    echo
    echo "-------------------------------------------------------------------------------"
    echo "STAGE VERDICTS"
    echo "-------------------------------------------------------------------------------"
    while IFS="$(printf '\t')" read -r k v rest; do
        [ -n "$k" ] || continue
        printf '  %-16s %-8s %s\n' "$k" "$v" "$rest"
    done < "$WORK/verdicts.txt"
    echo
    echo "-------------------------------------------------------------------------------"
    echo "HOW TO READ THIS"
    echo "-------------------------------------------------------------------------------"
    echo "  rt_device      Does a device pass pycanha's own gate,"
    echo "                 supportsRaytracing && supportsFamily:Apple9? If not,"
    echo "                 EVERY [gpu] test skips and a green ctest means nothing."
    echo "  split_atomic   Runs the split 64-bit counter (fp_add_raw in"
    echo "                 kernels/common.slang) on the real GPU under contention"
    echo "                 and checks the carries fold exactly."
    echo "  ray_query      Builds a one-triangle acceleration structure and traces"
    echo "                 two rays, independent of pycanha. Also reports whether"
    echo "                 MSL's front-facing predicate agrees with the geometric"
    echo "                 substitute common.slang uses on Metal."
    echo "  kernel_codegen Counts atomic adds in the generated exchange MSL. Zero"
    echo "                 means the deposit silently became a non-atomic"
    echo "                 read-modify-write, which still compiles and still runs."
    echo "  gpu_executed   Whether the [gpu] cases really ran or quietly skipped."
    echo "  determinism    Two runs of the same fixed seed, same machine. Integer"
    echo "                 accumulators make equality a guarantee, so a difference"
    echo "                 is a race / barrier / residency bug, never noise."
    echo "  conservation   The exchange row balance. Must be exactly 0."
    echo "  cross_backend  Byte-for-byte against a Vulkan machine's dump."
    echo "  validation     The suite under Metal API + shader validation, which is"
    echo "                 what catches a wrong per-kernel buffer index."
    echo
    echo "-------------------------------------------------------------------------------"
    echo "KEY FACTS"
    echo "-------------------------------------------------------------------------------"
    while IFS="$(printf '\t')" read -r k v; do
        [ -n "$k" ] || continue
        printf '  %-24s %s\n' "$k" "$v"
    done < "$FACTS"
    echo
    echo "-------------------------------------------------------------------------------"
    echo "FILES IN THIS BUNDLE"
    echo "-------------------------------------------------------------------------------"
    echo "  00-SUMMARY.txt                this file"
    echo "  00-console.log                the run's console transcript"
    echo "  summary.json                  machine-readable verdicts + facts"
    echo "  01-environment.txt            machine, Xcode, toolchain, git state"
    echo "  02-metal-device-probe.txt/.mm device capability report + split-counter run"
    echo "  03-metal-rayquery-probe.txt/.mm  acceleration structure + ray query"
    echo "  04-atomics64-matrix.txt       64-bit atomic compile matrix"
    echo "  10-conan-install.log          dependency resolution"
    echo "  11-cmake-configure.log        configure (slangc fetch, Metal probe)"
    echo "  12-build.log                  full compile/link output"
    echo "  12-build-diagnostics.txt      errors + deduplicated warnings"
    echo "  13-kernels/                   generated MSL, reflection, metallib, headers"
    echo "  13-kernel-analysis.txt        atomic census, buffer indices, disassembly"
    echo "  20-test-inventory.txt         ctest -N listings and Catch2 tags"
    echo "  20-ctest.log                  full suite"
    echo "  21-ctest-failures.txt         failing cases (if any)"
    echo "  22-tests-radiative-verbose.log  per-assertion values"
    echo "  23-tests-gpu-verbose.log      per-assertion values, [gpu] only"
    echo "  24-gpu-skip-analysis.txt      did the GPU cases really run?"
    echo "  30-bitdiff/run1, run2         fixed-seed dumps"
    echo "  31-bitdiff-selfcompare.txt    determinism"
    echo "  32-bitdiff-vs-reference.txt   cross-backend (with --reference-dir)"
    echo "  40-metal-validation.log       API + shader validation pass"
    echo "  50-crash-reports/             .ips reports from this run"
    echo "  51-lldb-backtrace.log         backtrace (only if something failed)"
    echo "  60-conan-create.log           CI parity run (with --ci-parity)"
    echo "  70-python.log                 wheel probe (with --python)"
    echo
    echo "-------------------------------------------------------------------------------"
    echo "SUGGESTED NEXT ACTIONS"
    echo "-------------------------------------------------------------------------------"
    if [ "$(verdict_of rt_device)" != "PASS" ]; then
        echo "  * This machine is not Apple9 / not RT-capable. Nothing about kernel"
        echo "    RESULTS can be learned here — only compile coverage. Find an M3/M4"
        echo "    (or A17 Pro) machine before drawing any conclusion."
    fi
    if [ "$(verdict_of build)" = "FAIL" ]; then
        echo "  * Start at 12-build-diagnostics.txt: the error list is at the top."
    fi
    if [ "$(verdict_of build)" = "WARN" ]; then
        echo "  * The build only succeeded with -Werror off. The deduplicated"
        echo "    warning list in 12-build-diagnostics.txt is the CI blocker;"
        echo "    fix the causes rather than suppressing them."
    fi
    if [ "$(verdict_of split_atomic)" = "FAIL" ]; then
        echo "  * fp_add_raw's split counter loses carries on this GPU. That is"
        echo "    kernels/common.slang; the probe source in 02-...mm reproduces it"
        echo "    in ~40 lines with no pycanha involved."
    fi
    if [ "$(verdict_of kernel_codegen)" = "FAIL" ]; then
        echo "  * The generated exchange MSL has no atomic add. Read"
        echo "    13-kernels/exchange.metal around the deposit; roadmap 22 §9"
        echo "    describes exactly this regression."
    fi
    if [ "$(verdict_of determinism)" = "FAIL" ]; then
        echo "  * Same seed, same machine, different answers: look for a missing"
        echo "    barrier between the accumulate dispatch and the readback, or a"
        echo "    resource missing from useResource: in mtl_scene/mtl_accum."
    fi
    if [ "$(verdict_of validation)" = "FAIL" ] || [ "$(verdict_of validation)" = "WARN" ]; then
        echo "  * The validation layer spoke: 40-metal-validation.log. Wrong"
        echo "    per-kernel buffer indices and non-resident resources show up"
        echo "    here and nowhere else."
    fi
    if [ "$(verdict_of cross_backend)" = "SKIP" ]; then
        echo "  * For the cross-backend check, copy 30-bitdiff/run1/*.txt to the"
        echo "    Windows/Linux machine, run ./tests \"[bitdiff]\" there at the same"
        echo "    commit, and diff. They must match byte for byte."
    fi
    echo
} > "$SUMMARY" 2>&1

# --- machine-readable summary ---------------------------------------------
{
    echo "{"
    echo "  \"overall\": \"$OVERALL\","
    echo "  \"generated\": \"$(date '+%Y-%m-%dT%H:%M:%S%z')\","
    echo "  \"elapsed_seconds\": $(( $(date +%s) - RUN_START_EPOCH )),"
    echo "  \"verdicts\": {"
    first=1
    while IFS="$(printf '\t')" read -r k v rest; do
        [ -n "$k" ] || continue
        [ $first -eq 1 ] || echo ","
        first=0
        printf '    "%s": {"result": "%s", "detail": "%s"}' \
            "$k" "$v" "$(printf '%s' "$rest" | json_escape)"
    done < "$WORK/verdicts.txt"
    echo
    echo "  },"
    echo "  \"facts\": {"
    first=1
    while IFS="$(printf '\t')" read -r k v; do
        [ -n "$k" ] || continue
        [ $first -eq 1 ] || echo ","
        first=0
        printf '    "%s": "%s"' "$k" "$(printf '%s' "$v" | json_escape)"
    done < "$FACTS"
    echo
    echo "  }"
    echo "}"
} > "$OUT_DIR/summary.json" 2>&1

cat "$SUMMARY"

# --- bundle ----------------------------------------------------------------
rm -rf "$WORK/facts.txt" 2>/dev/null
TARBALL="$OUT_DIR.tar.gz"
( cd "$(dirname "$OUT_DIR")" && tar czf "$TARBALL" "$(basename "$OUT_DIR")" ) 2>/dev/null

log ""
log "Diagnostics directory : $OUT_DIR"
log "Bundle to send back   : $TARBALL"
log "Overall verdict       : $OVERALL"

case "$OVERALL" in
    PASS|PROBE_ONLY_OK) exit 0 ;;
    INCONCLUSIVE)       exit 1 ;;
    *)                  exit 2 ;;
esac
