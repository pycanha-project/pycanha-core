#!/usr/bin/env bash
# Does Metal have 64-bit buffer atomics? (macOS + Xcode Metal toolchain only.)
#
# Why this exists: the radiative kernels deposit energy into 64-bit fixed-point
# cells with an atomic add. Metal Shading Language has NO 64-bit buffer atomic
# of any kind — not add, min/max, load/store or compare-exchange, on any -std —
# which is why kernels/common.slang splits the cell into a lo/hi pair of 32-bit
# counters on that backend. That is surprising (Apple's own feature tables
# suggest the opposite from GPU family Apple9 onwards), and it is a compiler
# limitation, so re-checking after a Metal toolchain upgrade is cheap: run this.
# If a future toolchain compiles the ulong rows, the split path in
# common.slang can collapse back to a single native atomic.
#
# This probe answers "does it COMPILE" only. A probe that claims 64-bit
# accumulation WORKS must force carries out of the low half and leave a
# non-zero low remainder — e.g. accumulating 128 x 0x0000000180000007, which
# produces 64 carries and a low word of 896. Adding a whole multiple of 2^32
# passes even with no 64-bit support at all.
#
# Usage: test/tools/probe_metal_atomics64.sh

set -u

if ! command -v xcrun > /dev/null 2>&1; then
    echo "xcrun not found — this probe only runs on macOS" >&2
    exit 1
fi
if ! xcrun metal --version > /dev/null 2>&1; then
    echo "The Metal shader compiler is missing. Install it with:" >&2
    echo "    xcodebuild -downloadComponent MetalToolchain" >&2
    exit 1
fi

echo "Metal compiler: $(xcrun metal --version 2>&1 | head -1)"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# One kernel body per feature. The control row must always pass; a FAIL there
# means the probe itself is broken, not the hardware.
emit_kernel() {
    local feature="$1"
    echo '#include <metal_stdlib>'
    echo 'using namespace metal;'
    case "$feature" in
        u32_add)
            echo 'kernel void probe(device atomic_uint* buf [[buffer(0)]]) {'
            echo '    atomic_fetch_add_explicit(buf, 1u, memory_order_relaxed);'
            echo '}'
            ;;
        u64_add_cast)  # exactly what slangc emits for InterlockedAdd(u64)
            echo 'kernel void probe(device ulong* buf [[buffer(0)]]) {'
            echo '    atomic_fetch_add_explicit((device atomic_ulong*)buf, 1ul,'
            echo '                              memory_order_relaxed);'
            echo '}'
            ;;
        u64_add_template)
            echo 'kernel void probe(device atomic<ulong>* buf [[buffer(0)]]) {'
            echo '    atomic_fetch_add_explicit(buf, 1ul, memory_order_relaxed);'
            echo '}'
            ;;
        u64_max)
            echo 'kernel void probe(device ulong* buf [[buffer(0)]]) {'
            echo '    atomic_fetch_max_explicit((device atomic_ulong*)buf, 1ul,'
            echo '                              memory_order_relaxed);'
            echo '}'
            ;;
        u64_load_store)
            echo 'kernel void probe(device ulong* buf [[buffer(0)]]) {'
            echo '    ulong v = atomic_load_explicit((device atomic_ulong*)buf,'
            echo '                                   memory_order_relaxed);'
            echo '    atomic_store_explicit((device atomic_ulong*)buf, v + 1ul,'
            echo '                          memory_order_relaxed);'
            echo '}'
            ;;
        u64_cas)
            echo 'kernel void probe(device ulong* buf [[buffer(0)]]) {'
            echo '    ulong expected = 0ul;'
            echo '    atomic_compare_exchange_weak_explicit('
            echo '        (device atomic_ulong*)buf, &expected, 1ul,'
            echo '        memory_order_relaxed, memory_order_relaxed);'
            echo '}'
            ;;
        *)
            echo "unknown feature $feature" >&2
            return 1
            ;;
    esac
}

features="u32_add u64_add_cast u64_add_template u64_max u64_load_store u64_cas"
# Empty entry = the toolchain's default -std.
standards="metal4.0 metal3.2 metal3.1 metal3.0 "

printf '\n%-18s' "feature"
for std in $standards; do
    printf '%-11s' "${std:-default}"
done
printf '\n'

any_u64_ok=0
for feature in $features; do
    printf '%-18s' "$feature"
    emit_kernel "$feature" > "$work/probe.metal"
    for std in $standards; do
        if [ -z "$std" ]; then
            xcrun metal -c "$work/probe.metal" -o "$work/probe.air" \
                > "$work/probe.log" 2>&1
            rc=$?
        else
            xcrun metal -std="$std" -c "$work/probe.metal" -o "$work/probe.air" \
                > "$work/probe.log" 2>&1
            rc=$?
        fi
        if [ "$rc" -eq 0 ]; then
            printf '%-11s' "ok"
            case "$feature" in
                u64_*) any_u64_ok=1 ;;
            esac
        else
            printf '%-11s' "FAIL"
        fi
    done
    printf '\n'
done

echo
if [ "$any_u64_ok" -eq 1 ]; then
    echo "A 64-bit atomic now compiles: re-evaluate the split counter in"
    echo "pycanha-core/src/radiative/kernels/common.slang (fp_add_raw)."
else
    echo "No 64-bit buffer atomic compiles — the split counter is still needed."
fi
