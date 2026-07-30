#!/usr/bin/env bash
# Diagnostic: what 64-bit buffer atomics does this Metal toolchain actually
# accept? Run on the Mac, from anywhere:
#
#     bash test/metal_spike/probe_atomics64.sh
#
# Context: the spike build failed with "no matching function for call to
# 'atomic_fetch_add_explicit'" on a `device atomic_ulong*`, with the note that
# `_valid_fetch_add_type<device unsigned long *, void>` is not satisfied. That
# is what slangc emits for InterlockedAdd on a uint64_t buffer, i.e. for
# fp_add/fp_add_raw in kernels/common.slang. Apple's Metal Feature Set Tables
# say the full set of 64-bit atomics is available from GPU family Apple9, so
# either a newer -std unlocks it or the tables mean something narrower.
#
# The answer decides how the exchange and solar kernels deposit energy:
#   - fetch_add works under some -std   -> pass that -std, nothing else changes.
#   - only compare_exchange works       -> CAS loop inside fp_add; no change to
#                                          buffer layout or host readback.
#   - only 32-bit atomics work          -> lo/hi split-counter emulation.
# This script prints which of those is true. It writes only to a temp dir.

set -u

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "toolchain: $(xcrun metal --version 2>&1 | head -1)"
echo

# --- the feature snippets ------------------------------------------------
# Each is a complete kernel so a failure is attributable to the one operation.

cat > "$work/control_u32_add.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
[[kernel]] void k(device uint *p [[buffer(0)]]) {
    atomic_fetch_add_explicit((device atomic_uint *)p, 1u, memory_order_relaxed);
}
EOF

# Exactly what slangc emits today.
cat > "$work/u64_add_cstyle.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
[[kernel]] void k(device ulong *p [[buffer(0)]]) {
    atomic_fetch_add_explicit((device atomic_ulong *)p, 1ul, memory_order_relaxed);
}
EOF

cat > "$work/u64_add_template.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
[[kernel]] void k(device ulong *p [[buffer(0)]]) {
    atomic_fetch_add_explicit((device metal::atomic<ulong> *)p, 1ul,
                              memory_order_relaxed);
}
EOF

cat > "$work/u64_max.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
[[kernel]] void k(device ulong *p [[buffer(0)]]) {
    atomic_fetch_max_explicit((device atomic_ulong *)p, 7ul, memory_order_relaxed);
}
EOF

cat > "$work/u64_load_store.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
[[kernel]] void k(device ulong *p [[buffer(0)]]) {
    ulong v = atomic_load_explicit((device atomic_ulong *)p, memory_order_relaxed);
    atomic_store_explicit((device atomic_ulong *)p, v + 1ul, memory_order_relaxed);
}
EOF

# The fallback that would let fp_add keep a single u64 cell and identical
# buffer layout: a compare-exchange retry loop.
cat > "$work/u64_cas.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
[[kernel]] void k(device ulong *p [[buffer(0)]]) {
    device atomic_ulong *a = (device atomic_ulong *)p;
    ulong expected = atomic_load_explicit(a, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        a, &expected, expected + 3ul, memory_order_relaxed,
        memory_order_relaxed)) {
    }
}
EOF

# --- which -std values does this toolchain know? ------------------------
candidates="metal4.1 metal4.0 metal3.2 metal3.1 metal3.0 default"
stds=""
for s in $candidates; do
    if [ "$s" = default ]; then
        stds="$stds default"
        continue
    fi
    if xcrun metal -std="$s" -c "$work/control_u32_add.metal" \
            -o "$work/probe.air" >/dev/null 2>&1; then
        stds="$stds $s"
    fi
done
echo "usable -std values:$stds"
echo

# --- the matrix ---------------------------------------------------------
printf '%-22s' "feature"
for s in $stds; do printf '%-12s' "$s"; done
printf '\n'

for f in control_u32_add u64_add_cstyle u64_add_template u64_max u64_load_store u64_cas; do
    printf '%-22s' "$f"
    for s in $stds; do
        if [ "$s" = default ]; then
            xcrun metal -c "$work/$f.metal" -o "$work/out.air" >"$work/err.txt" 2>&1
        else
            xcrun metal -std="$s" -c "$work/$f.metal" -o "$work/out.air" \
                >"$work/err.txt" 2>&1
        fi
        if [ $? -eq 0 ]; then printf '%-12s' "ok"; else printf '%-12s' "FAIL"; fi
    done
    printf '\n'
done

echo
echo "First error for the case that matters (u64_add_cstyle, default -std):"
xcrun metal -c "$work/u64_add_cstyle.metal" -o "$work/out.air" 2>&1 \
    | grep -E "error|note" | head -4
