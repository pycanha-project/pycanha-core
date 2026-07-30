// Self-contained ray-query + 64-bit-atomic probe for the Metal backend spike.
//
// This is deliberately independent of kernels/common.slang: it answers "does
// the Metal RUNTIME execute inline ray tracing and 64-bit buffer atomics
// correctly on this device" with the smallest possible amount of moving parts,
// so a failure here points at the driver/hardware rather than at the radiative
// engine. Compiled through the same pycanha_add_slang_kernel(TARGET metal)
// rule as the production kernels, so it also exercises the build path.
//
// The scene is a unit quad at z = 0 built from two triangles (prim 0 covers
// the lower-left half, prim 1 the upper-right). Even threads trace a ray that
// must hit prim 0; odd threads trace one that must miss entirely.

module probe;

// Threads per dispatch. Must match PROBE_THREADS in main.mm — the host checks
// exact sums, so a mismatch would look like a GPU fault rather than a typo.
public static const uint PROBE_THREADS = 128u;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
// Per-thread outcome: 0 = miss, otherwise committed primitive index + 1.
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> outcomes;
// Deliberately 32-bit. This probe used to accumulate into a uint64_t to prove
// 64-bit atomic adds, but the Metal compiler rejects them outright
// (_valid_fetch_add_type excludes ulong), so the question is a COMPILE-time one
// and belongs in probe_atomics64.sh, not in a runtime kernel that would then
// fail to build and take the ray-tracing checks down with it.
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> counter;

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 gid: SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= PROBE_THREADS) {
        return;
    }

    // Even threads aim into the quad, odd threads well outside it.
    bool aim_at_quad = (i & 1u) == 0u;
    float xy = aim_at_quad ? 0.25 : 5.0;

    RayDesc ray;
    ray.Origin = float3(xy, xy, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 100.0;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    q.Proceed();

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        outcomes[i] = q.CommittedPrimitiveIndex() + 1u;
    } else {
        outcomes[i] = 0u;
    }

    InterlockedAdd(counter[0], 1u);
}
