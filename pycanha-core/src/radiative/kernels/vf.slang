// VF kernel: geometric view factors, no materials. One trace per ray; score
// count[row][hit_slot] += 1 on the first committed hit. Planet faces DO
// score (spacecraft->planet view factors feed albedo/planet-IR
// post-processing). A miss means the ray went to space — reconstructed on
// the CPU as rays - row_sum, so this kernel has no space buffer.
// Normalization on the CPU: VF[i][j] = count[i][j] / rays_i.

import common;

// Plain u32 counting cells (counts need no fixed-point scale); the host
// guards rays_per_face < 2^31 so cells cannot overflow.
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> acc;

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 gid: SV_DispatchThreadID) {
    uint ray_local = gid.x;
    uint e = gid.y;
    if (ray_local >= pc.rays_this_chunk || e >= pc.num_emitters) {
        return;
    }
    // The emitter slot comes from the emitters buffer, NEVER from gid
    // arithmetic over slots (emitter subsets have non-contiguous slots);
    // ray_index includes the chunk offset so results are invariant under
    // dispatch chunking.
    uint face_slot = emitters[e];
    uint ray_index = pc.ray_index_offset + ray_local;
    Rng rng = rng_init(face_slot, ray_index, pc.batch_seed);
    uint row = face_slot - pc.row_offset;  // block-relative accumulator row

    EmittedRay ray = emit_from_face(face_slot, rng);
    Hit hit = trace_ray(ray.origin, ray.direction);
    if (!hit.miss) {
        InterlockedAdd(acc[row * pc.num_face_slots + hit.slot], 1u);
    }
}
