// Exchange kernel: multi-bounce MCRT radiative exchange factors for one
// band (IR or solar, selected by FLAG_SOLAR_BAND — the solar band uses the
// solar property triplet at EVERY bounce). Each ray starts with energy 1;
// absorption is deterministic (deposit E*a, continue with E*(1-a)), the
// specular/transmission/diffuse choice is stochastic. Normalization on the
// CPU: B[i][j] = deposited[i][j] / rays_i.
//
// Exact energy accounting: every ray also carries its remaining energy as a
// fixed-point integer, deposits are the DIFFERENCE of consecutive
// fixed-point values and every termination path flushes the remaining
// balance into a bucket (acc / space_vec / lost_vec). Per row,
// deposits + space + lost == rays * fp_scale holds exactly in wrapping u64
// arithmetic — the CI conservation assert. Russian-roulette boosts withdraw
// their fixed-point gain from lost_vec (wrapping subtraction) and kills
// deposit the balance there, so the identity survives RR while the estimate
// stays unbiased (the two adjustments cancel in expectation, which is why
// lost_vec is read back as a SIGNED total).

import common;

[[vk::binding(10, 0)]] RWStructuredBuffer<uint64_t> acc;
[[vk::binding(11, 0)]] RWStructuredBuffer<uint64_t> space_vec;
[[vk::binding(12, 0)]] RWStructuredBuffer<uint64_t> lost_vec;

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 gid: SV_DispatchThreadID) {
    uint ray_local = gid.x;
    uint e = gid.y;
    if (ray_local >= pc.rays_this_chunk || e >= pc.num_emitters) {
        return;
    }
    // The emitter slot comes from the emitters buffer, NEVER from gid
    // arithmetic over slots; ray_index includes the chunk offset so results
    // are invariant under dispatch chunking.
    uint face_slot = emitters[e];
    uint ray_index = pc.ray_index_offset + ray_local;
    Rng rng = rng_init(face_slot, ray_index, pc.batch_seed);
    uint row = face_slot - pc.row_offset;  // block-relative accumulator row

    EmittedRay ray = emit_from_face(face_slot, rng);
    float3 origin = ray.origin;
    float3 dir = ray.direction;
    float energy = 1.0;
    // fp_scale is a power of two, so to_fp(1.0) equals it exactly and the
    // host can compute rays * fp_scale without rounding.
    uint64_t fp_remaining = to_fp(1.0);

    for (uint bounce = 0u; bounce < pc.max_bounces; ++bounce) {
        Hit hit = trace_ray(origin, dir);
        if (hit.miss) {
            fp_add_raw(space_vec, row, fp_remaining);  // escaped to space
            return;
        }
        uint cell = row * pc.num_face_slots + hit.slot;
        if ((face_flags[hit.slot] & FACE_FLAG_PLANET) != 0u) {
            // The planet absorbs every band; its radiance is CPU
            // post-processing over the VF matrix, not a kernel concern.
            fp_add_raw(acc, cell, fp_remaining);
            return;
        }
        if ((face_flags[hit.slot] & FACE_FLAG_ACTIVE) == 0u) {
            fp_add_raw(lost_vec, row, fp_remaining);  // modeling convention
            return;
        }

        float a;
        float s;
        float t;
        if (!band_properties(hit.slot, a, s, t)) {
            // No material assigned: blackbody, absorbs the remainder.
            fp_add_raw(acc, cell, fp_remaining);
            return;
        }
        // Diffuse reflectance is the clamped remainder; never renormalize
        // upward — the host has already validated a + s + t <= 1.
        float d = max(1.0 - a - s - t, 0.0);

        // Deposit BEFORE the RR test and before event selection (reordering
        // would bias the estimate).
        float energy_after = energy * (1.0 - a);
        uint64_t fp_after = to_fp(energy_after);
        fp_add_raw(acc, cell, fp_remaining - fp_after);
        fp_remaining = fp_after;
        energy = energy_after;
        if (fp_remaining == uint64_t(0)) {
            return;  // nothing representable left to distribute
        }

        // Russian roulette below the threshold — unbiased termination.
        if (energy < pc.energy_threshold) {
            float u_rr = rng_next(rng);
            if (u_rr * pc.energy_threshold < energy) {
                uint64_t fp_boost = to_fp(pc.energy_threshold);
                // Wrapping subtraction: the boost's fixed-point gain is
                // withdrawn from the lost bucket to keep the row balance
                // exact.
                fp_add_raw(lost_vec, row, fp_remaining - fp_boost);
                fp_remaining = fp_boost;
                energy = pc.energy_threshold;
            } else {
                fp_add_raw(lost_vec, row, fp_remaining);
                return;
            }
        }

        // Event selection among the non-absorbed part. Do not normalize by
        // (1 - a): s + t + d may fall below it only through float rounding.
        float u_event = rng_next(rng) * (s + t + d);
        if (u_event < s) {
            dir = normalize(reflect(dir, hit.normal));
            origin = offset_origin(hit.position, hit.normal);
        } else if (u_event < s + t) {
            // Thin-surface transmission: same direction, far side.
            origin = offset_origin(hit.position, -hit.normal);
        } else {
            dir = cosine_hemisphere(hit.normal, rng);
            origin = offset_origin(hit.position, hit.normal);
        }
    }
    // The max_bounces safety bound killed the ray.
    fp_add_raw(lost_vec, row, fp_remaining);
}
