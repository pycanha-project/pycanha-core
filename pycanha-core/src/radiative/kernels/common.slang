// Shared module of every radiative kernel: RNG, emission sampling, hit
// resolution, fixed-point deposits, shared scene bindings and push constants.
//
// Determinism contract: the per-ray random stream depends only on
// (face_slot, ray_index, batch_seed), and the number of RNG draws per step
// is FROZEN (see the draw counts noted at each sampling function) — never
// insert a draw, only append at the end of a step, or every golden-value
// test and the CPU reference implementation silently break.

module common;

// ---------------------------------------------------------------------------
// Push constants — one struct for all kernels. Dispatches are split into
// ray-index chunks (GPU watchdog limits + progress reporting), which is why
// the chunk fields exist; chunking never changes results because the RNG is
// keyed on the global ray index. The C++ mirror struct on the host must
// match this byte-for-byte (scalar layout, 64 B).
// ---------------------------------------------------------------------------
public struct Push {
    public uint row_offset;        // first emitter row of this block
    public uint num_emitters;      // emitters in this dispatch
    public uint rays_this_chunk;   // rays per face in this chunk
    public uint ray_index_offset;  // first global ray index of this chunk
    public uint batch_seed;        // settings.seed + batch_index
    public uint max_bounces;
    public uint flags;             // bit0: normal-emission mode
    public uint num_face_slots;    // total face slots (accumulator row stride)
    public float energy_threshold;
    public float fp_scale;         // fixed-point scale S of the deposits
    public float inv_fp_scale;     // 1/S
    public float ray_tmin_scale;   // self-intersection epsilon scale
    // Solar kernel only (unit, sun -> scene). Stored as three scalars rather
    // than a float3 on purpose: float3 is 12 bytes under Vulkan scalar block
    // layout but 16 bytes (16-aligned) in Metal Shading Language, which would
    // size this struct at 64 bytes for SPIR-V and 80 for Metal. Scalars lay
    // out identically under both rules, so a single 64-byte C++ mirror serves
    // every backend. Read them through sun_direction().
    public float sun_dir_x;
    public float sun_dir_y;
    public float sun_dir_z;
    public float _pad;
};

[[vk::push_constant]]
public Push pc;

// Sun direction as a vector (see the Push field comment for why the three
// components are stored as scalars).
public float3 sun_direction() {
    return float3(pc.sun_dir_x, pc.sun_dir_y, pc.sun_dir_z);
}

public static const uint FLAG_NORMAL_EMISSION = 1u;
// Selects the solar property triplet (alpha_sol/spec_sol/tau_sol) instead of
// the IR one. A push-constant flag keeps one exchange pipeline for both
// bands; the branch is dispatch-uniform, so it costs nothing on the GPU.
public static const uint FLAG_SOLAR_BAND = 2u;

// ---------------------------------------------------------------------------
// Shared scene bindings (set 0, bindings 0-9). Kernel-specific accumulator
// buffers start at binding 10 in the kernel files.
// ---------------------------------------------------------------------------

// One TLAS instance per rigid part (80 bytes, scalar layout). The host
// writes the SAME transform here and into the TLAS instance on every
// commit(): the TLAS positions geometry for ray traversal, but emission
// sampling needs the transform explicitly — and it must be read per
// dispatch, never baked into the vertex buffers, because parts move
// between snapshots.
public struct InstanceData {
    public uint64_t vertices_addr;  // devaddr float3[] (12 B stride, scalar)
    public uint64_t indices_addr;   // devaddr uint3[]
    public uint64_t tri_face_addr;  // devaddr uint[] (even side-1 slot / tri)
    public uint part_id;
    public uint flags;              // bit0: is_planet_part
    // Part -> world as three explicit row vectors (xyz = rotation row,
    // w = translation component). An array of float4 has an unambiguous
    // memory layout, unlike matrix types whose row/column-major storage
    // depends on compiler settings.
    public float4 tf_rows[3];
};

// world = R * p + t, with R rows / t components taken from tf_rows.
public float3 transform_point(InstanceData inst, float3 p) {
    return float3(dot(inst.tf_rows[0].xyz, p) + inst.tf_rows[0].w,
                  dot(inst.tf_rows[1].xyz, p) + inst.tf_rows[1].w,
                  dot(inst.tf_rows[2].xyz, p) + inst.tf_rows[2].w);
}

// Rotation only (directions/normals; rigid transforms preserve lengths).
public float3 transform_direction(InstanceData inst, float3 d) {
    return float3(dot(inst.tf_rows[0].xyz, d), dot(inst.tf_rows[1].xyz, d),
                  dot(inst.tf_rows[2].xyz, d));
}

[[vk::binding(0, 0)]] public RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] public StructuredBuffer<InstanceData> instance_data;
[[vk::binding(2, 0)]] public StructuredBuffer<float> materials;  // 6*M floats
[[vk::binding(3, 0)]] public StructuredBuffer<int> face_material;
[[vk::binding(4, 0)]] public StructuredBuffer<uint> face_flags;
[[vk::binding(5, 0)]] public StructuredBuffer<uint> emitters;  // this block
[[vk::binding(6, 0)]] public StructuredBuffer<uint> emit_tri_offset;
[[vk::binding(7, 0)]] public StructuredBuffer<uint> emit_tri_part;
[[vk::binding(8, 0)]] public StructuredBuffer<uint> emit_tri_prim;
[[vk::binding(9, 0)]] public StructuredBuffer<float> emit_cum_area;

public static const uint FACE_FLAG_ACTIVE = 1u;
public static const uint FACE_FLAG_PLANET = 2u;
public static const uint INSTANCE_FLAG_PLANET = 1u;

// ---------------------------------------------------------------------------
// RNG — counter-based and grouping-invariant (PCG-XSH-RR style hash +
// stream). Streams depend ONLY on (face_slot, ray_index, batch_seed) —
// never on thread id, dispatch shape, tile size or emitter order — so any
// chunking/tiling of the dispatch reproduces bit-identical results.
// ---------------------------------------------------------------------------
public struct Rng {
    public uint state;
};

public uint pcg_hash(uint x) {
    x = x * 747796405u + 2891336453u;
    uint w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (w >> 22u) ^ w;
}

public Rng rng_init(uint face_slot, uint ray_index, uint batch_seed) {
    Rng r;
    r.state = pcg_hash(face_slot ^ pcg_hash(ray_index ^ pcg_hash(batch_seed)));
    return r;
}

// Uniform in [0, 1) — 24-bit mantissa, never 1.0.
public float rng_next(inout Rng r) {
    r.state = r.state * 747796405u + 2891336453u;
    uint w = ((r.state >> ((r.state >> 28u) + 4u)) ^ r.state) * 277803737u;
    w = (w >> 22u) ^ w;
    return float(w >> 8) * (1.0 / 16777216.0);
}

// ---------------------------------------------------------------------------
// Fixed-point deposit: float energy is converted to an integer once and
// accumulated with a 64-bit atomic add. Integer adds are associative (float
// atomics are not), which is what makes results independent of thread
// scheduling and accumulator layout. No zero-check: a zero add is harmless
// and branchless is faster.
// ---------------------------------------------------------------------------
public void fp_add(RWStructuredBuffer<uint64_t> buf, uint idx, float value) {
    InterlockedAdd(buf[idx], uint64_t(value * pc.fp_scale + 0.5));
}

// Fixed-point representation of a remaining-energy float. Monotone
// non-decreasing in `energy` (f32 multiply, +0.5 and the u64 truncation all
// preserve order), which is what lets deposits be computed as differences of
// consecutive values without ever underflowing.
public uint64_t to_fp(float energy) {
    return uint64_t(energy * pc.fp_scale + 0.5);
}

// Raw integer add for pre-converted fixed-point amounts (deposit
// differences and terminal balance flushes). u64 arithmetic wraps, which the
// exchange kernel exploits for the Russian-roulette boost withdrawal.
public void fp_add_raw(RWStructuredBuffer<uint64_t> buf, uint idx,
                       uint64_t value) {
    InterlockedAdd(buf[idx], value);
}

// Band-selected optical triplet of a face slot: absorptivity, specular
// reflectance, transmittance. The band comes from pc.flags (FLAG_SOLAR_BAND).
// Slots without a material (-1) are blackbodies: a=1, s=t=0, returns false.
public bool band_properties(uint slot, out float a, out float s, out float t) {
    int material_index = face_material[slot];
    if (material_index < 0) {
        a = 1.0;
        s = 0.0;
        t = 0.0;
        return false;
    }
    uint base = uint(material_index) * 6u +
                (((pc.flags & FLAG_SOLAR_BAND) != 0u) ? 3u : 0u);
    a = materials[base];
    s = materials[base + 1u];
    t = materials[base + 2u];
    return true;
}

// ---------------------------------------------------------------------------
// Geometry helpers (published reference implementations — do not re-derive).
// ---------------------------------------------------------------------------

// Branchless orthonormal basis (Duff et al. 2017). Sign subtlety at
// n.z ~ -1 is handled by `s`; do not substitute a cross(n, up) version.
public void onb(float3 n, out float3 t, out float3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

// Cosine-weighted hemisphere about n (2 draws).
public float3 cosine_hemisphere(float3 n, inout Rng rng) {
    float u1 = rng_next(rng);
    float u2 = rng_next(rng);
    float phi = 6.28318530718 * u1;
    float r = sqrt(u2);
    float3 t;
    float3 b;
    onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) +
                     n * sqrt(max(1.0 - u2, 0.0)));
}

// Self-intersection avoidance: rays start at the hit point pushed out along
// the geometric normal by an epsilon that scales with the distance from the
// origin, so it works both at spacecraft scale (meters) and at
// planet-distance scale (1e7 m) where float32 positions are coarse.
// TODO(radiative): if validation scenes show light leaks on thin geometry,
// upgrade to an adaptive integer-offset scheme.
public float3 offset_origin(float3 hit_p, float3 n_geom) {
    float eps = pc.ray_tmin_scale * max(1.0, length(hit_p));
    return hit_p + n_geom * eps;
}

// ---------------------------------------------------------------------------
// Emission sampling. FROZEN draw order: u0 triangle pick, u1/u2 barycentric
// point, u3/u4 direction (the direction draws are skipped in normal-emission
// mode).
// ---------------------------------------------------------------------------
public struct EmittedRay {
    public float3 origin;
    public float3 direction;
};

// A uniformly-sampled surface point with the side-resolved normal. The
// position is NOT offset yet — callers push it out with offset_origin along
// whatever side they trace from.
public struct SurfaceSample {
    public float3 position;
    public float3 normal;
};

// Draws u0 (triangle pick) + u1/u2 (barycentric point) — the first three
// draws of the frozen emission order.
public SurfaceSample sample_face_point(uint face_slot, inout Rng rng) {
    uint pair_base = face_slot & ~1u;  // even side-1 slot of the pair

    // 1. Area-weighted triangle pick: binary search u0 in the face's
    //    cumulative-area table (normalized to 1.0 at the last triangle).
    float u0 = rng_next(rng);
    uint lo = emit_tri_offset[pair_base];
    uint hi = emit_tri_offset[pair_base + 1u];  // one past the last triangle
    while (lo + 1u < hi) {
        uint mid = (lo + hi) / 2u;
        if (u0 < emit_cum_area[mid - 1u]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    uint part = emit_tri_part[lo];
    uint prim = emit_tri_prim[lo];

    InstanceData inst = instance_data[part];
    uint3 *indices = (uint3 *)inst.indices_addr;
    float3 *vertices = (float3 *)inst.vertices_addr;
    uint3 tri = indices[prim];
    float3 v0 = vertices[tri.x];
    float3 v1 = vertices[tri.y];
    float3 v2 = vertices[tri.z];

    // 2. Uniform barycentric point (sqrt trick, 2 draws).
    float u1 = rng_next(rng);
    float u2 = rng_next(rng);
    float s = sqrt(u1);
    float b0 = 1.0 - s;
    float b1 = s * (1.0 - u2);
    float b2 = s * u2;
    float3 p_local = b0 * v0 + b1 * v1 + b2 * v2;
    float3 p_world = transform_point(inst, p_local);

    // 3. Geometric normal, world-rotated. Winding: n points off side 1
    //    (even slot); odd slots (side 2) flip.
    float3 n_local = normalize(cross(v1 - v0, v2 - v0));
    float3 n = normalize(transform_direction(inst, n_local));
    if ((face_slot & 1u) != 0u) {
        n = -n;
    }

    SurfaceSample sample;
    sample.position = p_world;
    sample.normal = n;
    return sample;
}

public EmittedRay emit_from_face(uint face_slot, inout Rng rng) {
    SurfaceSample sample = sample_face_point(face_slot, rng);

    // 4. Direction: cosine hemisphere (2 draws) or the normal itself.
    float3 dir;
    if ((pc.flags & FLAG_NORMAL_EMISSION) != 0u) {
        dir = sample.normal;
    } else {
        dir = cosine_hemisphere(sample.normal, rng);
    }

    EmittedRay ray;
    ray.origin = offset_origin(sample.position, sample.normal);
    ray.direction = dir;
    return ray;
}

// ---------------------------------------------------------------------------
// Hit resolution — the ONLY place face-side parity is decided (front face
// <=> side 1 <=> even slot). Emission uses the same winding convention; a
// sign mismatch between the two produces view-factor matrices that are
// almost right, which is why parity lives in exactly one function.
// ---------------------------------------------------------------------------
public struct Hit {
    public bool miss;
    public uint slot;      // global face slot (side-resolved)
    public float3 normal;  // geometric, flipped so dot(n, ray_dir) < 0
    public float3 position;
};

public Hit trace_ray(float3 origin, float3 direction) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 3.4e38;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    q.Proceed();

    Hit hit;
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        hit.miss = true;
        hit.slot = 0u;
        hit.normal = float3(0.0, 0.0, 1.0);
        hit.position = origin;
        return hit;
    }

    uint inst_index = q.CommittedInstanceIndex();
    uint prim = q.CommittedPrimitiveIndex();
    InstanceData inst = instance_data[inst_index];
    uint *tri_face = (uint *)inst.tri_face_addr;
    uint base = tri_face[prim];  // even side-1 slot

    hit.miss = false;
    hit.position = origin + q.CommittedRayT() * direction;

    uint3 *indices = (uint3 *)inst.indices_addr;
    float3 *vertices = (float3 *)inst.vertices_addr;
    uint3 tri = indices[prim];
    float3 n_local = normalize(
        cross(vertices[tri.y] - vertices[tri.x],
              vertices[tri.z] - vertices[tri.x]));
    float3 n = normalize(transform_direction(inst, n_local));

    // Front-facing is derived from the winding normal instead of asking the ray
    // query, because the two are equivalent here and the intrinsic is not
    // portable: front face <=> the winding normal opposes the ray. The
    // equivalence needs the instance transform to preserve orientation, which
    // holds because ScenePart transforms are rigid (rotation + translation,
    // determinant +1) and transform_direction applies rotation only; a
    // mirroring transform would invert the test.
    // (The Slang 2026.12.2 Metal backend also emits the wrong MSL name for
    // RayQuery::CommittedTriangleFrontFace — get_ instead of is_ — so avoiding
    // it sidesteps an upstream bug at no cost.)
    bool front = dot(n, direction) < 0.0;
    hit.slot = front ? base : base + 1u;
    hit.normal = front ? n : -n;
    return hit;
}
