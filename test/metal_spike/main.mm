// Metal backend spike (M0 steps B and C) — see roadmap/21-metal-spike-report.md.
//
// This program is a GATE, not a test of pycanha-core: it deliberately links
// nothing but Metal and Foundation, so a failure here means "this Mac cannot
// run the radiative engine" rather than "the radiative engine is broken". It
// answers three questions:
//
//   1. Does this device meet the backend's requirements (ray tracing plus GPU
//      family Apple9, which is where Metal gained the full set of 64-bit
//      buffer atomics that the fixed-point deposits need)?
//   2. Do the three PRODUCTION kernels (vf / exchange / solar) survive the
//      whole toolchain — Slang -> MSL -> .metallib -> MTLComputePipelineState?
//      Creating the pipeline state is the real check: it is where Metal
//      validates the entry point and its buffer bindings.
//   3. Does the runtime actually execute inline ray tracing and 64-bit atomic
//      adds correctly, via the self-contained probe.slang kernel?
//
// Exits 0 only if every gate passes.

#import <Metal/Metal.h>

#include <dispatch/dispatch.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "pycanha-core/radiative/kernels/exchange_metallib.h"
#include "pycanha-core/radiative/kernels/probe_bindings.h"
#include "pycanha-core/radiative/kernels/probe_metallib.h"
#include "pycanha-core/radiative/kernels/solar_metallib.h"
#include "pycanha-core/radiative/kernels/vf_metallib.h"

namespace {

// Mirrors PROBE_THREADS / PROBE_INCREMENT in probe.slang. The host verifies
// exact sums, so these must stay in step with the kernel.
constexpr std::uint32_t probe_threads = 128;
constexpr std::uint64_t probe_increment = 4294967296ULL;  // 2^32

// Slang keeps the entry-point name for Metal (unlike the SPIR-V path, where it
// renames it to "main").
constexpr const char* entry_point = "csMain";

int failures = 0;

void gate(bool ok, const std::string& what, const std::string& detail = {}) {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(),
                detail.empty() ? "" : " — ", detail.c_str());
    if (!ok) {
        ++failures;
    }
}

// Wraps embedded .metallib bytes for newLibraryWithData:. The DEFAULT
// destructor makes dispatch_data_create copy the buffer, so the caller does not
// have to keep anything alive.
dispatch_data_t embedded_library(const std::uint8_t* bytes, std::size_t size) {
    return dispatch_data_create(bytes, size, nullptr,
                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
}

// Gate 2: the full shader toolchain, per kernel.
void check_pipeline(id<MTLDevice> device, const char* name,
                    const std::uint8_t* bytes, std::size_t size) {
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithData:embedded_library(bytes, size) error:&error];
    if (library == nil) {
        gate(false, std::string("load ") + name + ".metallib",
             error != nil ? error.localizedDescription.UTF8String : "unknown");
        return;
    }
    id<MTLFunction> function =
        [library newFunctionWithName:@(entry_point)];
    if (function == nil) {
        gate(false, std::string(name) + ": entry point " + entry_point,
             "not found in the library");
        return;
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        gate(false, std::string(name) + ": pipeline state",
             error != nil ? error.localizedDescription.UTF8String : "unknown");
        return;
    }
    gate(true, std::string(name) + ": metallib -> pipeline state",
         "max threads/threadgroup = " +
             std::to_string(pipeline.maxTotalThreadsPerThreadgroup));
}

struct Scene {
    id<MTLAccelerationStructure> primitive = nil;
    id<MTLAccelerationStructure> instance = nil;
};

// Two triangles forming the unit quad at z = 0, wrapped in a one-instance
// TLAS — the same BLAS-per-part / TLAS-of-instances shape the real scene uses.
Scene build_scene(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    // packed float3 (12-byte stride), matching what the kernels expect.
    const std::array<float, 12> vertices = {0.0F, 0.0F, 0.0F,  //
                                            1.0F, 0.0F, 0.0F,  //
                                            0.0F, 1.0F, 0.0F,  //
                                            1.0F, 1.0F, 0.0F};
    const std::array<std::uint32_t, 6> indices = {0, 1, 2, 1, 3, 2};

    id<MTLBuffer> vertex_buffer =
        [device newBufferWithBytes:vertices.data()
                            length:vertices.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> index_buffer =
        [device newBufferWithBytes:indices.data()
                            length:indices.size() * sizeof(std::uint32_t)
                           options:MTLResourceStorageModeShared];

    MTLAccelerationStructureTriangleGeometryDescriptor* geometry =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geometry.vertexBuffer = vertex_buffer;
    geometry.vertexStride = 3 * sizeof(float);
    geometry.indexBuffer = index_buffer;
    geometry.indexType = MTLIndexTypeUInt32;
    geometry.triangleCount = 2;

    MTLPrimitiveAccelerationStructureDescriptor* primitive_descriptor =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    primitive_descriptor.geometryDescriptors = @[ geometry ];

    // Identity transform. MTLPackedFloat4x3 is COLUMN-major 4x3 (columns[3] is
    // the translation), the transpose of Vulkan's row-major 3x4 — the single
    // conversion point the real backend will need a unit test for. Fields are
    // assigned one at a time because MTLPackedFloat3 wraps an anonymous union,
    // where brace initialisation is not portable; the {} above already zeroed
    // everything, including the translation column.
    MTLAccelerationStructureInstanceDescriptor instance_descriptor{};
    instance_descriptor.transformationMatrix.columns[0].x = 1.0F;
    instance_descriptor.transformationMatrix.columns[1].y = 1.0F;
    instance_descriptor.transformationMatrix.columns[2].z = 1.0F;
    instance_descriptor.options = MTLAccelerationStructureInstanceOptionOpaque;
    instance_descriptor.mask = 0xFF;
    instance_descriptor.intersectionFunctionTableOffset = 0;
    instance_descriptor.accelerationStructureIndex = 0;

    id<MTLBuffer> instance_buffer =
        [device newBufferWithBytes:&instance_descriptor
                            length:sizeof(instance_descriptor)
                           options:MTLResourceStorageModeShared];

    Scene scene;

    const MTLAccelerationStructureSizes primitive_sizes =
        [device accelerationStructureSizesWithDescriptor:primitive_descriptor];
    scene.primitive = [device
        newAccelerationStructureWithSize:primitive_sizes.accelerationStructureSize];
    id<MTLBuffer> primitive_scratch =
        [device newBufferWithLength:primitive_sizes.buildScratchBufferSize
                           options:MTLResourceStorageModePrivate];

    MTLInstanceAccelerationStructureDescriptor* instance_as_descriptor =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    instance_as_descriptor.instancedAccelerationStructures =
        @[ scene.primitive ];
    instance_as_descriptor.instanceCount = 1;
    instance_as_descriptor.instanceDescriptorBuffer = instance_buffer;

    const MTLAccelerationStructureSizes instance_sizes =
        [device accelerationStructureSizesWithDescriptor:instance_as_descriptor];
    scene.instance = [device
        newAccelerationStructureWithSize:instance_sizes.accelerationStructureSize];
    id<MTLBuffer> instance_scratch =
        [device newBufferWithLength:instance_sizes.buildScratchBufferSize
                           options:MTLResourceStorageModePrivate];

    // The instance build reads the primitive structure, so they cannot share
    // one encoder without a barrier; two encoders in one command buffer is the
    // simplest correct ordering.
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> primitive_encoder =
        [command_buffer accelerationStructureCommandEncoder];
    [primitive_encoder buildAccelerationStructure:scene.primitive
                                      descriptor:primitive_descriptor
                                   scratchBuffer:primitive_scratch
                             scratchBufferOffset:0];
    [primitive_encoder endEncoding];
    id<MTLAccelerationStructureCommandEncoder> instance_encoder =
        [command_buffer accelerationStructureCommandEncoder];
    [instance_encoder buildAccelerationStructure:scene.instance
                                     descriptor:instance_as_descriptor
                                  scratchBuffer:instance_scratch
                            scratchBufferOffset:0];
    [instance_encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.error != nil) {
        gate(false, "acceleration structure build",
             command_buffer.error.localizedDescription.UTF8String);
        return Scene{};
    }
    gate(true, "acceleration structure build",
         "BLAS " + std::to_string(primitive_sizes.accelerationStructureSize) +
             " B, TLAS " +
             std::to_string(instance_sizes.accelerationStructureSize) + " B");
    return scene;
}

// Gate 3: run probe.slang and check the ray-query outcomes and the 64-bit sum.
void run_probe(id<MTLDevice> device, id<MTLCommandQueue> queue,
               const Scene& scene) {
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithData:embedded_library(pycanha::radiative::kernels::
                                                       probe_metallib,
                                                   pycanha::radiative::kernels::
                                                       probe_metallib_size)
                            error:&error];
    id<MTLFunction> function = [library newFunctionWithName:@(entry_point)];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        gate(false, "probe pipeline state",
             error != nil ? error.localizedDescription.UTF8String : "unknown");
        return;
    }

    id<MTLBuffer> outcomes =
        [device newBufferWithLength:probe_threads * sizeof(std::uint32_t)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> counter =
        [device newBufferWithLength:sizeof(std::uint64_t)
                           options:MTLResourceStorageModeShared];
    std::memset(outcomes.contents, 0, outcomes.length);
    std::memset(counter.contents, 0, counter.length);

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setAccelerationStructure:scene.instance
                       atBufferIndex:pycanha::radiative::kernels::
                                         probe_binding_tlas];
    [encoder setBuffer:outcomes
                offset:0
               atIndex:pycanha::radiative::kernels::probe_binding_outcomes];
    [encoder setBuffer:counter
                offset:0
               atIndex:pycanha::radiative::kernels::probe_binding_counter];
    // REQUIRED and with no Vulkan counterpart: a TLAS does not make the BLASes
    // it references resident. Without this the traversal reads unmapped memory
    // and silently reports misses instead of faulting.
    [encoder useResource:scene.primitive usage:MTLResourceUsageRead];
    [encoder dispatchThreads:MTLSizeMake(probe_threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.error != nil) {
        gate(false, "probe dispatch",
             command_buffer.error.localizedDescription.UTF8String);
        return;
    }

    // Even threads aim at primitive 0 (outcome 1); odd threads miss (0).
    const auto* results = static_cast<const std::uint32_t*>(outcomes.contents);
    std::uint32_t bad = 0;
    for (std::uint32_t i = 0; i < probe_threads; ++i) {
        const std::uint32_t expected = (i % 2 == 0) ? 1U : 0U;
        if (results[i] != expected) {
            ++bad;
        }
    }
    gate(bad == 0, "inline ray tracing (hit/miss + primitive index)",
         bad == 0 ? std::to_string(probe_threads) + " rays correct"
                  : std::to_string(bad) + " of " +
                        std::to_string(probe_threads) + " rays wrong");

    const std::uint64_t total =
        *static_cast<const std::uint64_t*>(counter.contents);
    const std::uint64_t expected_total = probe_threads * probe_increment;
    gate(total == expected_total, "64-bit atomic add",
         "got " + std::to_string(total) + ", expected " +
             std::to_string(expected_total));
}

}  // namespace

int main() {
    std::printf("pycanha-core Metal backend spike\n\n");

    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    std::printf("Metal devices (%lu):\n",
                static_cast<unsigned long>(devices.count));

    id<MTLDevice> chosen = nil;
    for (id<MTLDevice> device in devices) {
        const bool ray_tracing = device.supportsRaytracing;
        const bool apple9 = [device supportsFamily:MTLGPUFamilyApple9];
        std::printf(
            "  %-40s raytracing=%-3s apple9=%-3s unified=%-3s budget=%llu MB\n",
            device.name.UTF8String, ray_tracing ? "yes" : "no",
            apple9 ? "yes" : "no", device.hasUnifiedMemory ? "yes" : "no",
            static_cast<unsigned long long>(device.recommendedMaxWorkingSetSize >>
                                            20));
        if (chosen == nil && ray_tracing && apple9) {
            chosen = device;
        }
    }
    std::printf("\n");

    std::printf("Gate 1 — device capability\n");
    gate(chosen != nil, "an Apple9 device with ray tracing",
         chosen != nil ? chosen.name.UTF8String
                       : "none found (M1/M2 and virtualised GPUs are expected "
                         "to fail here)");
    if (chosen == nil) {
        std::printf("\nRESULT: FAIL (%d gate(s))\n", failures);
        return 1;
    }

    std::printf("\nGate 2 — production kernels through the toolchain\n");
    namespace kernels = pycanha::radiative::kernels;
    check_pipeline(chosen, "vf", kernels::vf_metallib, kernels::vf_metallib_size);
    check_pipeline(chosen, "exchange", kernels::exchange_metallib,
                   kernels::exchange_metallib_size);
    check_pipeline(chosen, "solar", kernels::solar_metallib,
                   kernels::solar_metallib_size);

    std::printf("\nGate 3 — runtime ray tracing and 64-bit atomics\n");
    id<MTLCommandQueue> queue = [chosen newCommandQueue];
    const Scene scene = build_scene(chosen, queue);
    if (scene.instance != nil) {
        run_probe(chosen, queue, scene);
    }

    std::printf("\nRESULT: %s", failures == 0 ? "PASS\n" : "FAIL\n");
    if (failures != 0) {
        std::printf("%d gate(s) failed\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
