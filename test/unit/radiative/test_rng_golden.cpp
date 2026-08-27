#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

// CPU reference of the counter-based PCG stream in
// src/radiative/kernels/common.slang, locked against frozen golden values.
// The determinism contract (dense == tiled bit-equality, chunking
// invariance, reproducible seeds) rests on this exact algorithm: any edit
// to the shader RNG must consciously update both this reference and the
// goldens, invalidating every stored reference result.

namespace {

[[nodiscard]] std::uint32_t pcg_hash(std::uint32_t x) {
    x = (x * 747796405U) + 2891336453U;
    const std::uint32_t w = ((x >> ((x >> 28U) + 4U)) ^ x) * 277803737U;
    return (w >> 22U) ^ w;
}

[[nodiscard]] std::uint32_t rng_init(std::uint32_t face,
                                     std::uint32_t ray_index,
                                     std::uint32_t batch_seed) {
    return pcg_hash(face ^ pcg_hash(ray_index ^ pcg_hash(batch_seed)));
}

[[nodiscard]] float rng_next(std::uint32_t& state) {
    state = (state * 747796405U) + 2891336453U;
    std::uint32_t w = ((state >> ((state >> 28U) + 4U)) ^ state) * 277803737U;
    w = (w >> 22U) ^ w;
    return static_cast<float>(w >> 8U) * (1.0F / 16777216.0F);
}

struct Golden {
    std::uint32_t face;
    std::uint32_t ray_index;
    std::uint32_t batch_seed;
    std::uint32_t state;
    std::array<double, 4> draws;
};

void check_golden(const Golden& golden) {
    std::uint32_t state =
        rng_init(golden.face, golden.ray_index, golden.batch_seed);
    REQUIRE(state == golden.state);
    for (const double expected : golden.draws) {
        const float draw = rng_next(state);
        REQUIRE(static_cast<double>(draw) == Catch::Approx(expected));
        REQUIRE(draw >= 0.0F);
        REQUIRE(draw < 1.0F);
    }
}

}  // namespace

TEST_CASE("radiative rng: golden values freeze the counter-based stream",
          "[radiative][rng]") {
    check_golden(Golden{.face = 0,
                        .ray_index = 0,
                        .batch_seed = 0,
                        .state = 0x7fddb461U,
                        .draws = {0.55154848098754883, 0.61909717321395874,
                                  0.30766761302947998, 0.080833792686462402}});
    check_golden(Golden{.face = 2,
                        .ray_index = 7,
                        .batch_seed = 42,
                        .state = 0x75d6bffdU,
                        .draws = {0.91395241022109985, 0.4374813437461853,
                                  0.73279541730880737, 0.91659349203109741}});
    check_golden(
        Golden{.face = 1000,
               .ray_index = 123456,
               .batch_seed = 7,
               .state = 0x85989a5bU,
               .draws = {0.022035539150238037, 0.34756523370742798,
                         0.028152227401733398, 0.0030736923217773438}});
}

TEST_CASE("radiative rng: streams are keyed, not sequential",
          "[radiative][rng]") {
    // Neighboring keys must decorrelate: the stream depends only on the
    // (face, ray, seed) triple, so swapping components changes everything.
    const std::uint32_t base = rng_init(1, 2, 3);
    REQUIRE(base != rng_init(2, 1, 3));
    REQUIRE(base != rng_init(1, 3, 2));
    REQUIRE(base != rng_init(3, 2, 1));
    REQUIRE(base != rng_init(1, 2, 4));
}
