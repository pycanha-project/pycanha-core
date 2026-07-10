#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "pycanha-core/radiative/device.hpp"

namespace rad = pycanha::radiative;

TEST_CASE("radiative device: graceful error when unavailable",
          "[radiative][device]") {
    if (rad::is_available()) {
        SUCCEED("RT-capable device present: covered by the creation test");
        return;
    }
    // No driver / no capable device / stub build: a clear construction
    // error instead of a crash.
    REQUIRE_THROWS_AS(rad::Device::create(), std::runtime_error);
}

TEST_CASE("radiative device: enumeration lists devices",
          "[radiative][device]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const auto devices = rad::enumerate_devices();
    REQUIRE_FALSE(devices.empty());
    REQUIRE_FALSE(devices.front().name.empty());
}

TEST_CASE("radiative device: default creation picks an RT device",
          "[radiative][device]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const rad::Device device = rad::Device::create();
    REQUIRE(device.info().ray_tracing);
    REQUIRE(device.info().max_dispatch_rays > 0);
}

TEST_CASE("radiative device: explicit bad index throws",
          "[radiative][device]") {
    REQUIRE_THROWS_AS(rad::Device::create(9999), std::runtime_error);
}

TEST_CASE("radiative device: memory budget positive", "[radiative][device]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable Vulkan device: skipped");
        return;
    }
    const rad::Device device = rad::Device::create();
    REQUIRE(device.memory_budget() > 0);
}
