// VulkanMemoryAllocator implementation translation unit. The configuration
// (VMA_STATIC_VULKAN_FUNCTIONS=0 / VMA_DYNAMIC_VULKAN_FUNCTIONS=1: volk
// provides all entry points, VMA loads everything dynamically) comes from
// target-wide compile definitions in CMake.

#include <volk.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
