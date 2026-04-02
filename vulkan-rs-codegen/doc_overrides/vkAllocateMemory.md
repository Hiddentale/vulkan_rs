# Usage Notes

Memory allocation in Vulkan is expensive. Prefer sub-allocating from
large blocks using a memory allocator (e.g., `gpu-allocator`) rather
than calling this for every buffer or image.

The returned `DeviceMemory` must be freed with `free_memory` when no
longer needed. Vulkan does not garbage-collect device memory.

# Guide

See [Memory Management](https://hiddentale.github.io/vulkan_rs/concepts/memory.html) in the vulkan_rs guide.
