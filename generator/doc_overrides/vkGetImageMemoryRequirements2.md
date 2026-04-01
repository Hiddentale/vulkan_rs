# Usage Notes

Vulkan 1.1 version of `get_image_memory_requirements` that supports
extensible output structs via pNext.

Chain `MemoryDedicatedRequirements` to query whether the driver
prefers or requires a dedicated allocation for this image. Dedicated
allocations are common for large render targets and swapchain-sized
images, some drivers require them.

For multi-planar images, chain `ImagePlaneMemoryRequirementsInfo` in
the input to query requirements for a specific plane.

The base `MemoryRequirements` is identical to what
`get_image_memory_requirements` returns.
