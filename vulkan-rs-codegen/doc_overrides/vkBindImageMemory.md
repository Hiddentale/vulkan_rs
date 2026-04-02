# Usage Notes

Binds a `DeviceMemory` allocation (or a region of one) to an image.
Must be called after `create_image` and before the image is used.

**Requirements**:

1. Call `get_image_memory_requirements` first.
2. The `memory_type_index` used for allocation must have a bit set
   in the returned `memory_type_bits` mask.
3. `memory_offset` must be a multiple of the returned `alignment`.

**Dedicated allocations**: some drivers perform better when certain
images (especially swapchain-sized color or depth targets) have their
own allocation. Query `get_image_memory_requirements2` with
`MemoryDedicatedRequirements` to check whether the driver prefers or
requires a dedicated allocation.

**Sub-allocation**: like buffers, multiple images can share one
allocation at different offsets. Respect alignment from the memory
requirements, and note that linear and optimal-tiling images may
need `buffer_image_granularity` spacing between them.

Once bound, the memory binding is permanent. Destroy the image
before freeing its backing memory.
