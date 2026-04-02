# Usage Notes

Draws indexed geometry with both the draw parameters and the draw
**count** read from GPU buffers. The `count_buffer` contains a
`u32` that limits how many `DrawIndexedIndirectCommand` entries from
the main buffer are actually executed, up to `max_draw_count`.

This is the key primitive for GPU-driven rendering pipelines: a
compute shader fills an indirect buffer and writes the surviving
draw count after culling. The CPU does not need to know the count.

The main buffer must have `BUFFER_USAGE_INDIRECT_BUFFER`. The count
buffer must also have `BUFFER_USAGE_INDIRECT_BUFFER`. The count
offset must be a multiple of 4.

Core in Vulkan 1.2. Previously available via
`VK_KHR_draw_indirect_count`.
