# Usage Notes

Draws non-indexed geometry with both the draw parameters and the
draw **count** read from GPU buffers. The non-indexed counterpart to
`cmd_draw_indexed_indirect_count`.

See `cmd_draw_indexed_indirect_count` for a full explanation of the
GPU-driven rendering pattern. The only difference is that this
command reads `DrawIndirectCommand` entries instead of
`DrawIndexedIndirectCommand`.

Core in Vulkan 1.2. Previously available via
`VK_KHR_draw_indirect_count`.
