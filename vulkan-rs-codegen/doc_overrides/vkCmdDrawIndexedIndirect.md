# Usage Notes

Records one or more indexed draw calls whose parameters are read
from a GPU buffer. Each `DrawIndexedIndirectCommand` contains
`index_count`, `instance_count`, `first_index`, `vertex_offset`,
and `first_instance`.

This is the indexed counterpart to `cmd_draw_indirect` and the most
common indirect draw call for GPU-driven rendering pipelines. A
compute shader performs culling and writes surviving draw commands
into the buffer; the GPU then draws them without CPU involvement.

The buffer must have `BUFFER_USAGE_INDIRECT_BUFFER`. The `stride`
must be at least `sizeof(DrawIndexedIndirectCommand)` (20 bytes)
and a multiple of 4.

For dynamic draw counts, use `cmd_draw_indexed_indirect_count`
(Vulkan 1.2).
