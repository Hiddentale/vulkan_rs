# Usage Notes

Records one or more non-indexed draw calls whose parameters are read
from a GPU buffer at execution time rather than baked into the
command buffer.

Each `DrawIndirectCommand` in the buffer contains `vertex_count`,
`instance_count`, `first_vertex`, and `first_instance`, the same
parameters as `cmd_draw`.

**Use cases**:

- **GPU-driven rendering**: a compute shader fills the indirect
  buffer with draw parameters (e.g. after culling), and the GPU
  draws without CPU round-trips.
- **Conditional draw counts**: pair with
  `cmd_draw_indirect_count` (Vulkan 1.2) to have the GPU also
  determine how many draws to execute.

The buffer must have been created with
`BUFFER_USAGE_INDIRECT_BUFFER`. The `stride` between commands must
be at least `sizeof(DrawIndirectCommand)` (16 bytes) and a
multiple of 4.
