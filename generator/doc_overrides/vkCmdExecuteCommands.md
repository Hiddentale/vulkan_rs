# Usage Notes

Executes one or more secondary command buffers from a primary
command buffer. The secondary buffers are inlined into the primary
buffer's execution stream in array order.

**Use cases**:

- **Multi-threaded recording**: each thread records a secondary
  command buffer, and the main thread assembles them with a single
  `cmd_execute_commands` call. This is the primary scaling strategy
  for CPU-bound recording.
- **Reusable draw sequences**: record a secondary buffer once and
  execute it in multiple frames or from multiple primary buffers
  (requires `SIMULTANEOUS_USE` on the secondary buffer).

Secondary command buffers inherit certain state from the primary
buffer (viewport, scissor, etc.) only if declared in the
`CommandBufferInheritanceInfo`. The render pass and subpass must
match what the primary buffer is currently in.

This command can only be called from a primary command buffer.
