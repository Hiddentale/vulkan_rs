# Usage Notes

Launches a compute shader with the given number of workgroups in
X, Y, and Z dimensions. The total number of invocations is
`group_count * local_size` (defined in the shader's `local_size_x`,
`local_size_y`, `local_size_z`).

A compute pipeline must be bound before calling this. Compute
dispatches can be recorded outside a render pass.

**Sizing**: to process an image of `width × height` pixels with a
local size of 16×16, dispatch
`ceil(width / 16) × ceil(height / 16) × 1` workgroups.

**Limits**: each dimension is capped by
`max_compute_work_group_count` (at least 65535 per axis). The total
invocations per workgroup are capped by
`max_compute_work_group_invocations` (at least 128).

For GPU-driven dispatch counts, use `cmd_dispatch_indirect`.
