# Usage Notes

Launches a compute shader with workgroup counts read from a GPU
buffer. The buffer contains a `DispatchIndirectCommand` with
`group_count_x`, `group_count_y`, `group_count_z`.

Use this when a prior compute pass determines how much work to do, 
for example, a culling pass writes the surviving workgroup count
for a subsequent processing pass.

The buffer must have `BUFFER_USAGE_INDIRECT_BUFFER`. The offset must
be a multiple of 4.

The same workgroup count limits apply as for `cmd_dispatch`.
