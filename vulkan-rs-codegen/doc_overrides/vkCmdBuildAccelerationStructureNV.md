# Usage Notes

Builds or updates an NV acceleration structure from geometry data.
Set `update` to non-zero to refit an existing structure in place
(faster but lower quality than a full rebuild).

A scratch buffer is required; query its size with
`get_acceleration_structure_memory_requirements_nv`.

Requires `VK_NV_ray_tracing`.
