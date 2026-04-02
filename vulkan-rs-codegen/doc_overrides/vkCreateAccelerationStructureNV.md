# Usage Notes

Creates an NV ray tracing acceleration structure. This is the
legacy NV path; prefer `create_acceleration_structure_khr` for
new code.

The NV acceleration structure owns its memory implicitly, bind
memory with `bind_acceleration_structure_memory_nv`. Destroy with
`destroy_acceleration_structure_nv`.

Requires `VK_NV_ray_tracing`.
