# Usage Notes

Writes acceleration structure properties (such as compacted size)
into a query pool. Use this after a build to determine the
compacted size before calling `cmd_copy_acceleration_structure_nv`
with `COMPACT` mode.

Requires `VK_NV_ray_tracing`.
