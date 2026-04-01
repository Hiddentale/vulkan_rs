# Usage Notes

Retrieves opaque shader group handles from a ray tracing pipeline.
These handles are copied into GPU-visible buffers to build the
**shader binding table** (SBT) that `cmd_trace_rays_khr` indexes
into.

Each handle is `shaderGroupHandleSize` bytes (query from
`PhysicalDeviceRayTracingPipelinePropertiesKHR`, typically 32
bytes). The `p_data` buffer must be at least
`group_count * shaderGroupHandleSize` bytes.

`first_group` and `group_count` index into the `groups` array
from `RayTracingPipelineCreateInfoKHR`. Handles are written
sequentially, group `first_group` first, then
`first_group + 1`, and so on.

After retrieving handles, copy them into a buffer with
`BUFFER_USAGE_SHADER_BINDING_TABLE` at the correct stride and
alignment for each SBT region.
