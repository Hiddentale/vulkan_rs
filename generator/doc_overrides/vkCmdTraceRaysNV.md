# Usage Notes

Dispatches a ray tracing workload using the NV ray tracing
pipeline. Takes shader binding table buffer/offset/stride for
each shader stage (raygen, miss, closest hit, callable) and the
dispatch dimensions.

This is the legacy NV path; prefer `cmd_trace_rays_khr` for new
code.

Requires `VK_NV_ray_tracing`.
