# Usage Notes

Fully indirect ray dispatch, both the shader binding table
addresses and the launch dimensions are read from a GPU buffer.
This is the most flexible ray tracing dispatch.

The `indirect_device_address` points to a
`TraceRaysIndirectCommand2KHR` struct on the device, which
contains all four SBT regions (raygen, miss, hit, callable) plus
the `width`, `height`, and `depth`.

This allows the GPU to dynamically select which shaders to use
and how many rays to launch, enabling advanced techniques like
GPU-driven material sorting or multi-pass ray tracing without
CPU synchronization.

Provided by `VK_KHR_ray_tracing_maintenance1`, not the base
ray tracing pipeline extension. Requires the
`rayTracingPipelineTraceRaysIndirect2` feature.
