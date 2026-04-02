# Usage Notes

Dispatches rays into the scene. This is the primary ray tracing
dispatch command, the ray tracing equivalent of `cmd_draw` or
`cmd_dispatch`.

Each of the four shader binding table (SBT) regions points to a
device memory region containing shader group handles:

- **Raygen**: exactly one entry, the ray generation shader.
- **Miss**: shaders invoked when a ray hits nothing.
- **Hit**: shader groups invoked on ray-geometry intersection.
- **Callable**: shaders invoked explicitly from other stages.

The `width`, `height`, and `depth` parameters define the 3D launch
dimensions. Each invocation gets a unique `gl_LaunchIDEXT`. For
a fullscreen ray trace, use the render target resolution with
`depth = 1`.

The SBT entries must be built from handles retrieved with
`get_ray_tracing_shader_group_handles_khr`, stored in a buffer
with `BUFFER_USAGE_SHADER_BINDING_TABLE`. Each region's `stride`
must be a multiple of `shaderGroupHandleAlignment` and the base
address must be aligned to `shaderGroupBaseAlignment`.
