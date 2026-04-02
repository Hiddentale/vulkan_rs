# Usage Notes

Overrides the default ray tracing pipeline stack size for the
bound pipeline. The stack is scratch memory used during shader
execution and recursion.

The default stack size (set at pipeline creation) assumes
worst-case recursion depth across all shader groups. If your
application uses a lower effective recursion depth or only a
subset of shader groups, setting a smaller stack size reduces
per-invocation memory usage and may improve occupancy.

Compute the required size by querying individual shader
contributions with `get_ray_tracing_shader_group_stack_size_khr`
and applying the recursion formula from the spec.

This is a dynamic state command, it takes effect for subsequent
`cmd_trace_rays_khr` calls within the same command buffer.
Binding a new pipeline resets the stack size to the pipeline's
default.
